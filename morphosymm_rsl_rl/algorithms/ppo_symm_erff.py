# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import chain
import escnn
import torch
import torch.nn as nn
from escnn.group import Group
from escnn.nn import FieldType
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.extensions.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from morphosymm_rsl_rl.modules.ac_symm import SymmModel
from morphosymm_rsl_rl.storage import PrioritizedReplayBuffer, RunningStdScaler
from morphosymm_rsl_rl.symm_utils import configure_observation_space_representations
from morphosymm_rsl_rl.rff import (
    EquivariantRandomFourierFeatures,
    RunningLatentNormalizer,
    EquivariantKoopmanEstimator
)


class _AugmentedRolloutStorage(RolloutStorage):
    """Rollout storage over symmetry-augmented transitions.

    Tags identity-replica rows in each mini-batch so the adaptive learning-rate schedule
    evaluates genuine policy drift rather than the symmetry representation gap.
    """

    def __init__(self, *args, num_original_envs: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_original_envs = num_original_envs
        self.last_batch_is_identity: torch.Tensor | None = None

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        is_identity_per_step = torch.arange(self.num_envs, device=self.device) < self.num_original_envs
        is_identity_flat = is_identity_per_step.repeat(self.num_transitions_per_env)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                self.last_batch_is_identity = is_identity_flat[batch_idx]

                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )


def _identity_only_kl_divergence(actor: object, storage: _AugmentedRolloutStorage) -> None:
    """Patch actor.get_kl_divergence to only consider identity replicas."""
    original_get_kl_divergence = getattr(actor, "get_kl_divergence", None)
    if original_get_kl_divergence is None:
        return

    def get_kl_divergence(old_distribution_params, new_distribution_params):
        kl = original_get_kl_divergence(old_distribution_params, new_distribution_params)
        mask = storage.last_batch_is_identity
        if mask is None or not torch.any(mask):
            return kl
        return kl.new_full(kl.shape, kl[mask].mean())

    actor.get_kl_divergence = get_kl_divergence


class PPOSymmERFF:
    """Equivariant Proximal Policy Optimization with Online ERFF integration (rsl_rl v5.4.2 style)."""

    actor: SymmModel
    critic: SymmModel
    storage: _AugmentedRolloutStorage
    G: Group
    num_replica: int
    actor_in_type: FieldType
    critic_in_type: FieldType
    actor_out_type: FieldType
    state_type: FieldType

    def __init__(
        self,
        actor: SymmModel,
        critic: SymmModel,
        storage: _AugmentedRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        task: str = "erff_koopman",
        dt: float = 1.0,
        single_observation_space: int = 1,
        action_space: int = 1,
        history_length: int = 1,
        multi_gpu_cfg: dict | None = None,
        koopman_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if self.is_multi_gpu:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        self.symmetry = symmetry_cfg

        # Optimizer string resolution (v5.4.2 style)
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )
        self.learning_rate = learning_rate

        self.koopman_cfg = koopman_cfg or {}
        self.task = task
        self.dt = dt
        self.single_observation_space = single_observation_space
        self.history_length = history_length
        self.state_dim = single_observation_space
        self.action_dim = action_space

        self.replay_buffer = PrioritizedReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.koopman_cfg.get("beta_initial", 0.4),
            self.koopman_cfg.get("beta_annealing_steps", 1000),
            self.koopman_cfg.get("replay_buffer_size", 10000),
            device=self.device,
        )
        self.obs_action_normalizer = RunningStdScaler(self.state_dim, self.action_dim, device=self.device)

        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "PPOSymmERFF":
        """Construct the algorithm, resolving representations and models in v5.4.2 style."""
        alg_cfg = cfg["algorithm"]
        policy_cfg = cfg.get("policy", cfg.get("actor", {}))

        single_observation_space = env.cfg.single_observation_space
        history_length = env.cfg.history_length
        action_space = env.cfg.action_space
        dt = env.unwrapped.step_dt
        task = cfg["experiment_name"]

        koopman_cfg = cfg.get("koopman_cfg", {})
        symm_cfg = cfg.get("morphologycal_symmetries_cfg", {})

        # Setup observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        if "koopman_prediction" not in cfg["obs_groups"]["critic"]:
            cfg["obs_groups"]["critic"].append("koopman_prediction")
        koopman_dim = koopman_cfg.get("m", 129)
        obs.set("koopman_prediction", torch.zeros((obs["policy"].shape[0], koopman_dim), device=device))

        # 1. Resolve escnn symmetry representations cleanly (Like PPOSymmDAEOnline)
        all_space_names = list(
            dict.fromkeys(
                [
                    *symm_cfg["obs_space_names_actor"],
                    *symm_cfg["obs_space_names_critic"],
                    *symm_cfg["action_space_names"],
                ]
            )
        )
        G, representations = configure_observation_space_representations(
            symm_cfg["robot_name"], all_space_names, symm_cfg["joints_order"]
        )
        gspace = escnn.gspaces.no_base_space(G)
        actor_in_type = FieldType(gspace, [representations[n] for n in symm_cfg["obs_space_names_actor"]])
        critic_in_type = FieldType(gspace, [representations[n] for n in symm_cfg["obs_space_names_critic"]])
        state_type = FieldType(gspace, [representations[n] for n in symm_cfg["obs_space_names_single_state"]])
        actor_out_type = FieldType(gspace, [representations[n] for n in symm_cfg["action_space_names"]])
        num_replica = len(G.elements)

        # 2. Build Actor and Critic as SymmModels
        actor_class_name = cfg.get("actor", {}).pop("class_name", "SymmModel")
        critic_class_name = cfg.get("critic", {}).pop("class_name", "SymmModel")

        actor_class = SymmModel if actor_class_name == "SymmModel" else resolve_callable(actor_class_name)
        critic_class = SymmModel if critic_class_name == "SymmModel" else resolve_callable(critic_class_name)

        policy_copy = policy_cfg.copy()
        policy_copy.pop("class_name", None)
        actor_cfg = cfg.get("actor", policy_copy).copy()
        critic_cfg = cfg.get("critic", policy_copy).copy()

        for cfg_dict in [actor_cfg, critic_cfg]:
            cfg_dict.pop("class_name", None)

        actor_cfg.setdefault("hidden_dims", actor_cfg.pop("actor_hidden_dims", [256, 256, 256]))
        critic_cfg.setdefault("hidden_dims", critic_cfg.pop("critic_hidden_dims", [256, 256, 256]))
        actor_cfg.setdefault("obs_normalization", actor_cfg.pop("actor_obs_normalization", False))
        critic_cfg.setdefault("obs_normalization", critic_cfg.pop("critic_obs_normalization", False))

        actor_cfg.setdefault(
            "distribution_cfg",
            {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": actor_cfg.pop("init_noise_std", 1.0),
                "std_type": actor_cfg.pop("noise_std_type", "scalar"),
            },
        )

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg, **symm_cfg).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg, **symm_cfg).to(device)

        # 3. Create Augmented Rollout Storage
        tiled_obs = TensorDict(
            {key: PPOSymmERFF._tile(value, num_replica) for key, value in obs.items()},
            batch_size=[obs.batch_size[0] * num_replica],
            device=obs.device,
        )
        storage = _AugmentedRolloutStorage(
            "rl",
            env.num_envs * num_replica,
            cfg["num_steps_per_env"],
            tiled_obs,
            [env.num_actions],
            device,
            num_original_envs=env.num_envs,
        )

        alg_cfg.pop("class_name", None)
        alg_cfg.pop("share_cnn_encoders", None)

        alg = cls(
            actor=actor,
            critic=critic,
            storage=storage,
            device=device,
            task=task,
            dt=dt,
            single_observation_space=single_observation_space,
            action_space=action_space,
            history_length=history_length,
            koopman_cfg=koopman_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )

        # Attach explicit FieldTypes
        alg.G = G
        alg.num_replica = num_replica
        alg.actor_in_type = actor_in_type
        alg.critic_in_type = critic_in_type
        alg.actor_out_type = actor_out_type
        alg.state_type = state_type  # Specifically sized for the 60-dim DAE input

        # 4. Initialize ERFF and Koopman using the single state_type
        m_features = koopman_cfg.get('m', 129)
        group_order = G.order()
        num_reps = round(m_features / group_order)

        alg.rff = EquivariantRandomFourierFeatures(
            task=task,
            in_features=single_observation_space,
            in_type=alg.state_type,  # Safely injects the 60-dim FieldType
            m=num_reps,
            sigma=koopman_cfg.get('sigma', 1.0),
            kernel_type=koopman_cfg.get('kernel_type', 'gaussian')
        ).to(device)

        latent_dim = num_reps * group_order
        alg.latent_normalizer = RunningLatentNormalizer(num_features=latent_dim, device=device)

        if "koopman" in task:
            alg.koopman_estimator = EquivariantKoopmanEstimator(
                alg.rff,
                alg.latent_normalizer,
                action_type=alg.actor_out_type,
                gamma=koopman_cfg.get('gamma', 1.0),
                device=device
            )

        _identity_only_kl_divergence(alg.actor, alg.storage)
        return alg

    def get_policy(self) -> SymmModel:
        return self.actor

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def rff_predict(self, obs: TensorDict, action: torch.Tensor | None = None) -> torch.Tensor:
        critic_obs = obs["critic"]
        dae_input = critic_obs[
            :,
            self.single_observation_space * (self.history_length - 1) : self.single_observation_space * self.history_length,
        ]

        # Now safely casts using the 60-dimensional state_type
        dae_input = escnn.nn.GeometricTensor(dae_input, self.state_type)
        latent_raw = self.rff(dae_input)

        if action is not None and "koopman" in self.task:
            next_latent = self.koopman_estimator.predict_from_lifted_state(latent_raw, action)
        else:
            next_latent = latent_raw

        if hasattr(next_latent, 'tensor'):
            next_latent = next_latent.tensor
        return next_latent

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.actor.is_recurrent:
            self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())

        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        obs.set("koopman_prediction", self.rff_predict(obs, self.transition.actions))

        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self._augment_transition()

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def _augment_transition(self) -> None:
        """Replicate every field of the current transition across the symmetry group."""
        t = self.transition
        elements = self.G.elements[1:]

        t.observations = self._augment_observations(t.observations)
        t.actions = torch.cat(
            [t.actions] + [self.actor_out_type.transform_fibers(t.actions, g) for g in elements], dim=0
        )
        mean, std = t.distribution_params
        augmented_mean = torch.cat(
            [mean] + [self.actor_out_type.transform_fibers(mean, g) for g in elements], dim=0
        )
        augmented_std = torch.abs(
            torch.cat([std] + [self.actor_out_type.transform_fibers(std, g) for g in elements], dim=0)
        )
        t.distribution_params = (augmented_mean, augmented_std)
        t.actions_log_prob = self._tile(t.actions_log_prob, self.num_replica)

        t.values = self.critic(t.observations).detach()

        t.rewards = self._tile(t.rewards, self.num_replica)
        t.dones = self._tile(t.dones, self.num_replica)

    def _augment_observations(self, obs: TensorDict) -> TensorDict:
        """Transform each observation group across the symmetry group."""
        raw_actor, raw_critic = self._raw_actor, self._raw_critic
        augmented: dict[str, torch.Tensor] = {}
        for group_name, value in self._transform_obs_group(obs, raw_actor.obs_groups, self.actor_in_type).items():
            augmented[group_name] = value
        for group_name, value in self._transform_obs_group(
            obs, raw_critic.obs_groups, self.critic_in_type
        ).items():
            augmented.setdefault(group_name, value)
        for key in obs.keys():
            if key not in augmented:
                augmented[key] = self._tile(obs[key], self.num_replica)

        batch_size = next(iter(augmented.values())).shape[0]
        return TensorDict(augmented, batch_size=[batch_size], device=obs.device)

    def _transform_obs_group(
        self, obs: TensorDict, obs_groups: list[str], in_type: FieldType
    ) -> dict[str, torch.Tensor]:
        """Transform concatenated observation groups for a model and split them back apart."""
        widths = [obs[group_name].shape[-1] for group_name in obs_groups]
        flat = torch.cat([obs[group_name] for group_name in obs_groups], dim=-1)
        replicas = [in_type.transform_fibers(flat, g) for g in self.G.elements[1:]]

        per_group = {group_name: [obs[group_name]] for group_name in obs_groups}
        for replica in replicas:
            for group_name, chunk in zip(obs_groups, torch.split(replica, widths, dim=-1)):
                per_group[group_name].append(chunk)
        return {group_name: torch.cat(chunks, dim=0) for group_name, chunks in per_group.items()}

    @staticmethod
    def _tile(value: torch.Tensor, num_replica: int) -> torch.Tensor:
        return value.repeat(num_replica, *([1] * (value.dim() - 1)))

    def compute_returns(self, obs: TensorDict, last_action: torch.Tensor | None = None) -> None:
        st = self.storage
        if last_action is not None:
            obs.set("koopman_prediction", self.rff_predict(obs, last_action))

        last_values = self.critic(self._augment_observations(obs)).detach()

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry and self.symmetry.get("use_data_augmentation", False):
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    obs=batch.observations, actions=batch.actions, env=self.symmetry["_env"]
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob_batch = self.actor.get_output_log_prob(batch.actions)

            batch.observations.set("koopman_prediction", self.rff_predict(batch.observations, batch.actions))
            value_batch = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])

            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            mu_batch = distribution_params[0]
            sigma_batch = distribution_params[1]
            entropy_batch = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    old_mu_batch = batch.old_distribution_params[0]
                    old_sigma_batch = batch.old_distribution_params[1]

                    if getattr(self.actor, "use_log_prob_kl", False):
                        old_actions_log_prob = batch.old_actions_log_prob.squeeze(-1)
                        if old_actions_log_prob.shape != actions_log_prob_batch.shape:
                            old_actions_log_prob = old_actions_log_prob.reshape_as(actions_log_prob_batch)
                        kl = old_actions_log_prob - actions_log_prob_batch.detach()
                    elif getattr(self.actor, "use_masked_action_kl", False):
                        active_dims = (old_sigma_batch > 0.0) & (sigma_batch > 0.0)
                        old_sigma = old_sigma_batch.clamp_min(1.0e-6)
                        sigma = sigma_batch.clamp_min(1.0e-6)
                        kl = torch.sum(
                            (
                                torch.log(sigma / old_sigma)
                                + (torch.square(old_sigma) + torch.square(old_mu_batch - mu_batch))
                                / (2.0 * torch.square(sigma))
                                - 0.5
                            )
                            * active_dims,
                            axis=-1,
                        )
                    else:
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (value_batch - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry.get("use_data_augmentation", False):
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=batch.observations, actions=None, env=self.symmetry["_env"])
                else:
                    obs_batch = batch.observations
                    data_augmentation_func = self.symmetry["data_augmentation_func"]

                mean_actions_batch = self.actor(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )

                if self.symmetry.get("use_mirror_loss", False):
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.state_normalizer(
                        self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    )
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()

            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            import torch.nn.utils as nn_utils
            nn_utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        return loss_dict

    def broadcast_parameters(self) -> None:
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel