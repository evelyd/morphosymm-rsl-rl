import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict
import escnn
from escnn.nn import FieldType

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.extensions.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups, resolve_optimizer

from morphosymm_rsl_rl.storage import RunningStdScaler, PrioritizedReplayBuffer
from morphosymm_rsl_rl.symm_utils import configure_observation_space_representations
from morphosymm_rsl_rl.modules import ActorCriticSymm
from morphosymm_rsl_rl.rff import (
    EquivariantRandomFourierFeatures,
    RunningLatentNormalizer,
    EquivariantKoopmanEstimator
)


class PPOSymmERFF:
    """Equivariant Proximal Policy Optimization algorithm with Online ERFF integration (rsl_rl v5.4.2 style)."""

    def __init__(
        self,
        policy: ActorCriticSymm,
        storage: RolloutStorage,
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
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if self.is_multi_gpu:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.policy.to(self.device)

        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        self.symmetry = symmetry_cfg
        self.G = self.policy.G
        self.num_replica = len(self.G.elements)
        self.actor_in_type = self.policy.actor_in_type
        self.actor_out_type = self.policy.actor_out_type
        self.critic_in_field_type = self.policy.critic_in_field_type

        # Optimizer string resolution (v5 style)
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        self.koopman_cfg = koopman_cfg or {}
        self.task = task
        self.dt = dt
        self.single_observation_space = single_observation_space
        self.history_length = history_length
        self.state_dim = single_observation_space
        self.action_dim = action_space

        self.replay_buffer = PrioritizedReplayBuffer(
            self.state_dim, self.action_dim,
            self.koopman_cfg.get("beta_initial", 0.4),
            self.koopman_cfg.get("beta_annealing_steps", 1000),
            self.koopman_cfg.get("replay_buffer_size", 10000),
            device=self.device
        )
        self.obs_action_normalizer = RunningStdScaler(self.state_dim, self.action_dim, device=self.device)

        m_features = self.koopman_cfg.get('m', 129)
        group_order = self.G.order()
        num_reps = round(m_features / group_order)

        self.rff = EquivariantRandomFourierFeatures(
            task=task,
            in_features=self.state_dim,
            in_type=self.policy.state_type,
            m=num_reps,
            sigma=self.koopman_cfg.get('sigma', 1.0),
            kernel_type=self.koopman_cfg.get('kernel_type', 'gaussian')
        ).to(self.device)

        latent_dim = num_reps * group_order
        self.latent_normalizer = RunningLatentNormalizer(num_features=latent_dim, device=self.device)

        if "koopman" in task:
            self.koopman_estimator = EquivariantKoopmanEstimator(
                self.rff,
                self.latent_normalizer,
                action_type=self.policy.actor_out_type,
                gamma=self.koopman_cfg.get('gamma', 1.0),
                device=self.device
            )

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
        alg_cfg = cfg["algorithm"]
        policy_cfg = cfg.get("policy", {})

        single_observation_space = env.cfg.single_observation_space
        history_length = env.cfg.history_length
        action_space = env.cfg.action_space
        dt = env.unwrapped.step_dt
        task = cfg["experiment_name"]

        koopman_cfg = cfg.get("koopman_cfg", {})
        morphologycal_symmetries_cfg = cfg.get("morphologycal_symmetries_cfg", {})

        default_sets = ["critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        if "koopman_prediction" not in cfg["obs_groups"]["critic"]:
            cfg["obs_groups"]["critic"].append("koopman_prediction")
        koopman_dim = koopman_cfg.get("m", 129)
        obs.set("koopman_prediction", torch.zeros((obs["policy"].shape[0], koopman_dim), device=device))

        policy_copy = policy_cfg.copy()
        policy_copy.pop("class_name", None)

        actor_critic = ActorCriticSymm(
            obs,
            cfg["obs_groups"],
            env.num_actions,
            **policy_copy,
            **morphologycal_symmetries_cfg,
        ).to(device)

        G_actor, _ = configure_observation_space_representations(
            morphologycal_symmetries_cfg["robot_name"],
            morphologycal_symmetries_cfg["obs_space_names_actor"],
            morphologycal_symmetries_cfg["joints_order"]
        )
        num_replica = len(G_actor.elements)
        batch_size = int(env.num_envs * num_replica)

        policy_shape = tuple(obs["policy"].shape[1:])
        critic_shape = tuple(obs["critic"].shape[1:])

        expanded_obs = TensorDict(
            {
                "policy": torch.empty((batch_size, *policy_shape), device=device, dtype=obs["policy"].dtype),
                "critic": torch.empty((batch_size, *critic_shape), device=device, dtype=obs["critic"].dtype),
                "koopman_prediction": torch.empty((batch_size, *obs["koopman_prediction"].shape[1:]), device=device, dtype=obs["koopman_prediction"].dtype)
            },
            batch_size=torch.Size([batch_size]),
            device=device,
        )

        storage = RolloutStorage("rl", batch_size, cfg["num_steps_per_env"], expanded_obs, [env.num_actions], device)

        alg_cfg.pop("class_name", None)
        alg_cfg.pop("share_cnn_encoders", None)

        return cls(
            policy=actor_critic,
            storage=storage,
            device=device,
            task=task,
            dt=dt,
            single_observation_space=single_observation_space,
            action_space=action_space,
            history_length=history_length,
            morphologycal_symmetries_cfg=morphologycal_symmetries_cfg,
            koopman_cfg=koopman_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )

    def get_policy(self):
        return self.policy

    def train_mode(self):
        self.policy.train()
        if self.rnd: self.rnd.train()

    def eval_mode(self):
        self.policy.eval()
        if self.rnd: self.rnd.eval()

    def rff_predict(self, obs, action=None):
        critic_obs = obs["critic"]
        dae_input = critic_obs[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length]
        dae_input = escnn.nn.GeometricTensor(dae_input, self.policy.state_type)
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

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]) -> None:
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)

        self.augment_transitions()
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def augment_transitions(self):
        t = self.transition
        out_field_type = self.actor_out_type
        in_field_type = self.actor_in_type
        critic_in_field_type = self.critic_in_field_type
        G = self.G

        t.actions = torch.cat([t.actions] + [out_field_type.transform_fibers(t.actions, g) for g in G.elements[1:]], dim=0)
        t.actions_log_prob = torch.cat([t.actions_log_prob] * self.num_replica, dim=0)
        t.action_mean = torch.cat([t.action_mean] + [out_field_type.transform_fibers(t.action_mean, g) for g in G.elements[1:]], dim=0)
        t.action_sigma = torch.abs(torch.cat([t.action_sigma] + [out_field_type.transform_fibers(t.action_sigma, g) for g in G.elements[1:]], dim=0))
        t.values = torch.cat([t.values] * self.num_replica, dim=0)
        t.rewards = torch.cat([t.rewards] * self.num_replica, dim=0)
        t.dones = torch.cat([t.dones] * self.num_replica, dim=0)

        policy_obs_aug = torch.cat([t.observations["policy"]] + [in_field_type.transform_fibers(t.observations["policy"], g) for g in G.elements[1:]], dim=0)

        full_critic_obs = self.policy.get_critic_obs(t.observations)
        transformed_full_critic = [critic_in_field_type.transform_fibers(full_critic_obs, g) for g in G.elements[1:]]

        base_critic_dim = t.observations["critic"].shape[-1]
        critic_obs_aug = torch.cat([t.observations["critic"]] + [tf[..., :base_critic_dim] for tf in transformed_full_critic], dim=0)
        koopman_aug = torch.cat([t.observations["koopman_prediction"]] + [tf[..., base_critic_dim:] for tf in transformed_full_critic], dim=0)

        t.observations = TensorDict({"policy": policy_obs_aug, "critic": critic_obs_aug, "koopman_prediction": koopman_aug}, batch_size=policy_obs_aug.shape[:1])

    def augment_values(self, values):
        return torch.cat([values] * self.num_replica, dim=0)

    def compute_returns(self, obs: TensorDict, last_action: torch.Tensor | None = None) -> None:
        st = self.storage
        if last_action is not None:
            obs.set("koopman_prediction", self.rff_predict(obs, last_action))

        # V5 Critic evaluation
        last_values_temp = self.critic(obs).detach()
        last_values = self.augment_values(last_values_temp)

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

            # --- Symmetry Augmentation ---
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

            # --- Forward Passes (v5.4.2 Style) ---
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob_batch = self.actor.get_output_log_prob(batch.actions)

            # Predict next latent state using ERFF / Koopman
            batch.observations.set("koopman_prediction", self.rff_predict(batch.observations, batch.actions))

            value_batch = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])

            # Extract current distribution parameters for the original samples (v5.4.2 Style)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            mu_batch = distribution_params[0]
            sigma_batch = distribution_params[1]
            entropy_batch = self.actor.output_entropy[:original_batch_size]

            # --- Adaptive KL Divergence Schedule ---
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    # Unpack old parameters from the v5 batch
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

            # --- PPO Losses ---
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

            # --- Symmetry Mirror Loss ---
            if self.symmetry:
                if not self.symmetry.get("use_data_augmentation", False):
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=batch.observations, actions=None, env=self.symmetry["_env"])
                else:
                    obs_batch = batch.observations
                    data_augmentation_func = self.symmetry["data_augmentation_func"]

                # Inference pass without stochastic output provides the mean directly
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

            # --- RND Loss ---
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.state_normalizer(
                        self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    )
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            # --- Optimization Step ---
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

            # --- Accumulate Metrics ---
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
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel