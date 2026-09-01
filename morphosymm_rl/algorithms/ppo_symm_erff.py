import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict
import escnn
from escnn.nn import FieldType

from rsl_rl.storage import RolloutStorage
from morphosymm_rl.storage import RunningStdScaler, PrioritizedReplayBuffer
from morphosymm_rl.symm_utils import configure_observation_space_representations

# NOTE: Adjust these imports to match where your RFF utilities are located in IsaacLab
from morphosymm_rl.rff import (
    EquivariantRandomFourierFeatures,
    RunningLatentNormalizer,
    EquivariantKoopmanEstimator
)

# Import helper functions from PPOSymmDataAugmented
from morphosymm_rl.algorithms.ppo_symm_data_augment import (
    _field_representation_matrices, _hidden_representation_matrices,
    _linear_layers, _project_network_initialization
)

class PPOSymmERFF:
    def __init__(
        self,
        policy,
        storage_hack: RolloutStorage,
        obs_hack: TensorDict,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        task: str = "erff_koopman",
        dt: int = 1,
        single_observation_space: int = 1,
        action_space: int = 1,
        history_length: int = 1,
        multi_gpu_cfg: dict | None = None,
        koopman_cfg: dict | None = None,
        **morphologycal_symmetries_cfg,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if self.is_multi_gpu:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = None

        # PPO parameters
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
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Setup Policy
        self.policy = policy
        self.policy.to(self.device)

        # MorphoSymm components[cite: 11]
        obs_space_names_actor = morphologycal_symmetries_cfg["obs_space_names_actor"]
        obs_space_names_critic = morphologycal_symmetries_cfg["obs_space_names_critic"]
        action_space_names = morphologycal_symmetries_cfg["action_space_names"]
        joints_order = morphologycal_symmetries_cfg["joints_order"]
        robot_name = morphologycal_symmetries_cfg["robot_name"]
        symmetric_initialization = morphologycal_symmetries_cfg.pop("symmetric_initialization", True)

        G_actor, obs_reps_actor = configure_observation_space_representations(robot_name, obs_space_names_actor, joints_order)
        G_critic, obs_reps_critic = configure_observation_space_representations(robot_name, obs_space_names_critic, joints_order)

        self.G = G_actor
        gspace = escnn.gspaces.no_base_space(self.G)
        self.num_replica = len(self.G.elements)

        self.actor_in_type = FieldType(gspace, [obs_reps_actor[n] for n in obs_space_names_actor])
        self.actor_out_type = FieldType(gspace, [obs_reps_actor[n] for n in action_space_names])
        self.critic_in_field_type = FieldType(gspace, [obs_reps_critic[n] for n in obs_space_names_critic])

        # Pull Symmetry Configs directly for loss/augmentation
        self.symmetry = None

        if symmetric_initialization:
            self._apply_symmetric_initialization()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # ERFF and Koopman components setup[cite: 9]
        self.koopman_cfg = koopman_cfg
        self.task = task
        self.dt = dt
        self.single_observation_space = single_observation_space
        self.history_length = history_length
        self.state_dim = single_observation_space
        self.action_dim = action_space

        self.replay_buffer = PrioritizedReplayBuffer(self.state_dim, self.action_dim, koopman_cfg["beta_initial"], koopman_cfg["beta_annealing_steps"], koopman_cfg["replay_buffer_size"], device=self.device)
        self.obs_action_normalizer = RunningStdScaler(self.state_dim, self.action_dim, device=self.device)

        # Initialize ERFF and Latent Normalizer[cite: 9]
        m_features = koopman_cfg['m']
        group_order = self.G.order()
        num_reps = round(m_features / group_order)

        self.rff = EquivariantRandomFourierFeatures(
            task=task,
            in_features=self.state_dim,
            in_type=self.policy.state_type,  # Pass the state_type from ActorCriticSymm[cite: 9]
            m=num_reps,
            sigma=koopman_cfg['sigma'],
            kernel_type=koopman_cfg['kernel_type']
        ).to(self.device)

        latent_dim = num_reps * group_order
        self.latent_normalizer = RunningLatentNormalizer(num_features=latent_dim, device=self.device)

        if "koopman" in task:
            self.koopman_estimator = EquivariantKoopmanEstimator(
                self.rff,
                self.latent_normalizer,
                action_type=self.policy.actor_out_type,
                gamma=koopman_cfg['gamma'],
                device=self.device
            )

        # Setup Expanded Storage for Symmetries[cite: 11]
        batch_size = int(storage_hack.num_envs * self.num_replica)
        policy_shape = tuple(obs_hack["policy"].shape[1:])
        critic_shape = tuple(obs_hack["critic"].shape[1:])
        device = obs_hack["policy"].device

        obs = TensorDict(
            {
                "policy": torch.empty((batch_size, *policy_shape), device=device, dtype=obs_hack["policy"].dtype),
                "critic": torch.empty((batch_size, *critic_shape), device=device, dtype=obs_hack["critic"].dtype),
                "koopman_prediction": torch.empty((batch_size, *obs_hack["koopman_prediction"].shape[1:]), device=device, dtype=obs_hack["koopman_prediction"].dtype)
            },
            batch_size=torch.Size([batch_size]),
            device=device,
        )

        self.storage = RolloutStorage("rl", storage_hack.num_envs * self.num_replica, storage_hack.num_transitions_per_env, obs, storage_hack.actions_shape, storage_hack.device)
        self.transition = RolloutStorage.Transition()

    def _apply_symmetric_initialization(self) -> None:
        """Project the initial policy weights to symmetry-compatible subspaces.[cite: 11]"""
        with torch.no_grad():
            dtype = next(self.policy.parameters()).dtype
            device = next(self.policy.parameters()).device
            elements = self.G.elements
            hidden_mats_cache = {}

            actor_in_mats = _field_representation_matrices(self.actor_in_type, elements, device, dtype)
            actor_out_mats = _field_representation_matrices(self.actor_out_type, elements, device, dtype)
            critic_in_mats = _field_representation_matrices(self.critic_in_field_type, elements, device, dtype)
            critic_out_mats = [torch.ones((1, 1), device=device, dtype=dtype) for _ in elements]

            def hidden_mats_factory(size: int):
                if size not in hidden_mats_cache:
                    hidden_mats_cache[size] = _hidden_representation_matrices(self.G, size, elements, device, dtype)
                return hidden_mats_cache[size]

            actor_layers = _linear_layers(self.policy.actor)
            if actor_layers:
                actor_last_out_features = actor_layers[-1].out_features
                if actor_last_out_features == self.actor_out_type.size:
                    projected_actor_out_mats = actor_out_mats
                elif actor_last_out_features == 2 * self.actor_out_type.size:
                    projected_actor_out_mats = [
                        torch.block_diag(action_mat, action_mat.abs()) for action_mat in actor_out_mats
                    ]
                else:
                    projected_actor_out_mats = actor_out_mats
            else:
                projected_actor_out_mats = actor_out_mats

            _project_network_initialization(
                self.policy.actor,
                actor_in_mats,
                projected_actor_out_mats,
                hidden_mats_factory,
                "actor",
            )
            _project_network_initialization(
                self.policy.critic,
                critic_in_mats,
                critic_out_mats,
                hidden_mats_factory,
                "critic",
            )
            self._project_action_std_initialization(actor_out_mats)

    def _project_action_std_initialization(self, actor_out_mats) -> None:
        std_param = None
        if hasattr(self.policy, "std"):
            std_param = self.policy.std
        elif hasattr(self.policy, "log_std"):
            std_param = self.policy.log_std

        if std_param is None:
            return

        if std_param.ndim == 1 and std_param.shape[0] == self.actor_out_type.size:
            projected_std = torch.zeros_like(std_param)
            for action_mat in actor_out_mats:
                projected_std += action_mat.abs().transpose(0, 1) @ std_param
            std_param.copy_(projected_std / len(actor_out_mats))
        elif std_param.ndim == 2 and std_param.shape[1] == self.actor_out_type.size:
            projected_std = torch.zeros_like(std_param)
            for action_mat in actor_out_mats:
                projected_std += std_param @ action_mat.abs()
            std_param.copy_(projected_std / len(actor_out_mats))

    def rff_predict(self, obs, action=None):
        """Processes critic_obs through ERFF/Koopman to get augmented input for the critic."""
        critic_obs = obs["critic"]

        # Extract the most recent state
        dae_input = critic_obs[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length]

        # Wrap the state in a GeometricTensor for the Equivariant RFF[cite: 9]
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
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        self.transition.actions = self.policy.act(obs).detach()

        # Inject the Koopman prediction *before* evaluation so the critic sees it[cite: 11]
        koopman_pred = self.rff_predict(obs, self.transition.actions)
        obs.set("koopman_prediction", koopman_pred)

        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]) -> None:
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)

        # Augment the transitions symmetrically before pushing to storage[cite: 11]
        self.augment_transitions()

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def augment_transitions(self):
        """Symmetrically augments actions, policy obs, critic obs, and koopman predictions.[cite: 11]"""
        t = self.transition

        out_field_type = self.actor_out_type
        in_field_type = self.actor_in_type
        critic_in_field_type = self.critic_in_field_type
        G = self.G

        # 1. Augment Actions and Values[cite: 11]
        t.actions = torch.cat(
            [t.actions] + [out_field_type.transform_fibers(t.actions, g) for g in G.elements[1:]],
            dim=0,
        )
        t.actions_log_prob = torch.cat([t.actions_log_prob] * self.num_replica, dim=0)
        t.action_mean = torch.cat(
            [t.action_mean] + [out_field_type.transform_fibers(t.action_mean, g) for g in G.elements[1:]],
            dim=0,
        )
        t.action_sigma = torch.abs(
            torch.cat(
                [t.action_sigma] + [out_field_type.transform_fibers(t.action_sigma, g) for g in G.elements[1:]],
                dim=0,
            )
        )
        t.values = torch.cat([t.values] * self.num_replica, dim=0)
        t.rewards = torch.cat([t.rewards] * self.num_replica, dim=0)
        t.dones = torch.cat([t.dones] * self.num_replica, dim=0)

        # 2. Augment Policy Obs[cite: 11]
        policy_obs_aug = torch.cat(
            [t.observations["policy"]]
            + [in_field_type.transform_fibers(t.observations["policy"], g) for g in G.elements[1:]],
            dim=0,
        )

        # 3. Augment Critic Obs & Koopman Prediction[cite: 11]
        full_critic_obs = self.policy.get_critic_obs(t.observations)
        transformed_full_critic = [critic_in_field_type.transform_fibers(full_critic_obs, g) for g in G.elements[1:]]

        base_critic_dim = t.observations["critic"].shape[-1]

        critic_obs_aug = torch.cat(
            [t.observations["critic"]]
            + [tf[..., :base_critic_dim] for tf in transformed_full_critic],
            dim=0,
        )

        koopman_aug = torch.cat(
            [t.observations["koopman_prediction"]]
            + [tf[..., base_critic_dim:] for tf in transformed_full_critic],
            dim=0,
        )

        # 4. Rebuild the TensorDict[cite: 11]
        t.observations = TensorDict(
            {
                "policy": policy_obs_aug,
                "critic": critic_obs_aug,
                "koopman_prediction": koopman_aug
            },
            batch_size=policy_obs_aug.shape[:1],
        )

    def augment_values(self, values):
        return torch.cat([values] * self.num_replica, dim=0)

    def compute_returns(self, obs: TensorDict, last_action: torch.Tensor) -> None:
        st = self.storage
        obs["koopman_prediction"] = self.rff_predict(obs, last_action)
        last_values_temp = self.policy.evaluate(obs).detach()
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

        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs) if self.policy.is_recurrent else self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hidden_states_batch, masks_batch) in generator:

            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation for the current batch[cite: 11]
            if self.symmetry and self.symmetry.get("use_data_augmentation", False):
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)

            # Inject Koopman prediction[cite: 11]
            obs_batch["koopman_prediction"] = self.rff_predict(obs_batch, actions_batch)

            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if getattr(self.policy, "use_log_prob_kl", False):
                        old_actions_log_prob = old_actions_log_prob_batch.squeeze(-1)
                        if old_actions_log_prob.shape != actions_log_prob_batch.shape:
                            old_actions_log_prob = old_actions_log_prob.reshape_as(actions_log_prob_batch)
                        kl = old_actions_log_prob - actions_log_prob_batch.detach()
                    elif getattr(self.policy, "use_masked_action_kl", False):
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

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Standard PPO Loss Computations[cite: 11]
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry Mirror Loss[cite: 11]
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"])

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:])

                if self.symmetry.get("use_mirror_loss", False):
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Optimization Step
            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu: self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Tally Losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_symmetry_loss is not None: mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if self.symmetry: loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs.[cite: 11]"""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.[cite: 11]"""
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel