import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict
import escnn

from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable

from morphosymm_rl.storage import PrioritizedReplayBuffer, RunningStdScaler
# NOTE: Adjust these imports to match where your rff.py is located
from morphosymm_rl.rff import (
    RandomFourierFeatures, EquivariantRandomFourierFeatures,
    KoopmanEstimator, EquivariantKoopmanEstimator, RunningLatentNormalizer
)

class PPORFF:
    def __init__(self, policy, storage, num_learning_epochs=5, num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.01, learning_rate=0.001, max_grad_norm=1.0, use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01, normalize_advantage_per_mini_batch=False, device="cpu", task="rff", dt=1, single_observation_space=1, action_space=1, history_length=1, rnd_cfg=None, symmetry_cfg=None, multi_gpu_cfg=None, morphologycal_symmetries_cfg=None, koopman_cfg=None):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if self.is_multi_gpu:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.task = task
        self.dt = dt
        self.single_observation_space = single_observation_space
        self.history_length = history_length
        self.state_dim = single_observation_space
        self.action_dim = action_space
        self.koopman_cfg = koopman_cfg

        # 1. Initialize Replay Buffer & Normalizer
        N = koopman_cfg.get("replay_buffer_size", 10000)
        self.replay_buffer = PrioritizedReplayBuffer(self.state_dim, self.action_dim, koopman_cfg["beta_initial"], koopman_cfg["beta_annealing_steps"], N, device=self.device)
        self.obs_action_normalizer = RunningStdScaler(self.state_dim, self.action_dim, device=self.device)

        # 2. Initialize RFF & Koopman Estimators
        m = koopman_cfg.get('m', 129)
        sigma = koopman_cfg.get('sigma', 1.0)
        kernel_type = koopman_cfg.get('kernel_type', 'gaussian')

        if "erff" in task:
            self.state_type = policy.state_type
            self.action_field_type = policy.out_field_type
            group_order = policy.G.order()
            num_reps = round(m / group_order)

            self.rff = EquivariantRandomFourierFeatures(
                task=task, in_features=self.state_dim, in_type=self.state_type, m=num_reps, sigma=sigma, kernel_type=kernel_type
            ).to(self.device)

            latent_dim = num_reps * group_order
            self.latent_normalizer = RunningLatentNormalizer(num_features=latent_dim, device=self.device)

            if "koopman" in task:
                self.koopman_estimator = EquivariantKoopmanEstimator(
                    self.rff, self.latent_normalizer, action_type=self.action_field_type, gamma=koopman_cfg.get('gamma', 1.0), device=self.device
                )
        elif "rff" in task:
            self.rff = RandomFourierFeatures(in_features=self.state_dim, m=m, sigma=sigma, kernel_type=kernel_type).to(self.device)
            self.latent_normalizer = RunningLatentNormalizer(num_features=m, device=self.device)

            if "koopman" in task:
                self.koopman_estimator = KoopmanEstimator(self.rff, self.latent_normalizer, self.action_dim, gamma=koopman_cfg.get('gamma', 1.0), device=self.device)

        # 3. PPO Components
        self.policy = policy
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # Configs
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

        # Symmetry & RND Configuration (Identical to PPODAEOnline)
        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        if symmetry_cfg is not None:
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

    def rff_predict(self, obs, action):
        """Processes critic_obs through RFF/Koopman to get augmented input for the critic."""
        critic_obs = obs["critic"]

        # Extract the most recent state exactly as PPODAEOnline does
        dae_input = critic_obs[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length]

        # Wrap for equivariant if needed
        if "erff" in self.task:
            dae_input = escnn.nn.GeometricTensor(dae_input, self.state_type)

        # Lift the features
        latent_raw = self.rff(dae_input)

        # Predict next lifted state
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

        # Inject RFF Koopman prediction directly into the TensorDict
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
        if self.rnd: self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict, last_action: torch.Tensor) -> None:
        st = self.storage
        obs["koopman_prediction"] = self.rff_predict(obs, last_action)
        last_values = self.policy.evaluate(obs).detach()

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

            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"])
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Re-evaluate
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)

            # Re-inject Koopman prediction for the current batch
            obs_batch["koopman_prediction"] = self.rff_predict(obs_batch, actions_batch)

            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Standard PPO Clipping and Loss computations
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

            # Symmetry and RND logic (Exactly matching PPODAEOnline)
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"])
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:])
                if self.symmetry["use_mirror_loss"]: loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.state_normalizer(self.rnd.get_rnd_state(obs_batch[:original_batch_size]))
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            # Optimization Step
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu: self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer: self.rnd_optimizer.step()

            # Tally Losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None: mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None: mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if self.rnd: loss_dict["rnd"] = mean_rnd_loss / num_updates
        if self.symmetry: loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        return loss_dict

    # Include broadcast_parameters() and reduce_parameters() exactly as they are in PPODAEOnline.