# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.extensions.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from morphosymm_rsl_rl.storage import RunningStdScaler, PrioritizedReplayBuffer
from morphosymm_rsl_rl.modules.dae_actor_critic import DAEModel
from dha.utils.utils import initialize_dae_model


class PPODAEOnline:
    """Proximal Policy Optimization algorithm with Online DAE integration (rsl_rl v5.4.2 style)."""

    actor: DAEModel
    critic: DAEModel

    def __init__(
        self,
        actor: DAEModel,
        critic: DAEModel,
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
        task: str = "flat_direct",
        dt: float = 0.02,
        single_observation_space: int = 1,
        action_space: int = 1,
        history_length: int = 1,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        morphologycal_symmetries_cfg: dict | None = None,
        koopman_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.actor = actor
        self.critic = critic

        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

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

        self.koopman_cfg = koopman_cfg
        self.state_dim = single_observation_space
        self.action_dim = action_space
        self.replay_buffer = PrioritizedReplayBuffer(
            self.state_dim, self.action_dim,
            koopman_cfg["beta_initial"], koopman_cfg["beta_annealing_steps"],
            koopman_cfg["replay_buffer_size"]
        )
        self.obs_action_normalizer = RunningStdScaler(self.state_dim, self.action_dim, device=self.device)
        self.task = task

        G_component = getattr(self.actor, "G", None) if "ecdae" in self.task else None
        self.dae_model = initialize_dae_model(
            morphologycal_symmetries_cfg=morphologycal_symmetries_cfg,
            koopman_cfg=koopman_cfg,
            G=G_component,
            task=self.task,
            state_dim=single_observation_space,
            action_dim=action_space,
            dt=dt,
            device=self.device
        )
        self.dae_optimizer = torch.optim.Adam(self.dae_model.parameters(), lr=koopman_cfg["lr"])

        self.single_observation_space = single_observation_space
        self.history_length = history_length

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )
        self.learning_rate = learning_rate

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

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPODAEOnline:
        alg_cfg = cfg["algorithm"]
        policy_cfg = cfg.get("policy", cfg.get("actor", {}))

        single_observation_space = env.cfg.single_observation_space
        history_length = env.cfg.history_length
        action_space = env.cfg.action_space
        dt = env.unwrapped.step_dt
        task = cfg["experiment_name"]

        koopman_cfg = cfg.get("koopman_cfg", {})
        morphologycal_symmetries_cfg = cfg.get("morphologycal_symmetries_cfg", {})

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        if "koopman_prediction" not in cfg["obs_groups"]["critic"]:
            cfg["obs_groups"]["critic"].append("koopman_prediction")
        koopman_dim = single_observation_space * koopman_cfg.get("obs_state_ratio", 1)
        obs["koopman_prediction"] = torch.zeros((obs["policy"].shape[0], koopman_dim), device=device)

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

        actor_cfg.setdefault("distribution_cfg", {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": actor_cfg.pop("init_noise_std", 1.0),
            "std_type": actor_cfg.pop("noise_std_type", "scalar"),
        })

        actor = DAEModel(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic = DAEModel(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)

        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg_cfg.pop("class_name", None)
        alg_cfg.pop("share_cnn_encoders", None)

        return PPODAEOnline(
            actor=actor,
            critic=critic,
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

    def get_policy(self) -> DAEModel:
        return self.actor

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()
        if hasattr(self, 'dae_model'):
            self.dae_model.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()
        if hasattr(self, 'dae_model'):
            self.dae_model.eval()

    def dae_predict(self, obs, action):
        critic_obs = obs["critic"]
        dae_input = critic_obs[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length]
        dae_input = dae_input.to(dtype=next(self.dae_model.parameters()).dtype)
        dae_input = dae_input.to(device=next(self.dae_model.parameters()).device)

        n_steps = 1
        dae_input_normed, action_normed = self.obs_action_normalizer.normalize(dae_input, action)
        action_normed = action_normed.unsqueeze(1).repeat(1, n_steps, 1)

        _, next_latents = self.dae_model.forecast(dae_input_normed, action_normed, n_steps=n_steps)
        next_latent = next_latents[:, -1, :].detach()

        return next_latent

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.actor.is_recurrent:
            self.transition.hidden_states = self.actor.get_hidden_states()

        self.transition.actions = self.actor.act(obs).detach()
        koopman_pred = self.dae_predict(obs, self.transition.actions)
        obs.set("koopman_prediction", koopman_pred)

        self.transition.values = self.critic.evaluate(obs).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()

        self.transition.distribution_params = (self.transition.action_mean, self.transition.action_sigma)
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

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict, last_action: torch.Tensor = None) -> None:
        st = self.storage
        if last_action is not None:
            obs.set("koopman_prediction", self.dae_predict(obs, last_action))
        last_values = self.critic.evaluate(obs).detach()

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

        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:

            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor.act(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0])
            actions_log_prob_batch = self.actor.get_actions_log_prob(batch.actions)
            batch.observations.set("koopman_prediction", self.dae_predict(batch.observations, batch.actions))

            value_batch = self.critic.evaluate(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            mu_batch = self.actor.action_mean[:original_batch_size]
            sigma_batch = self.actor.action_std[:original_batch_size]
            entropy_batch = self.actor.entropy[:original_batch_size]

            # --- Adaptive Learning Rate Schedule ---
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    # Unpack the old parameters from the v5 batch object
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
            # ---------------------------------------

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (value_batch - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            from itertools import chain
            nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

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
        grads = [param.grad.view(-1) for param in chain(self.actor.parameters(), self.critic.parameters()) if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel