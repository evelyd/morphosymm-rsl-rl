# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import chain
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import resolve_rnd_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from morphosymm_rsl_rl.storage import PrioritizedReplayBuffer, RunningStdScaler
from morphosymm_rsl_rl.rff import (
    RandomFourierFeatures,
    RunningLatentNormalizer,
    KoopmanEstimator
)


class PPORFF(PPO):
    """Proximal Policy Optimization algorithm with RFF/Koopman integration, natively extending standard PPO."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        storage: RolloutStorage,
        task: str = "rff",
        dt: float = 1.0,
        single_observation_space: int = 1,
        action_space: int = 1,
        history_length: int = 1,
        koopman_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        # Pass standard kwargs directly to base PPO
        super().__init__(actor, critic, storage, **kwargs)

        self.task = task
        self.dt = dt
        self.single_observation_space = single_observation_space
        self.history_length = history_length
        self.state_dim = single_observation_space
        self.action_dim = action_space
        self.koopman_cfg = koopman_cfg or {}

        # Initialize Replay Buffer & Normalizer
        N = self.koopman_cfg.get("replay_buffer_size", 10000)
        self.replay_buffer = PrioritizedReplayBuffer(
            self.state_dim, self.action_dim,
            self.koopman_cfg.get("beta_initial", 0.4),
            self.koopman_cfg.get("beta_annealing_steps", 1000),
            N, device=self.device
        )
        self.obs_action_normalizer = RunningStdScaler(self.state_dim, self.action_dim, device=self.device)

        # Initialize RFF & Koopman Estimators
        m = self.koopman_cfg.get('m', 129)
        sigma = self.koopman_cfg.get('sigma', 1.0)
        kernel_type = self.koopman_cfg.get('kernel_type', 'gaussian')

        self.rff = RandomFourierFeatures(
            in_features=self.state_dim, m=m, sigma=sigma, kernel_type=kernel_type
        ).to(self.device)
        self.latent_normalizer = RunningLatentNormalizer(num_features=m, device=self.device)

        if "koopman" in task:
            self.koopman_estimator = KoopmanEstimator(
                self.rff, self.latent_normalizer, self.action_dim,
                gamma=self.koopman_cfg.get('gamma', 1.0), device=self.device
            )

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPORFF:
        """Construct PPORFF using standard RSL-RL v5.4.2 factory architecture."""
        alg_cfg = cfg["algorithm"]
        policy_cfg = cfg.get("policy", cfg.get("actor", {}))

        single_observation_space = env.cfg.single_observation_space
        history_length = env.cfg.history_length
        action_space = env.cfg.action_space
        dt = env.unwrapped.step_dt
        task = cfg["experiment_name"]
        koopman_cfg = cfg.get("koopman_cfg", {})

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)

        if "koopman_prediction" not in cfg["obs_groups"]["critic"]:
            cfg["obs_groups"]["critic"].append("koopman_prediction")
        koopman_dim = koopman_cfg.get("m", 129)
        obs.set("koopman_prediction", torch.zeros((obs["policy"].shape[0], koopman_dim), device=device))

        # Standard instantiation
        actor_class_name = cfg.get("actor", policy_cfg).get("class_name", "MLPModel")
        critic_class_name = cfg.get("critic", policy_cfg).get("class_name", "MLPModel")
        actor_class = resolve_callable(actor_class_name)
        critic_class = resolve_callable(critic_class_name)

        actor_cfg = cfg.get("actor", policy_cfg).copy()
        critic_cfg = cfg.get("critic", policy_cfg).copy()
        for c in [actor_cfg, critic_cfg]:
            c.pop("class_name", None)

        actor_cfg.setdefault("hidden_dims", actor_cfg.pop("actor_hidden_dims", [256, 256, 256]))
        critic_cfg.setdefault("hidden_dims", critic_cfg.pop("critic_hidden_dims", [256, 256, 256]))
        actor_cfg.setdefault("obs_normalization", actor_cfg.pop("actor_obs_normalization", False))
        critic_cfg.setdefault("obs_normalization", critic_cfg.pop("critic_obs_normalization", False))
        actor_cfg.setdefault("distribution_cfg", {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": actor_cfg.pop("init_noise_std", 1.0),
            "std_type": actor_cfg.pop("noise_std_type", "scalar")
        })

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg_cfg.pop("class_name", None)
        alg_cfg.pop("share_cnn_encoders", None)

        return cls(
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

    def train_mode(self) -> None:
        super().train_mode()
        if hasattr(self, "rff"): self.rff.train()
        if hasattr(self, "latent_normalizer"): self.latent_normalizer.train()
        if hasattr(self, "koopman_estimator"): self.koopman_estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if hasattr(self, "rff"): self.rff.eval()
        if hasattr(self, "latent_normalizer"): self.latent_normalizer.eval()
        if hasattr(self, "koopman_estimator"): self.koopman_estimator.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["normalizer_state_dict"] = self.obs_action_normalizer.state_dict()
        saved_dict["latent_normalizer_state_dict"] = self.latent_normalizer.state_dict()
        if "koopman" in self.task and hasattr(self, "koopman_estimator"):
            saved_dict["koopman_state_dict"] = self.koopman_estimator.state_dict()
            saved_dict["koopman_config"] = {
                'koopman_input_dim': self.koopman_estimator.koopman_input_dim,
                'koopman_output_dim': self.koopman_estimator.koopman_output_dim,
                'gamma': self.koopman_estimator.gamma,
                'K': self.koopman_estimator.K_matrix
            }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> int:
        iteration = super().load(loaded_dict, load_cfg=load_cfg, strict=strict)

        if "normalizer_state_dict" in loaded_dict:
            self.obs_action_normalizer.load_state_dict(loaded_dict["normalizer_state_dict"])
        if "latent_normalizer_state_dict" in loaded_dict:
            self.latent_normalizer.load_state_dict(loaded_dict["latent_normalizer_state_dict"])
        if "koopman" in self.task and "koopman_state_dict" in loaded_dict and hasattr(self, "koopman_estimator"):
            self.koopman_estimator.load_state_dict(loaded_dict["koopman_state_dict"], strict=strict)
            if "koopman_config" in loaded_dict:
                self.koopman_estimator.K_matrix = loaded_dict["koopman_config"]["K"].to(self.device)

        return iteration

    def rff_predict(self, obs: TensorDict, action: torch.Tensor | None = None) -> torch.Tensor:
        critic_obs = obs["critic"]
        dae_input = critic_obs[
            :,
            self.single_observation_space * (self.history_length - 1) : self.single_observation_space * self.history_length,
        ]

        latent_raw = self.rff(dae_input)

        if action is not None and "koopman" in self.task:
            next_latent = self.koopman_estimator.predict_from_lifted_state(latent_raw, action)
        else:
            next_latent = latent_raw

        if hasattr(next_latent, 'tensor'):
            next_latent = next_latent.tensor
        return next_latent

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Override to predict Koopman states and inject before critic evaluates."""
        if self.actor.is_recurrent:
            self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())

        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        obs.set("koopman_prediction", self.rff_predict(obs, self.transition.actions))

        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def compute_returns(self, obs: TensorDict, last_action: torch.Tensor | None = None) -> None:
        """Inject Koopman prediction before final return computation."""
        if last_action is not None:
            obs.set("koopman_prediction", self.rff_predict(obs, last_action))
        else:
            obs.set("koopman_prediction", self.rff_predict(obs, None))
        super().compute_returns(obs)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob_batch = self.actor.get_output_log_prob(batch.actions)

            # Predict latent state
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
                            (torch.log(sigma / old_sigma) + (torch.square(old_sigma) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma)) - 0.5) * active_dims, axis=-1,
                        )
                    else:
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1,
                        )
                    kl_mean = torch.mean(kl)

                    if hasattr(self, "is_multi_gpu") and self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= getattr(self, "gpu_world_size", 1)

                    if getattr(self, "gpu_global_rank", 0) == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if hasattr(self, "is_multi_gpu") and self.is_multi_gpu:
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

            if self.rnd and hasattr(self.rnd, "optimizer"):
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if hasattr(self, "is_multi_gpu") and self.is_multi_gpu:
                self.reduce_parameters()

            import torch.nn.utils as nn_utils
            nn_utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            if self.rnd and hasattr(self.rnd, "optimizer"):
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / num_updates

        return loss_dict

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