# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
