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
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

# Inherit from your working SymmPPO to drastically reduce boilerplate
from morphosymm_rsl_rl.algorithms.ppo import PPO as SymmPPO
from morphosymm_rsl_rl.modules.ac_symm import SymmModel
from morphosymm_rsl_rl.storage import PrioritizedReplayBuffer, RunningStdScaler
from morphosymm_rsl_rl.symm_utils import configure_observation_space_representations
from morphosymm_rsl_rl.rff import (
    EquivariantRandomFourierFeatures,
    RunningLatentNormalizer,
    EquivariantKoopmanEstimator
)


class PPOSymmERFF(SymmPPO):
    """Equivariant Proximal Policy Optimization with Online ERFF integration, inheriting from SymmPPO."""

    G: Group
    actor_out_type: FieldType
    state_type: FieldType

    def __init__(
        self,
        actor: SymmModel,
        critic: SymmModel,
        storage: RolloutStorage,
        task: str = "erff_koopman",
        dt: float = 1.0,
        single_observation_space: int = 1,
        action_space: int = 1,
        history_length: int = 1,
        koopman_cfg: dict | None = None,
        morphologycal_symmetries_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        # Pass standard PPO kwargs and empirical symmetry_cfg to SymmPPO / RslRlPPO
        super().__init__(actor, critic, storage, symmetry_cfg=symmetry_cfg, **kwargs)

        self.task = task
        self.dt = dt
        self.single_observation_space = single_observation_space
        self.history_length = history_length
        self.state_dim = single_observation_space
        self.action_dim = action_space
        self.koopman_cfg = koopman_cfg or {}

        # Store morphological symmetry configuration separately to prevent shadowing self.symmetry
        self.morphologycal_symmetries_cfg = morphologycal_symmetries_cfg

        self.replay_buffer = PrioritizedReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.koopman_cfg.get("beta_initial", 0.4),
            self.koopman_cfg.get("beta_annealing_steps", 1000),
            self.koopman_cfg.get("replay_buffer_size", 10000),
            device=self.device,
        )
        self.obs_action_normalizer = RunningStdScaler(self.state_dim, self.action_dim, device=self.device)

        # Pull representations natively from the actor
        self.G = self.actor.G
        self.actor_out_type = getattr(self.actor, "out_type", getattr(self.actor, "actor_out_type", None))
        all_space_names = list(
            dict.fromkeys(
                [
                    *morphologycal_symmetries_cfg["obs_space_names_actor"],
                    *morphologycal_symmetries_cfg["obs_space_names_critic"],
                    *morphologycal_symmetries_cfg["action_space_names"],
                ]
            )
        )
        G, representations = configure_observation_space_representations(
            morphologycal_symmetries_cfg["robot_name"], all_space_names, morphologycal_symmetries_cfg["joints_order"]
        )
        gspace = escnn.gspaces.no_base_space(G)
        self.state_type = FieldType(gspace, [representations[n] for n in self.morphologycal_symmetries_cfg["obs_space_names_single_state"]])

        # Initialize ERFF and Koopman using the single state_type
        m_features = self.koopman_cfg.get('m', 129)
        group_order = self.G.order()
        num_reps = round(m_features / group_order)

        self.rff = EquivariantRandomFourierFeatures(
            task=task,
            in_features=single_observation_space,
            in_type=self.state_type,
            m=num_reps,
            sigma=self.koopman_cfg.get('sigma', 1.0),
            kernel_type=self.koopman_cfg.get('kernel_type', 'gaussian')
        ).to(self.device)

        latent_dim = num_reps * group_order
        self.latent_normalizer = RunningLatentNormalizer(num_features=latent_dim, device=self.device)

        # TODO remove the equivariant Koopman estimator since it is not needed
        if "koopman" in task:
            self.koopman_estimator = EquivariantKoopmanEstimator(
                self.rff,
                self.latent_normalizer,
                action_type=self.actor_out_type,
                gamma=self.koopman_cfg.get('gamma', 1.0),
                device=self.device
            )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOSymmERFF:
        """Construct the algorithm relying heavily on SymmModel factory."""
        alg_cfg = cfg["algorithm"]
        policy_cfg = cfg.get("policy", cfg.get("actor", {}))

        single_observation_space = env.cfg.single_observation_space
        history_length = env.cfg.history_length
        action_space = env.cfg.action_space
        dt = env.unwrapped.step_dt
        task = cfg["experiment_name"]

        koopman_cfg = cfg.get("koopman_cfg", {})
        symm_cfg = cfg.get("morphologycal_symmetries_cfg", {})

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

        all_space_names = list(
            dict.fromkeys(
                [
                    *symm_cfg["obs_space_names_actor"],
                    *symm_cfg["obs_space_names_critic"],
                    *symm_cfg["action_space_names"],
                    *(symm_cfg.get("obs_space_names_single_state") or []),
                ]
            )
        )
        G, _ = configure_observation_space_representations(
            symm_cfg["robot_name"], all_space_names, symm_cfg["joints_order"]
        )
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

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg_cfg.pop("class_name", None)
        alg_cfg.pop("share_cnn_encoders", None)

        alg = PPOSymmERFF(
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
            morphologycal_symmetries_cfg=symm_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )

        return alg

    def train_mode(self) -> None:
        super().train_mode()
        if hasattr(self, "rff"):
            self.rff.train()
        if hasattr(self, "latent_normalizer"):
            self.latent_normalizer.train()
        if hasattr(self, "koopman_estimator"):
            self.koopman_estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if hasattr(self, "rff"):
            self.rff.eval()
        if hasattr(self, "latent_normalizer"):
            self.latent_normalizer.eval()
        if hasattr(self, "koopman_estimator"):
            self.koopman_estimator.eval()

    def save(self) -> dict:
        """Hooks the ERFF networks into the native RSL-RL v5 dictionary architecture."""
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
        """Natively restores ERFF networks alongside standard PPO."""
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

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict, last_action: torch.Tensor | None = None) -> None:
        st = self.storage
        if last_action is not None:
            obs.set("koopman_prediction", self.rff_predict(obs, last_action))
        else:
            obs.set("koopman_prediction", self.rff_predict(obs, None))

        last_values = self.critic(obs).detach()

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