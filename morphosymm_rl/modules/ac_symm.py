from __future__ import annotations

from typing import Any, NoReturn

import escnn
import numpy as np
import torch
import torch.nn as nn
from escnn.nn import FieldType
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import EmpiricalNormalization
from symm_learning.models import EMLP, IMLP
from symm_learning.nn import EquivMultivariateNormal
from tensordict import TensorDict
from torch.distributions import Normal

from morphosymm_rl.symm_utils import configure_observation_space_representations


class _EquivariantEmpiricalNormalization(EmpiricalNormalization):
    """Empirical normalization with statistics symmetrized over group orbits."""

    def __init__(self, field_type: FieldType) -> None:
        super().__init__(field_type.size)
        group_matrices = torch.stack(
            [
                torch.as_tensor(
                    field_type.representation(group_element),
                    dtype=torch.get_default_dtype(),
                )
                for group_element in field_type.fibergroup.elements
            ]
        )
        self.register_buffer("_group_matrices", group_matrices)
        self._validate_signed_permutation_action()

    def _validate_signed_permutation_action(self) -> None:
        for matrix in self._group_matrices:
            rounded_matrix = matrix.round()
            is_signed_permutation = (
                torch.allclose(matrix, rounded_matrix)
                and torch.all((rounded_matrix == -1) | (rounded_matrix == 0) | (rounded_matrix == 1))
                and torch.all(rounded_matrix.abs().sum(dim=0) == 1)
                and torch.all(rounded_matrix.abs().sum(dim=1) == 1)
            )
            if not is_signed_permutation:
                raise ValueError(
                    "Empirical observation normalization is only guaranteed equivariant for signed-permutation "
                    "representations. Disable observation normalization for this symmetry representation."
                )

    @torch.jit.unused
    def update(self, observations: torch.Tensor) -> None:
        """Update statistics using the complete symmetry orbit of each sample."""
        orbit_observations = torch.einsum("gij,bj->gbi", self._group_matrices, observations)
        orbit_observations = orbit_observations.reshape(-1, observations.shape[-1])
        super().update(orbit_observations)


class _ExportedSymmetricActor(nn.Module):
    """Plain PyTorch deterministic actor used by deployment exporters."""

    def __init__(self, actor: EMLP, normalizer: nn.Module, num_actions: int) -> None:
        super().__init__()
        self.actor = actor.export()
        self.normalizer = normalizer
        self.num_actions = num_actions

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return only the mean portion of the actor output."""
        actor_output = self.actor(self.normalizer(observations))
        return actor_output[..., : self.num_actions]


class _ActionMean(nn.Module):
    """Select the action-mean portion of distribution parameters."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions

    def forward(self, actor_output: torch.Tensor) -> torch.Tensor:
        """Return the action means."""
        return actor_output[..., : self.num_actions]


class _IsaacLabExportPolicy(nn.Module):
    """Minimal plain-PyTorch policy interface expected by IsaacLab exporters."""

    is_recurrent = False

    def __init__(self, actor: EMLP, num_actions: int) -> None:
        super().__init__()
        exported_actor = actor.export()
        self.actor = nn.Sequential(*exported_actor.children(), _ActionMean(num_actions))


class ActorCriticSymm(ActorCritic):
    """Actor-critic with an equivariant actor and an invariant critic."""

    is_recurrent = False
    _ESCNN_CACHE_NAMES = frozenset({"matrix", "expanded_bias"})

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        small_init_output: bool = True,
        obs_space_names_actor: list[str] | tuple[str, ...] | None = None,
        obs_space_names_critic: list[str] | tuple[str, ...] | None = None,
        obs_space_names_single_state: list[str] | None = None,
        action_space_names: list[str] | tuple[str, ...] | None = None,
        joints_order: list[str] | None = None,
        robot_name: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the symmetric actor-critic.

        The observation-space name lists must describe the flattened tensors formed
        by the RSL-RL ``policy`` and ``critic`` observation groups, in exactly the
        same order.
        """
        nn.Module.__init__(self)
        if kwargs:
            print(
                "ActorCriticSymm.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs))
            )

        if obs_space_names_actor is None or obs_space_names_critic is None or action_space_names is None:
            raise ValueError(
                "obs_space_names_actor, obs_space_names_critic, and action_space_names must all be configured."
            )
        if joints_order is None or robot_name is None:
            raise ValueError("joints_order and robot_name must be configured.")
        if noise_std_type not in {"scalar", "log"}:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'.")
        if init_noise_std <= 0.0:
            raise ValueError(f"init_noise_std must be positive, got {init_noise_std}.")

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        actor_obs_dim = self._obs_dim(obs, obs_groups["policy"], "policy")
        critic_obs_dim = self._obs_dim(obs, obs_groups["critic"], "critic")

        # Load all representations together. Apart from avoiding duplicate robot
        # loads, this guarantees that actor, critic, and action types share the
        # exact same escnn group instance.
        all_space_names = list(
            dict.fromkeys([*obs_space_names_actor, *obs_space_names_critic, *obs_space_names_single_state, *action_space_names])
        )
        self.G, representations = configure_observation_space_representations(
            robot_name, all_space_names, joints_order
        )
        gspace = escnn.gspaces.no_base_space(self.G)
        self.num_replica = len(self.G.elements)
        self.actor_in_type = FieldType(gspace, [representations[name] for name in obs_space_names_actor])
        self.critic_in_type = FieldType(gspace, [representations[name] for name in obs_space_names_critic])
        self.state_type = FieldType(gspace, [representations[name] for name in obs_space_names_single_state])
        self.actor_out_type = FieldType(gspace, [representations[name] for name in action_space_names])

        self._validate_dimension("actor observations", actor_obs_dim, self.actor_in_type.size)
        self._validate_dimension("critic observations", critic_obs_dim, self.critic_in_type.size)
        self._validate_dimension("actions", num_actions, self.actor_out_type.size)

        self.action_gaussian = EquivMultivariateNormal(y_type=self.actor_out_type)
        actor_output_type = self.action_gaussian.in_type if state_dependent_std else self.actor_out_type
        self.actor = EMLP(
            in_type=self.actor_in_type,
            out_type=actor_output_type,
            bias=True,
            hidden_units=list(actor_hidden_dims),
            activation=activation,
        )
        self.critic = IMLP(
            in_type=self.critic_in_type,
            out_dim=1,
            bias=True,
            hidden_units=list(critic_hidden_dims),
            activation=activation,
        )
        if small_init_output:
            self._initialize_small_outputs()

        self.actor_obs_normalizer = (
            _EquivariantEmpiricalNormalization(self.actor_in_type) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            _EquivariantEmpiricalNormalization(self.critic_in_type) if critic_obs_normalization else nn.Identity()
        )

        # A state-independent covariance is the symmetric counterpart of
        # RSL-RL's default learned action noise. One parameter is maintained per
        # irreducible action subspace and EquivMultivariateNormal maps these to
        # component variances without breaking equivariance.
        if not state_dependent_std:
            noise_shape = (self.action_gaussian.n_cov_params,)
            if noise_std_type == "scalar":
                self.std = nn.Parameter(torch.full(noise_shape, init_noise_std))
            else:
                self.log_std = nn.Parameter(torch.full(noise_shape, float(np.log(init_noise_std))))

        self.distribution: torch.distributions.MultivariateNormal | None = None
        Normal.set_default_validate_args(False)

        actor_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print(f"Actor [{actor_params / 1e6:.2f}M params]:\n{self.actor}")
        print(f"Critic [{critic_params / 1e6:.2f}M params]:\n{self.critic}")

    def _initialize_small_outputs(self) -> None:
        """Start with small mean actions and a neutral value estimate."""
        with torch.no_grad():
            actor_head = self.actor.net[-1]
            actor_head.weights.mul_(0.2)
            if actor_head.bias is not None:
                actor_head.bias.zero_()

            self.critic.head.weight.zero_()
            if self.critic.head.bias is not None:
                self.critic.head.bias.zero_()

    @staticmethod
    def _obs_dim(obs: TensorDict, group_names: list[str], group_label: str) -> int:
        dimension = 0
        for name in group_names:
            if name not in obs.keys():
                raise KeyError(f"Observation group {name!r} from {group_label!r} is missing from the TensorDict.")
            if len(obs[name].shape) != 2:
                raise ValueError(
                    f"ActorCriticSymm only supports 1D observations; {name!r} has shape {tuple(obs[name].shape)}."
                )
            dimension += obs[name].shape[-1]
        return dimension

    @staticmethod
    def _validate_dimension(name: str, actual: int, represented: int) -> None:
        if actual != represented:
            raise ValueError(
                f"The configured representations describe {represented} {name}, but the environment provides "
                f"{actual}. Check the order and contents of the morphological symmetry configuration."
            )

    def forward(self) -> NoReturn:
        """Disallow calling the actor-critic without selecting actor or critic."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load learned state while ignoring escnn's mode-dependent linear caches.

        ``escnn.nn.Linear`` adds ``matrix`` and ``expanded_bias`` buffers in
        evaluation mode and removes them in training mode. Checkpoints therefore
        contain different keys depending on the mode in which they were saved,
        even though these buffers are derived from the learned parameters.
        """
        was_training = self.training
        self.train()
        learned_state = {
            key: value
            for key, value in state_dict.items()
            if key.rsplit(".", maxsplit=1)[-1] not in self._ESCNN_CACHE_NAMES
        }
        try:
            super().load_state_dict(learned_state, strict=strict)
        finally:
            self.train(was_training)
        return True

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Flatten the configured policy observation groups."""
        return torch.cat([obs[name] for name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Flatten the configured critic observation groups."""
        return torch.cat([obs[name] for name in self.obs_groups["critic"]], dim=-1)

    def _update_distribution(self, actor_observations: torch.Tensor) -> None:
        geometric_obs = self.actor_in_type(actor_observations)
        actor_output = self.actor(geometric_obs)
        if self.state_dependent_std:
            distribution_parameters = actor_output
        else:
            if self.noise_std_type == "scalar":
                log_variances = 2.0 * torch.log(self.std.clamp_min(1.0e-6))
            else:
                log_variances = 2.0 * self.log_std
            expanded_log_variances = log_variances.expand(*actor_output.tensor.shape[:-1], -1)
            distribution_parameters = self.action_gaussian.in_type(
                torch.cat((actor_output.tensor, expanded_log_variances), dim=-1)
            )
        self.distribution = self.action_gaussian.get_distribution(distribution_parameters)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Sample actions for a batch of observations."""
        actor_observations = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(actor_observations)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Return deterministic mean actions."""
        actor_observations = self.actor_obs_normalizer(self.get_actor_obs(obs))
        actor_output = self.actor(self.actor_in_type(actor_observations)).tensor
        return actor_output[..., : self.actor_out_type.size]

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the invariant value function."""
        critic_observations = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(self.critic_in_type(critic_observations)).tensor

    @property
    def action_mean(self) -> torch.Tensor:
        """Return the current distribution mean."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Return the current distribution component standard deviations."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Return entropy summed over the multivariate action event."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        return self.distribution.entropy()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return joint log probabilities for a batch of actions."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        return self.distribution.log_prob(actions)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update enabled empirical observation normalizers."""
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset recurrent state; the symmetric policy is feed-forward."""

    def export(self) -> nn.Module:
        """Export the deterministic actor network as a plain PyTorch module."""
        return _ExportedSymmetricActor(
            actor=self.actor,
            normalizer=self.actor_obs_normalizer,
            num_actions=self.actor_out_type.size,
        )

    def export_for_isaaclab(self) -> nn.Module:
        """Convert the actor to the plain policy interface used by IsaacLab exporters."""
        return _IsaacLabExportPolicy(self.actor, self.actor_out_type.size)
