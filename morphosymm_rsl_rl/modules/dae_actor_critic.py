from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import resolve_callable


class DAEModel(MLPModel):
    """RSL-RL v5.4.2 MLP model for online DAE, natively supporting both
    actor (policy distribution) and critic (value estimation) roles.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Clean up any custom keyword arguments not recognized by base MLPModel
        kwargs.pop("morphologycal_symmetries_cfg", None)
        kwargs.pop("koopman_cfg", None)

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            **kwargs,
        )
        self.obs_set = obs_set

    @property
    def action_mean(self) -> torch.Tensor:
        """Return action mean for actor models."""
        if self.obs_set == "actor" and hasattr(self, "distribution") and self.distribution is not None:
            return self.distribution.mean
        raise AttributeError("action_mean is only available for actor models.")

    @property
    def action_std(self) -> torch.Tensor:
        """Return action standard deviation for actor models."""
        if self.obs_set == "actor" and hasattr(self, "distribution") and self.distribution is not None:
            return self.distribution.std
        raise AttributeError("action_std is only available for actor models.")

    @property
    def entropy(self) -> torch.Tensor:
        """Return distribution entropy for actor models."""
        if self.obs_set == "actor" and hasattr(self, "distribution") and self.distribution is not None:
            return self.distribution.entropy
        raise AttributeError("entropy is only available for actor models.")

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        """Sample actions from the policy distribution."""
        if self.obs_set != "actor":
            raise RuntimeError("`act` can only be called on an actor model.")
        mlp_output = super().forward(obs)
        if hasattr(self, "distribution") and self.distribution is not None:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return mlp_output

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Run deterministic inference."""
        mlp_output = super().forward(obs)
        if self.obs_set == "actor" and hasattr(self, "distribution") and self.distribution is not None:
            if hasattr(self.distribution, "deterministic_output"):
                return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def evaluate(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        """Evaluate state values (critic role)."""
        if self.obs_set != "critic":
            raise RuntimeError("`evaluate` can only be called on a critic model.")
        return super().forward(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of given actions."""
        if self.obs_set == "actor" and hasattr(self, "distribution") and self.distribution is not None:
            return self.distribution.log_prob(actions)
        raise RuntimeError("`get_actions_log_prob` is only available for actor models.")

    def update_distribution(self, obs: TensorDict) -> None:
        """Update internal distribution parameters based on current observations."""
        if self.obs_set == "actor" and hasattr(self, "distribution") and self.distribution is not None:
            mlp_output = super().forward(obs)
            self.distribution.update(mlp_output)