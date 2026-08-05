# rsl_rl/modules/actor_critic.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.networks import MLP, EmpiricalNormalization

class DAEActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs: dict[str, torch.Tensor],
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_extra_critic_obs: int = 0,  # Added for DAE Latent augmentation
        **kwargs,
    ):
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"]) + num_extra_critic_obs

        # Actor
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else torch.nn.Identity()

        # Critic (dimension handles raw critic obs + DAE latent)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else torch.nn.Identity()

        # Action noise & distribution
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs: dict[str, torch.Tensor]):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        mean = self.actor(actor_obs)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        self.distribution = Normal(mean, std)

    def act(self, obs: dict[str, torch.Tensor], **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: dict[str, torch.Tensor]):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs)

    def evaluate(self, obs: dict[str, torch.Tensor], **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: dict[str, torch.Tensor]):
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: dict[str, torch.Tensor]):
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: dict[str, torch.Tensor]):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))