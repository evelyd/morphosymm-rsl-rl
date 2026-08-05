# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import warnings
from tensordict import TensorDict


from morphosymm_rl.algorithms import PPO, PPODAEOnline
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from morphosymm_rl.modules import DAEActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger

import torch

import torch

def fill_replay_buffer(algorithm_instance, env_instance, state_dim, num_initial_steps=None):
    device = algorithm_instance.device

    if hasattr(algorithm_instance, 'storage'):
        num_transitions_per_env = algorithm_instance.storage.num_transitions_per_env
    else:
        num_transitions_per_env = algorithm_instance.num_steps_per_env

    if num_initial_steps is None:
        required_steps = algorithm_instance.replay_buffer.buffer_size
        steps_per_rollout = max(1, env_instance.num_envs * num_transitions_per_env)
        num_initial_rollouts = max(1, (required_steps + steps_per_rollout - 1) // steps_per_rollout)
    else:
        steps_per_rollout = max(1, env_instance.num_envs * num_transitions_per_env)
        num_initial_rollouts = max(1, (num_initial_steps + steps_per_rollout - 1) // steps_per_rollout)

    print(f"Initializing replay buffer with {num_initial_rollouts} full rollouts...")

    if hasattr(algorithm_instance, 'eval_mode'):
        algorithm_instance.eval_mode()
    elif hasattr(algorithm_instance, 'actor_critic'):
        algorithm_instance.actor_critic.eval()

    def inject_koopman_prediction(obs_dict):
        if isinstance(obs_dict, TensorDict):
            num_envs = obs_dict["policy"].shape[0]
            koopman_dim = env_instance.cfg.single_observation_space * algorithm_instance.koopman_cfg["obs_state_ratio"]
            zero_tensor = torch.zeros((num_envs, koopman_dim), device=device)

            # TensorDict requires setting keys using set() or direct dictionary set if unlocked
            obs_dict.set("koopman_prediction", zero_tensor)
        elif isinstance(obs_dict, dict):
            num_envs = obs_dict["policy"].shape[0]
            koopman_dim = env_instance.cfg.single_observation_space * env_instance.cfg.history_length
            obs_dict["koopman_prediction"] = torch.zeros((num_envs, koopman_dim), device=device)
        return obs_dict

    # Reset environment and inject koopman prediction
    env_instance.reset()
    obs = env_instance.get_observations()
    if isinstance(obs, dict):
        obs = {k: v.to(device) for k, v in obs.items()}
    obs = inject_koopman_prediction(obs)

    with torch.inference_mode():
        for rollout_idx in range(num_initial_rollouts):
            new_states, new_actions, new_next_states = [], [], []

            for _ in range(num_transitions_per_env):

                # 1. Take actions directly from the initialized Actor policy
                actions = algorithm_instance.act(obs)

                # 2. Extract DAE state
                critic_obs = obs["critic"]
                current_states_for_dae = critic_obs[:, state_dim*(algorithm_instance.history_length-1):state_dim*algorithm_instance.history_length].clone()
                current_actions_for_dae = actions.clone()

                # 3. Step Environment
                obs, rewards, dones, extras = env_instance.step(actions.to(env_instance.device))

                if isinstance(obs, dict):
                    obs = {k: v.to(device) for k, v in obs.items()}

                # --- INJECT ZERO TENSOR BEFORE PASSING TO PPO ---
                obs = inject_koopman_prediction(obs)

                # 4. Process step (stores the padded dict in PPO storage)
                algorithm_instance.process_env_step(obs, rewards.to(device), dones.to(device), extras)

                # 5. Extract next states for DAE
                critic_obs_next = obs["critic"]
                new_states.append(current_states_for_dae)
                new_actions.append(current_actions_for_dae)
                new_next_states.append(critic_obs_next[:, state_dim*(algorithm_instance.history_length-1):state_dim*algorithm_instance.history_length].clone())

                if hasattr(algorithm_instance, 'replay_buffer'):
                    algorithm_instance.replay_buffer.insert(
                        current_states_for_dae, current_actions_for_dae, new_next_states[-1]
                    )

            if hasattr(algorithm_instance, 'obs_action_normalizer') and new_states:
                algorithm_instance.obs_action_normalizer.update(torch.cat(new_states, dim=0), torch.cat(new_actions, dim=0))

            # 6. Compute returns using the padded dictionary!
            try:
                algorithm_instance.compute_returns(obs, actions)
            except TypeError:
                algorithm_instance.compute_returns(obs)

            if hasattr(algorithm_instance, 'storage'):
                algorithm_instance.storage.clear()

    print("Replay buffer initialization complete.")
    if hasattr(algorithm_instance, 'train_mode'):
        algorithm_instance.train_mode()
    elif hasattr(algorithm_instance, 'actor_critic'):
        algorithm_instance.actor_critic.train()

class DAEOnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env
        self.task = train_cfg["experiment_name"]

        # Koopman cfg
        self.koopman_cfg = train_cfg["koopman_cfg"]
        self.morphologycal_symmetries_cfg = train_cfg["morphologycal_symmetries_cfg"]

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        # Create the algorithm
        self.single_observation_space = self.env.cfg.single_observation_space
        self.history_length = self.env.cfg.history_length
        self.action_space = self.env.cfg.action_space
        self.dt = self.env.unwrapped.step_dt

        self.alg = self._construct_algorithm(obs)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

        # Setup for online DAE learning
        if "dae" in self.task:

            # Initialize the replay buffer
            if hasattr(self.alg, 'replay_buffer'): # and env.cfg.mode not in ["play", "test"]:
                fill_replay_buffer(self.alg, self.env, self.alg.state_dim) # in the buffer, states are only the state, not the full obs which has whatever history length of states

                # Perform initial update of normalizers
                batch_states_raw, batch_actions_raw, batch_next_states_raw, _, _ = self.alg.replay_buffer.sample(len(self.alg.replay_buffer), self.alg.replay_buffer.beta_initial)
                self.alg.obs_action_normalizer.update(batch_states_raw, batch_actions_raw)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()

            # Initialize lists to collect NEW data for online algorithms
            new_states_this_iter = []
            new_actions_this_iter = []
            new_next_states_this_iter = []

            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)

                    if "dae" in self.task:

                        # Collect data for DAE
                        current_critic_obs_for_dae = obs["critic"]
                        current_states_for_dae = current_critic_obs_for_dae[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length].clone()
                        current_actions_for_dae = actions.clone()

                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    if "dae" in self.task:
                        # Get the next states for the DAE
                        next_critic_obs_for_dae = obs["critic"]
                        next_states_for_dae = next_critic_obs_for_dae[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length].clone()

                        # Add to our new data collection
                        new_states_this_iter.append(current_states_for_dae)
                        new_actions_this_iter.append(current_actions_for_dae)
                        new_next_states_this_iter.append(next_states_for_dae)

                        # Fill the PER buffer
                        self.alg.replay_buffer.insert(
                            current_states_for_dae,
                            current_actions_for_dae,
                            next_states_for_dae,
                        )

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start

                if "dae" in self.task:
                    # Anneal beta for Importance Sampling weights
                    current_beta = self.alg.replay_buffer.beta_initial + (1.0 - self.alg.replay_buffer.beta_initial) * \
                                min(1.0, (it - self.current_learning_iteration) / self.alg.replay_buffer.beta_annealing_steps)

                    # Concatenate the new data
                    batch_states_new = torch.cat(new_states_this_iter, dim=0)
                    batch_actions_new = torch.cat(new_actions_this_iter, dim=0)
                    batch_next_states_new = torch.cat(new_next_states_this_iter, dim=0)

                    # Perform update of normalizers using only the new data
                    self.alg.obs_action_normalizer.update(batch_states_new, batch_actions_new)

                start = stop

                # Compute returns
                if "dae" in self.task:
                    self.alg.compute_returns(obs, actions)
                else:
                    self.alg.compute_returns(obs)

            if "dae" in self.task:
                # Perform DAE training step
                if len(self.alg.replay_buffer) >= self.koopman_cfg["mini_batch_size"]:
                    dae_training_start_time = time.time()

                    dae_num_mini_batches = self.koopman_cfg["num_mini_batches"]
                    dae_mini_batch_size = self.koopman_cfg["mini_batch_size"]

                    dae_losses_this_iter = []
                    dae_obs_pred_losses_this_iter = []
                    dae_state_rec_losses_this_iter = []
                    dae_state_pred_losses_this_iter = []

                    # Iterating over mini-batches for DAE training
                    for _ in range(dae_num_mini_batches):
                        # Sample from the Prioritized Replay Buffer
                        batch_states_raw, batch_actions_raw, batch_next_states_raw, batch_tree_indices, is_weights = \
                            self.alg.replay_buffer.sample(dae_mini_batch_size, current_beta)

                        # Transfer is_weights to cuda device
                        is_weights = is_weights.to(self.device)

                        sample_generator = self.alg.replay_buffer.preprocess_samples(
                            batch_states_raw, batch_actions_raw, batch_next_states_raw,
                            frames_per_step=self.koopman_cfg["frames_per_state"],
                            prediction_horizon=self.koopman_cfg["pred_horizon"]
                        )

                        all_state_observations = []
                        all_action_observations = []
                        all_next_state_observations = []
                        for sample in sample_generator:
                            all_state_observations.append(sample["state_observations"].unsqueeze(0))
                            all_action_observations.append(sample["action_observations"].unsqueeze(0))
                            all_next_state_observations.append(sample["next_state_observations"].unsqueeze(0))

                        # If preprocess_samples yielded no valid samples (e.g., traj too short), skip this mini-batch
                        if not all_state_observations:
                            print("Warning: No valid samples generated by preprocess_samples, skipping DAE mini-batch.")
                            continue

                        combined_state_observations = torch.cat(all_state_observations, dim=0).to(self.device)
                        combined_action_observations = torch.cat(all_action_observations, dim=0).to(self.device)
                        combined_next_state_observations = torch.cat(all_next_state_observations, dim=0).to(self.device)

                        # Use the combined_state_observations and combined_action_observations
                        # (which are the raw, un-normalized inputs) to update the statistics.
                        self.alg.obs_action_normalizer.update(
                            batch_states_raw,
                            batch_actions_raw,
                        )

                        # Move preprocessed and normalized batch to the correct device for the DAE model
                        batch = self.alg.replay_buffer.shape_states_actions(
                                combined_state_observations, combined_action_observations, combined_next_state_observations #TODO no norming
                        )

                        batch_on_device = {k: v.to(self.device) for k, v in batch.items()}

                        # Forward pass through DAE
                        if hasattr(self.alg.dae_model, 'action_dim') and self.alg.dae_model.action_dim > 0:
                            outputs = self.alg.dae_model(**batch_on_device)
                        else:
                            outputs = self.alg.dae_model(**batch_on_device)

                        # Compute DAE losses
                        dae_loss_per_sample, dae_metrics = self.alg.dae_model.compute_loss_and_metrics(**outputs, **batch_on_device)

                        # Apply importance sampling weights to the loss
                        actual_batch_size_for_loss = dae_loss_per_sample.shape[0]
                        if is_weights.shape[0] != actual_batch_size_for_loss:
                            is_weights_aligned = is_weights[:actual_batch_size_for_loss]
                        else:
                            is_weights_aligned = is_weights

                        weighted_dae_loss = (dae_loss_per_sample * is_weights_aligned).mean()

                        # Backpropagate and update DAE weights
                        self.alg.dae_optimizer.zero_grad()
                        weighted_dae_loss.backward()
                        self.alg.dae_optimizer.step()

                        # Update priorities in the replay buffer
                        # Use the per-sample losses as errors
                        dae_errors_for_priority_update = dae_loss_per_sample.detach().cpu().numpy()

                        # Ensure that the batch_tree_indices also aligns with the number of samples that actually generated a loss.
                        if len(batch_tree_indices) != actual_batch_size_for_loss:
                            batch_tree_indices_aligned = batch_tree_indices[:actual_batch_size_for_loss]
                        else:
                            batch_tree_indices_aligned = batch_tree_indices

                        # Prioritize samples based on the overall DAE loss
                        self.alg.replay_buffer.update_priorities(
                            batch_tree_indices_aligned,
                            dae_errors_for_priority_update
                        )

                        dae_losses_this_iter.append(weighted_dae_loss.item())
                        dae_obs_pred_losses_this_iter.append(dae_metrics["obs_pred_loss"].item())
                        dae_state_rec_losses_this_iter.append(dae_metrics["state_rec_loss"].item())
                        dae_state_pred_losses_this_iter.append(dae_metrics["state_pred_loss"].item())

                    if dae_losses_this_iter:
                        mean_dae_loss = sum(dae_losses_this_iter) / len(dae_losses_this_iter)
                        mean_dae_obs_pred_loss = sum(dae_obs_pred_losses_this_iter) / len(dae_obs_pred_losses_this_iter)
                        mean_dae_state_rec_loss = sum(dae_state_rec_losses_this_iter) / len(dae_state_rec_losses_this_iter)
                        mean_dae_state_pred_loss = sum(dae_state_pred_losses_this_iter) / len(dae_state_pred_losses_this_iter)
                    else:
                        mean_dae_loss = 0.0 # No batches trained
                        mean_dae_obs_pred_loss = 0.0
                        mean_dae_state_rec_loss = 0.0
                        mean_dae_state_pred_loss = 0.0

                    dae_train_time = time.time() - dae_training_start_time


            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if "dae" in self.task:
                loss_dict["dae_loss"] = mean_dae_loss
                loss_dict["dae_obs_pred_loss"] = mean_dae_obs_pred_loss
                loss_dict["dae_state_rec_loss"] = mean_dae_state_rec_loss
                loss_dict["dae_state_pred_loss"] = mean_dae_state_pred_loss
                loss_dict["dae_train_time"] = dae_train_time

            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )

            # Save model
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save main policy and optimizer
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()

        # Save DAE Models and Normalizer
        if "dae" in self.task:
            saved_dict["dae_state_dict"] = self.alg.dae_model.state_dict()
            saved_dict["dae_optimizer_state_dict"] = self.alg.dae_optimizer.state_dict()
            saved_dict["normalizer_state_dict"] = self.alg.obs_action_normalizer.state_dict()

        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
            loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

            # Load main policy
            resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

            # Load RND model if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

            # Load main optimizer if used
            if load_optimizer and resumed_training:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                if self.alg_cfg["rnd_cfg"]:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

            # Load DAE Models and Normalizer
            if "dae" in self.task and "dae_state_dict" in loaded_dict:
                dae_state = loaded_dict["dae_state_dict"]

                # Standard cdae loads perfectly with strict enforcement
                self.alg.dae_model.load_state_dict(dae_state)

                if load_optimizer and "dae_optimizer_state_dict" in loaded_dict:
                    self.alg.dae_optimizer.load_state_dict(loaded_dict["dae_optimizer_state_dict"])

                if "normalizer_state_dict" in loaded_dict:
                    self.alg.obs_action_normalizer.load_state_dict(loaded_dict["normalizer_state_dict"])

            # Load current learning iteration
            if resumed_training:
                self.current_learning_iteration = loaded_dict["iter"]

            return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.train()
        # DAE
        if "dae" in self.task:
                self.alg.dae_model.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _get_default_obs_sets(self) -> list[str]:
        """Get the the default observation sets required for the algorithm.

        .. note::
            See :func:`resolve_obs_groups` for more details on the handling of observation sets.
        """
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        return default_sets

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> PPO | PPODAEOnline:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Set up the Koopman prediction in the obs
        self.cfg["obs_groups"]["critic"].append("koopman_prediction")

        # Set the shape of the Koopman prediction obs
        koopman_dim = self.single_observation_space * self.koopman_cfg["obs_state_ratio"]
        num_envs = obs["policy"].shape[0]
        obs["koopman_prediction"] = torch.zeros((num_envs, koopman_dim), device=self.device)
        # Initialize the policy
        if self.policy_cfg["class_name"] == "DAEActorCritic":
            self.policy_cfg.pop("class_name")
            actor_critic: DAEActorCritic = DAEActorCritic(
                            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
                        ).to(self.device)
        else:
            actor_critic_class = resolve_callable(self.policy_cfg.pop("class_name"))
            actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN = actor_critic_class(
                obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
            ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Initialize the algorithm
        if self.alg_cfg["class_name"] == "PPODAEOnline":
            self.alg_cfg.pop("class_name")  # Remove class name from config to avoid passing it to the constructor
            alg: PPODAEOnline = PPODAEOnline(
                actor_critic, storage, device=self.device, task=self.task, dt=self.dt, single_observation_space=self.single_observation_space, action_space=self.action_space, history_length=self.history_length, morphologycal_symmetries_cfg=self.morphologycal_symmetries_cfg, koopman_cfg=self.koopman_cfg, **self.alg_cfg
            )
        else:
            alg_class = resolve_callable(self.alg_cfg.pop("class_name"))
            alg: PPO = alg_class(
                actor_critic, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
            )

        return alg