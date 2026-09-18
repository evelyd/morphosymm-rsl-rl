# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan


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
            koopman_dim = env_instance.cfg.single_observation_space * algorithm_instance.koopman_cfg.get("obs_state_ratio", 1)
            zero_tensor = torch.zeros((num_envs, koopman_dim), device=device)
            obs_dict.set("koopman_prediction", zero_tensor)
        elif isinstance(obs_dict, dict):
            num_envs = obs_dict["policy"].shape[0]
            koopman_dim = env_instance.cfg.single_observation_space * env_instance.cfg.history_length
            obs_dict["koopman_prediction"] = torch.zeros((num_envs, koopman_dim), device=device)
        return obs_dict

    env_instance.reset()
    obs = env_instance.get_observations()
    if isinstance(obs, dict):
        obs = {k: v.to(device) for k, v in obs.items()}
    obs = inject_koopman_prediction(obs)

    with torch.inference_mode():
        for rollout_idx in range(num_initial_rollouts):
            new_states, new_actions, new_next_states = [], [], []

            for _ in range(num_transitions_per_env):
                actions = algorithm_instance.act(obs)

                critic_obs = obs["critic"]
                current_states_for_dae = critic_obs[:, state_dim*(algorithm_instance.history_length-1):state_dim*algorithm_instance.history_length].clone()
                current_actions_for_dae = actions.clone()

                obs, rewards, dones, extras = env_instance.step(actions.to(env_instance.device))
                if isinstance(obs, dict):
                    obs = {k: v.to(device) for k, v in obs.items()}

                obs = inject_koopman_prediction(obs)
                algorithm_instance.process_env_step(obs, rewards.to(device), dones.to(device), extras)

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


class DAEOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of standard actor-critic methods with DAE/RFF integration."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.task = train_cfg.get("experiment_name", "rff")
        self.koopman_cfg = train_cfg.get("koopman_cfg", {})
        self.morphologycal_symmetries_cfg = train_cfg.get("morphologycal_symmetries_cfg", {})

        # Intercept the class name to cleanly resolve the DAE/RFF algorithms
        algorithm_class_name = train_cfg["algorithm"]["class_name"].strip()
        if algorithm_class_name.endswith("PPODAEOnline"):
            train_cfg["algorithm"]["class_name"] = "morphosymm_rsl_rl.algorithms.ppo_dae_online:PPODAEOnline"
        elif algorithm_class_name.endswith("PPORFF"):
            train_cfg["algorithm"]["class_name"] = "morphosymm_rsl_rl.algorithms.ppo_rff:PPORFF"

        # Delegate initialization, logging, multi-gpu, and alg factory to OnPolicyRunner
        super().__init__(env, train_cfg, log_dir, device)

        self.single_observation_space = self.env.cfg.single_observation_space
        self.history_length = self.env.cfg.history_length
        self.action_space = self.env.cfg.action_space
        self.dt = self.env.unwrapped.step_dt

        # Setup for online DAE/RFF learning
        if "dae" in self.task or "rff" in self.task:
            if hasattr(self.alg, 'replay_buffer'):
                if "rff" in self.task:
                    correct_koopman_dim = self.koopman_cfg.get("m", 129)
                else:
                    correct_koopman_dim = self.single_observation_space * self.koopman_cfg.get("obs_state_ratio", 1)

                fill_replay_buffer(self.alg, self.env, self.alg.state_dim, correct_koopman_dim)
                batch_states_raw, batch_actions_raw, batch_next_states_raw, _, _ = self.alg.replay_buffer.sample(len(self.alg.replay_buffer), self.alg.replay_buffer.beta_initial)
                self.alg.obs_action_normalizer.update(batch_states_raw, batch_actions_raw)

                if "rff" in self.task:
                    batch_latent_states = self.alg.rff(batch_states_raw.to(self.device))
                    self.alg.latent_normalizer.update(batch_latent_states)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Custom learning loop integrating online DAE/Koopman training updates."""
        if "rff" in self.task and self.logger.log_dir is not None:
            rff_path = os.path.join(self.logger.log_dir, 'rff.pt')
            torch.save({
                'rff_state_dict': self.alg.rff.state_dict(),
                'rff_config': {
                    'in_features': self.alg.rff.in_features,
                    'sigma': self.alg.rff.sigma,
                    'kernel_type': self.alg.rff.kernel_type,
                    'm': self.alg.rff.m,
                }
            }, rff_path)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            new_states_this_iter, new_actions_this_iter, new_next_states_this_iter = [], [], []

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)

                    if "dae" in self.task or "rff" in self.task:
                        current_critic_obs_for_dae = obs["critic"]
                        current_states_for_dae = current_critic_obs_for_dae[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length].clone()
                        current_actions_for_dae = actions.clone()

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)

                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    if "dae" in self.task or "rff" in self.task:
                        next_critic_obs_for_dae = obs["critic"]
                        next_states_for_dae = next_critic_obs_for_dae[:, self.single_observation_space*(self.history_length-1):self.single_observation_space*self.history_length].clone()
                        new_states_this_iter.append(current_states_for_dae)
                        new_actions_this_iter.append(current_actions_for_dae)
                        new_next_states_this_iter.append(next_states_for_dae)
                        self.alg.replay_buffer.insert(current_states_for_dae, current_actions_for_dae, next_states_for_dae)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start

                if "dae" in self.task or "rff" in self.task:
                    current_beta = self.alg.replay_buffer.beta_initial + (1.0 - self.alg.replay_buffer.beta_initial) * \
                                min(1.0, (it - self.current_learning_iteration) / self.alg.replay_buffer.beta_annealing_steps)
                    batch_states_new = torch.cat(new_states_this_iter, dim=0)
                    batch_actions_new = torch.cat(new_actions_this_iter, dim=0)
                    batch_next_states_new = torch.cat(new_next_states_this_iter, dim=0)

                    self.alg.obs_action_normalizer.update(batch_states_new, batch_actions_new)
                    if "rff" in self.task:
                        batch_latent_states_new = self.alg.rff(batch_states_new)
                        self.alg.latent_normalizer.update(batch_latent_states_new)

                start = stop
                if "dae" in self.task or "koopman" in self.task:
                    self.alg.compute_returns(obs, actions)
                else:
                    self.alg.compute_returns(obs)

            if "rff_koopman" in self.task:
                koopman_computation_start_time = time.time()
                self.alg.koopman_estimator.compute_koopman_op(batch_states_new, batch_actions_new, batch_next_states_new)
                pred_error = self.alg.koopman_estimator.compute_pred_error(batch_states_new, batch_actions_new, batch_next_states_new)
                koopman_computation_time = time.time() - koopman_computation_start_time

            if "dae" in self.task:
                if len(self.alg.replay_buffer) >= self.koopman_cfg["mini_batch_size"]:
                    dae_training_start_time = time.time()
                    dae_num_mini_batches = self.koopman_cfg["num_mini_batches"]
                    dae_mini_batch_size = self.koopman_cfg["mini_batch_size"]
                    dae_losses_this_iter, dae_obs_pred_losses_this_iter, dae_state_rec_losses_this_iter, dae_state_pred_losses_this_iter = [], [], [], []

                    for _ in range(dae_num_mini_batches):
                        batch_states_raw, batch_actions_raw, batch_next_states_raw, batch_tree_indices, is_weights = \
                            self.alg.replay_buffer.sample(dae_mini_batch_size, current_beta)
                        is_weights = is_weights.to(self.device)

                        sample_generator = self.alg.replay_buffer.preprocess_samples(
                            batch_states_raw, batch_actions_raw, batch_next_states_raw,
                            frames_per_step=self.koopman_cfg["frames_per_state"], prediction_horizon=self.koopman_cfg["pred_horizon"]
                        )

                        all_state_observations, all_action_observations, all_next_state_observations = [], [], []
                        for sample in sample_generator:
                            all_state_observations.append(sample["state_observations"].unsqueeze(0))
                            all_action_observations.append(sample["action_observations"].unsqueeze(0))
                            all_next_state_observations.append(sample["next_state_observations"].unsqueeze(0))

                        if not all_state_observations:
                            continue

                        combined_state_observations = torch.cat(all_state_observations, dim=0).to(self.device)
                        combined_action_observations = torch.cat(all_action_observations, dim=0).to(self.device)
                        combined_next_state_observations = torch.cat(all_next_state_observations, dim=0).to(self.device)

                        self.alg.obs_action_normalizer.update(batch_states_raw, batch_actions_raw)
                        batch = self.alg.replay_buffer.shape_states_actions(combined_state_observations, combined_action_observations, combined_next_state_observations)
                        batch_on_device = {k: v.to(self.device) for k, v in batch.items()}

                        outputs = self.alg.dae_model(**batch_on_device)
                        dae_loss_per_sample, dae_metrics = self.alg.dae_model.compute_loss_and_metrics(**outputs, **batch_on_device)

                        actual_batch_size_for_loss = dae_loss_per_sample.shape[0]
                        is_weights_aligned = is_weights[:actual_batch_size_for_loss] if is_weights.shape[0] != actual_batch_size_for_loss else is_weights
                        weighted_dae_loss = (dae_loss_per_sample * is_weights_aligned).mean()

                        self.alg.dae_optimizer.zero_grad()
                        weighted_dae_loss.backward()
                        self.alg.dae_optimizer.step()

                        dae_errors_for_priority_update = dae_loss_per_sample.detach().cpu().numpy()
                        batch_tree_indices_aligned = batch_tree_indices[:actual_batch_size_for_loss] if len(batch_tree_indices) != actual_batch_size_for_loss else batch_tree_indices
                        self.alg.replay_buffer.update_priorities(batch_tree_indices_aligned, dae_errors_for_priority_update)

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
                        mean_dae_loss, mean_dae_obs_pred_loss, mean_dae_state_rec_loss, mean_dae_state_pred_loss = 0.0, 0.0, 0.0, 0.0

                    dae_train_time = time.time() - dae_training_start_time

            loss_dict = self.alg.update()
            learn_time = time.time() - start

            if "dae" in self.task:
                loss_dict.update({"dae_loss": mean_dae_loss, "dae_obs_pred_loss": mean_dae_obs_pred_loss,
                                  "dae_state_rec_loss": mean_dae_state_rec_loss, "dae_state_pred_loss": mean_dae_state_pred_loss,
                                  "dae_train_time": dae_train_time})

            if "rff_koopman" in self.task:
                a_matrix = self.alg.koopman_estimator.K_matrix[:, :self.alg.koopman_estimator.feature_dim].detach()
                eigvals = torch.linalg.eigvals(a_matrix)
                loss_dict.update({"koopman_computation_time": koopman_computation_time, "koopman_pred_error": pred_error,
                                  "max_eigval": torch.max(torch.abs(eigvals)).item(), "min_eigval": torch.min(torch.abs(eigvals)).item()})

            self.current_learning_iteration = it

            # Use v5.4.2 property logic to pull standard deviation natively
            action_std = self.alg.actor.output_distribution_params[1] if hasattr(self.alg.actor, 'output_distribution_params') else None

            self.logger.log(
                it=it, start_it=start_it, total_it=total_it, collect_time=collect_time, learn_time=learn_time,
                loss_dict=loss_dict, learning_rate=self.alg.learning_rate, action_std=action_std,
                rnd_weight=self.alg.rnd.weight if getattr(self.alg, 'rnd', None) else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()