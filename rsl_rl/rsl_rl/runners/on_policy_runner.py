# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
from typing import Union

from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from rsl_rl.algorithms import PPO, PPOAugmented, PPODAE, PPODAEOnline, PPODAELatentOnly, PPODAEOnlineLatentOnly
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticSymm
from rsl_rl.env import VecEnv

from legged_gym.utils.helpers import fill_replay_buffer

from dha.utils.mysc import flatten_dict

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.use_wandb = train_cfg["use_wandb"]
        self.task = self.cfg["experiment_name"]
        if "online" in self.task:
            self.koopman_cfg = train_cfg["koopman"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if "dae" in self.task and "online" in self.task:
            if "latent_only" in self.task:
                num_extra_obs = self.env.num_privileged_obs - self.env.num_obs if self.env.num_privileged_obs is not None else 0
                num_critic_obs = num_extra_obs + self.koopman_cfg["robot"]["obs_state_ratio"] * self.koopman_cfg["robot"]["state_dim"]
            else:
                num_critic_obs += self.koopman_cfg['robot']['state_dim'] * self.koopman_cfg['robot']['obs_state_ratio']
        elif "dae" in self.task:
            if "push_door" in self.task:
                num_critic_obs += 129
            else:
                num_critic_obs += 141
            self.model_path = self.cfg["model_path"]
        # input(f"num_critic_obs: {num_critic_obs}, num_obs: {self.env.num_obs}, num_privileged_obs: {self.env.num_privileged_obs}, num_actions: {self.env.num_actions}")
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: Union[ActorCritic | ActorCriticSymm] = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        self.task,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        # input(f"Using algorithm: {alg_class.__name__}, Policy: {actor_critic.__class__.__name__}, Task: {self.task}")
        if "ppodaeonline" in alg_class.__name__.lower():
            self.alg = alg_class(actor_critic, self.task, self.koopman_cfg,  dt=env.dt, device=self.device, **self.alg_cfg)
        elif "ppodae" in alg_class.__name__.lower():
            self.alg = alg_class(actor_critic, self.task, model_path=self.model_path, device=self.device, **self.alg_cfg)
        else:
            self.alg = alg_class(actor_critic, self.task, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Initialize the replay buffer
        if "online" in self.task:
            if hasattr(self.alg, 'replay_buffer') and env.cfg.mode not in ["play", "test"]:
                fill_replay_buffer(self.alg, self.env, self.alg.state_dim)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self.alg.actor_critic.train() # switch to train mode (for dropout for example)
            # Set DAE to train mode also
            if "online" in self.task:
                self.alg.dae_model.train()
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    if "online" in self.task:
                        # Collect data for DAE
                        current_states_for_dae = obs[:, -self.alg.state_dim:].clone()
                        current_actions_for_dae = actions.clone()

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    if "online" in self.task:
                        # Get the next states for the DAE
                        next_states_for_dae = obs[:, -self.alg.state_dim:].clone()

                        # Fill the PER buffer
                        self.alg.replay_buffer.insert(
                            current_states_for_dae.cpu(),
                            current_actions_for_dae.cpu(),
                            next_states_for_dae.cpu()
                        )

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop

                if "online" in self.task:
                    self.alg.compute_returns(critic_obs, actions)
                else:
                    self.alg.compute_returns(critic_obs)

            if "online" in self.task:
                # Perform DAE training step
                mean_dae_loss = 0.0
                dae_train_time = 0.0
                mean_dae_obs_pred_loss = 0.0
                mean_dae_state_rec_loss = 0.0
                mean_dae_state_pred_loss = 0.0

                if len(self.alg.replay_buffer) >= self.koopman_cfg["model"]["mini_batch_size"]:
                    dae_training_start_time = time.time()

                    dae_num_mini_batches = self.koopman_cfg["model"]["num_mini_batches"]
                    dae_mini_batch_size = self.koopman_cfg["model"]["mini_batch_size"]

                    dae_losses_this_iter = []
                    dae_obs_pred_losses_this_iter = []
                    dae_state_rec_losses_this_iter = []
                    dae_state_pred_losses_this_iter = []

                    # Anneal beta for Importance Sampling weights
                    current_beta = self.alg.replay_buffer.beta_initial + (1.0 - self.alg.replay_buffer.beta_initial) * \
                                min(1.0, (it - self.current_learning_iteration) / self.alg.replay_buffer.beta_annealing_steps)


                    # Iterating over mini-batches for DAE training
                    for _ in range(dae_num_mini_batches):
                        # Sample from the Prioritized Replay Buffer
                        batch_states_raw, batch_actions_raw, batch_next_states_raw, batch_tree_indices, is_weights = \
                            self.alg.replay_buffer.sample(dae_mini_batch_size, current_beta)

                        # Transfer is_weights to cuda device
                        is_weights = is_weights.to(self.device)

                        sample_generator = self.alg.replay_buffer.preprocess_samples(
                            batch_states_raw, batch_actions_raw, batch_next_states_raw,
                            frames_per_step=self.koopman_cfg["robot"]["frames_per_state"],
                            prediction_horizon=self.koopman_cfg["robot"]["pred_horizon"]
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

                        # Normalize the observations and actions
                        normed_states, normed_actions = self.alg.obs_action_normalizer.normalize(
                            combined_state_observations, combined_action_observations
                        )
                        next_normed_states = self.alg.obs_action_normalizer.normalize(
                            combined_next_state_observations
                        )

                        # Move preprocessed and normalized batch to the correct device for the DAE model
                        batch = self.alg.replay_buffer.shape_states_actions(
                                normed_states, normed_actions, next_normed_states
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

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                locs = locals()
                if "online" in self.task:
                    locs["mean_dae_loss"] = mean_dae_loss
                    locs["mean_dae_obs_pred_loss"] = mean_dae_obs_pred_loss
                    locs["mean_dae_state_rec_loss"] = mean_dae_state_rec_loss
                    locs["mean_dae_state_pred_loss"] = mean_dae_state_pred_loss
                    locs["dae_train_time"] = dae_train_time
                self.log(locs)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                if "online" in self.task:
                    dae_save_dict = {
                        'dae_state_dict': self.alg.dae_model.state_dict(),
                        'normalizer_state_dict': self.alg.obs_action_normalizer.state_dict()
                    }
                    torch.save(dae_save_dict, os.path.join(self.log_dir, 'dae_model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        if "online" in self.task:
            dae_save_dict = {
                'dae_state_dict': self.alg.dae_model.state_dict(),
                'normalizer_state_dict': self.alg.obs_action_normalizer.state_dict()
            }
            torch.save(dae_save_dict, os.path.join(self.log_dir, 'dae_model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_log_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                wandb_log_dict[f'Episode/{key}'] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if "online" in self.task:
            self.writer.add_scalar("DAE/loss", locs["mean_dae_loss"], locs["it"]) # or self.current_learning_iteration
            self.writer.add_scalar("DAE/obs_pred_loss", locs["mean_dae_obs_pred_loss"], locs["it"])
            self.writer.add_scalar("DAE/state_rec_loss", locs["mean_dae_state_rec_loss"], locs["it"])
            self.writer.add_scalar("DAE/state_pred_loss", locs["mean_dae_state_pred_loss"], locs["it"])
            self.writer.add_scalar("DAE/train_time", locs["dae_train_time"], locs["it"])
        wandb_log_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_log_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_log_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_log_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_log_dict['Perf/total_fps'] = fps
        wandb_log_dict['Perf/collection time'] = locs['collection_time']
        wandb_log_dict['Perf/learning_time'] = locs['learn_time']
        if "online" in self.task:
            wandb_log_dict["DAE/loss"] = locs["mean_dae_loss"]
            wandb_log_dict["DAE/obs_pred_loss"] = locs["mean_dae_obs_pred_loss"]
            wandb_log_dict["DAE/state_rec_loss"] = locs["mean_dae_state_rec_loss"]
            wandb_log_dict["DAE/state_pred_loss"] = locs["mean_dae_state_pred_loss"]
            wandb_log_dict["DAE/train_time"] = locs["dae_train_time"]
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            wandb_log_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_log_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        if self.use_wandb:
            wandb.log(wandb_log_dict, step=self.tot_timesteps)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        self.alg.actor_critic.eval() # switch to eval mode for saving the model
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        # if load_optimizer:
        #     self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
