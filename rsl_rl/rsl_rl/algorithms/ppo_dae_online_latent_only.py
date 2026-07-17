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
# and/or other materials provided from the distribution.
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

import os
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from dha.utils.utils import get_trained_dae_model,load_normalization_stats, safe_standardize
from escnn.nn import GeometricTensor
import legged_gym

# Assuming ppo.py is in the same directory or accessible via PYTHONPATH
from rsl_rl.algorithms import PPODAEOnline
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.storage.replay_buffer import RunningStdScaler

from dha.utils.utils import initialize_dae_model

class PPODAEOnlineLatentOnly(PPODAEOnline):
    def __init__(self,
                 actor_critic,
                 task,
                 koopman_cfg,
                 dt,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 replay_buffer_size=100000,
                 ):

        # Initialize the base PPO class first
        super().__init__(
            actor_critic=actor_critic,
            task=task,
            koopman_cfg=koopman_cfg,
            dt=dt,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            replay_buffer_size=replay_buffer_size,
        )

    def get_critic_input(self, critic_obs):
        """Processes critic_obs through DAE to get augmented input for the critic."""
        if "push_door" in self.task:
            dae_input = critic_obs[:, 172:215]  # most recent state
            extra_obs = critic_obs[:, 215:]
        else:
            dae_input = critic_obs[:, 94:141] # most recent state
            extra_obs = critic_obs[:, 141:]
        dae_input = dae_input.to(dtype=next(self.dae_model.parameters()).dtype)
        dae_input = dae_input.to(device=next(self.dae_model.parameters()).device)

        # dae_input_normed = safe_standardize(dae_input, self.state_mean, self.state_std)
        dae_input_normed = self.actor_obs_normalizer.normalize_states(dae_input)

        # Wrap as GeometricTensor for E-DAE/EC-DAE
        if "edae" in self.task or "ecdae" in self.task:
            dae_input_normed = GeometricTensor(dae_input_normed, self.dae_model.obs_fn.in_type)
            latent = self.dae_model.obs_fn(dae_input_normed).tensor.detach()
        else:
            latent = self.dae_model.obs_fn(dae_input_normed).detach()
        return torch.cat([extra_obs, latent], dim=-1)