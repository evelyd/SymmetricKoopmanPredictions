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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .cyberdog2.c2_standdance_config import CyberStandDanceConfig, CyberStandDanceConfigLatentOnly, CyberStandDanceCfgPPO, CyberStandDanceCfgPPOAug, CyberStandDanceCfgPPOEMLP, CyberStandDanceCfgPPODAE, CyberStandDanceCfgPPOEMLPEDAE, CyberStandDanceCfgPPOCDAE, CyberStandDanceCfgPPOCDAEOnline, CyberStandDanceCfgPPOCDAEOnlineNextLatent, CyberStandDanceCfgPPOCDAEOnlineLatentOnly, CyberStandDanceCfgPPOECDAE, CyberStandDanceCfgPPOEMLPECDAE, CyberStandDanceCfgPPOEMLPECDAEOnline, CyberStandDanceCfgPPOEMLPECDAEOnlineNextLatent, CyberStandDanceCfgPPOEMLPCDAE, CyberStandDanceCfgPPOEMLPDAE, CyberStandDanceCfgPPOCDAELatentOnly, CyberStandDanceCfgPPOEMLPCDAELatentOnly, CyberStandDanceCfgPPOECDAELatentOnly, CyberStandDanceCfgPPOEMLPECDAELatentOnly
from .cyberdog2.c2_standdance_env import CyberStandDanceEnv

from .cyberdog2.c2_pushdoor_config import CyberPushDoorConfig, CyberPushDoorCfgPPO, CyberPushDoorCfgPPOAug, CyberPushDoorCfgPPOEMLP, CyberPushDoorCfgPPOCDAE, CyberPushDoorCfgPPOCDAEOnline, CyberPushDoorCfgPPOCDAEOnlineNextLatent, CyberPushDoorCfgPPOECDAE, CyberPushDoorCfgPPOEMLPCDAE, CyberPushDoorCfgPPOEMLPECDAE, CyberPushDoorCfgPPOEMLPECDAEOnline, CyberPushDoorCfgPPOEMLPECDAEOnlineNextLatent
from .cyberdog2.c2_pushdoor_env import CyberPushDoorEnv

from .cyberdog2.c2_walkslope_config import CyberWalkSlopeConfig, CyberWalkSlopeCfgPPO, CyberWalkSlopeCfgPPOAug, CyberWalkSlopeCfgPPOEMLP, CyberWalkSlopeCfgPPOCDAE, CyberWalkSlopeCfgPPOCDAEOnline, CyberWalkSlopeCfgPPOCDAEOnlineNextLatent, CyberWalkSlopeCfgPPOECDAE, CyberWalkSlopeCfgPPOEMLPCDAE, CyberWalkSlopeCfgPPOEMLPECDAE, CyberWalkSlopeCfgPPOEMLPECDAEOnline, CyberWalkSlopeCfgPPOEMLPECDAEOnlineNextLatent
from .cyberdog2.c2_walkslope_env import CyberWalkSlopeEnv

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register("cyber2_stand_dance", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPO())
task_registry.register("cyber2_stand_dance_aug", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOAug())
task_registry.register("cyber2_stand_dance_emlp", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLP())
task_registry.register("cyber2_stand_dance_dae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPODAE())
task_registry.register("cyber2_stand_dance_cdae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOCDAE())
task_registry.register("cyber2_stand_dance_cdae_online", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOCDAEOnline())
task_registry.register("cyber2_stand_dance_cdae_online_next_latent", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOCDAEOnlineNextLatent())
task_registry.register("cyber2_stand_dance_cdae_latent_only", CyberStandDanceEnv, CyberStandDanceConfigLatentOnly(), CyberStandDanceCfgPPOCDAELatentOnly())
task_registry.register("cyber2_stand_dance_cdae_online_latent_only", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOCDAEOnlineLatentOnly())
task_registry.register("cyber2_stand_dance_edae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLPEDAE())
task_registry.register("cyber2_stand_dance_ecdae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOECDAE())
task_registry.register("cyber2_stand_dance_ecdae_latent_only", CyberStandDanceEnv, CyberStandDanceConfigLatentOnly(), CyberStandDanceCfgPPOECDAELatentOnly())
task_registry.register("cyber2_stand_dance_emlp_dae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLPDAE())
task_registry.register("cyber2_stand_dance_emlp_edae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLPEDAE())
task_registry.register("cyber2_stand_dance_emlp_cdae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLPCDAE())
task_registry.register("cyber2_stand_dance_emlp_cdae_latent_only", CyberStandDanceEnv, CyberStandDanceConfigLatentOnly(), CyberStandDanceCfgPPOEMLPCDAELatentOnly())
task_registry.register("cyber2_stand_dance_emlp_ecdae", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLPECDAE())
task_registry.register("cyber2_stand_dance_emlp_ecdae_online", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLPECDAEOnline())
task_registry.register("cyber2_stand_dance_emlp_ecdae_online_next_latent", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOEMLPECDAEOnlineNextLatent())
task_registry.register("cyber2_stand_dance_emlp_ecdae_latent_only", CyberStandDanceEnv, CyberStandDanceConfigLatentOnly(), CyberStandDanceCfgPPOEMLPECDAELatentOnly())

task_registry.register("cyber2_push_door", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPO())
task_registry.register("cyber2_push_door_aug", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOAug())
task_registry.register("cyber2_push_door_emlp", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOEMLP())
task_registry.register("cyber2_push_door_cdae", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOCDAE())
task_registry.register("cyber2_push_door_cdae_online", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOCDAEOnline())
task_registry.register("cyber2_push_door_cdae_online_next_latent", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOCDAEOnlineNextLatent())
task_registry.register("cyber2_push_door_ecdae", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOECDAE())
task_registry.register("cyber2_push_door_emlp_cdae", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOEMLPCDAE())
task_registry.register("cyber2_push_door_emlp_ecdae", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOEMLPECDAE())
task_registry.register("cyber2_push_door_emlp_ecdae_online", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOEMLPECDAEOnline())
task_registry.register("cyber2_push_door_emlp_ecdae_online_next_latent", CyberPushDoorEnv, CyberPushDoorConfig(), CyberPushDoorCfgPPOEMLPECDAEOnlineNextLatent())

task_registry.register("cyber2_walk_slope", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPO())
task_registry.register("cyber2_walk_slope_aug", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOAug())
task_registry.register("cyber2_walk_slope_emlp", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOEMLP())
task_registry.register("cyber2_walk_slope_cdae", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOCDAE())
task_registry.register("cyber2_walk_slope_cdae_online", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOCDAEOnline())
task_registry.register("cyber2_walk_slope_cdae_online_next_latent", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOCDAEOnlineNextLatent())
task_registry.register("cyber2_walk_slope_ecdae", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOECDAE())
task_registry.register("cyber2_walk_slope_emlp_cdae", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOEMLPCDAE())
task_registry.register("cyber2_walk_slope_emlp_ecdae", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOEMLPECDAE())
task_registry.register("cyber2_walk_slope_emlp_ecdae_online", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOEMLPECDAEOnline())
task_registry.register("cyber2_walk_slope_emlp_ecdae_online_next_latent", CyberWalkSlopeEnv, CyberWalkSlopeConfig(), CyberWalkSlopeCfgPPOEMLPECDAEOnlineNextLatent())
