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

import pinocchio as pin # fix for incompatible pinocchio bindings with isaacgym. need to import before isaacgym
import argparse
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger
import os
import numpy as np
import torch
import shutil
import time

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.task == "go1_highlevel":
        low_env_cfg = env_cfg.low_env
    else:
        low_env_cfg = env_cfg

    # low_env_cfg.env.num_envs = min(low_env_cfg.env.num_envs, 5)
    low_env_cfg.env.num_envs = args.num_envs if args.num_envs else 32 #128
    low_env_cfg.env.episode_length_s = 40
    low_env_cfg.record.record = RECORD_FRAMES
    low_env_cfg.record.folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    low_env_cfg.terrain.curriculum = False
    low_env_cfg.rewards.curriculum = False
    low_env_cfg.mode = "test"
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.com_displacement_range = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    if "stand_dance" in args.task:
        low_env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        low_env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        low_env_cfg.commands.ranges.heading = [0.5 * np.pi, 0.5 * np.pi]
    elif "push_door" in args.task:
        if args.left:
            low_env_cfg.asset.left_or_right = 1 # 0: right, 1: left
        elif args.right:
            low_env_cfg.asset.left_or_right = 0
        if args.ood:
            low_env_cfg.init_state.randomize_rot = True
    elif "walk_slope" in args.task:
        low_env_cfg.terrain.curriculum = True
        low_env_cfg.terrain.max_init_terrain_level = 4
        low_env_cfg.commands.ranges.lin_vel_x = [0.3, 0.3]
        low_env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        low_env_cfg.commands.ranges.heading = [0., 0.]

    if os.path.exists(low_env_cfg.record.folder):
        shutil.rmtree(low_env_cfg.record.folder)
    os.makedirs(low_env_cfg.record.folder, exist_ok=True)

    # Set robustness params
    # Assuming nominal base mass is ~15kg for Cyberdog2
    added_mass = args.mass_offset
    low_env_cfg.domain_rand.friction_range = [args.friction, args.friction]
    low_env_cfg.domain_rand.added_mass_range = [added_mass, added_mass]
    low_env_cfg.domain_rand.push_robots = (args.push_vel > 0)
    low_env_cfg.domain_rand.max_push_vel_xy = args.push_vel
    low_env_cfg.asset.armature = args.armature

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, is_highlevel=(args.task == "go1_highlevel"))
    print("num_obs:", env.num_obs)
    low_env = env.low_level_env if args.task == "go1_highlevel" else env
    # obs = env.get_observations()
    obs, priv_obs = env.reset()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        if isinstance(args.checkpoint, int):
            checkpoint_name = f"model_{args.checkpoint}"
        elif isinstance(args.checkpoint, str) and args.checkpoint.endswith(".pt"):
            checkpoint_name = args.checkpoint.replace(".pt", "")
        else:
            checkpoint_name = f"model_{args.checkpoint}"
    else:
        checkpoint_name = "latest"


    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        # currently, both high and low level shares the same number of obs
        if hasattr(ppo_runner.alg.actor_critic, "adaptation_module"):
            input_dim = env.num_obs * env.num_history + env.num_obs * env.num_stacked_obs
        else:
            input_dim = env.num_obs
        export_policy_as_onnx(ppo_runner.alg.actor_critic, input_dim, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(low_env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 100 # number of steps before print average episode rewards
    
    camera_vel = np.array([1., 1., 0.])
    camera_position = np.array([10.3, 10.1, 100.5], dtype=np.float64)
    camera_target = np.array([10.0, 10.0, 100.4], dtype=np.float64)
    camera_direction = camera_target - camera_position

    img_idx = 0
    total_steps = int(1000) if "walk_slope" in args.task else int(2000)
    data = {}
    episode_reward_tmp = 0
    episode_length_tmp = 0
    episode_reward_buf = []
    episode_length_buf = []

    base_positions = []
    foot_positions = []
    joint_positions = []

    cmd_yaw_vels = []
    actual_yaw_vels = []
    cmd_heading = []
    current_headings = []

    base_x_positions = []
    base_y_positions = []

    cmd_x_vels = []
    cmd_y_vels = []
    actual_x_vels = []
    actual_y_vels = []

    obs_data = [] # to save obs-action pairs for plotting
    data_log = []

    i = 0
    while True:
        # Loop termination condition depending on the task
        if "push_door" in args.task:
            if torch.sum(env.env_finish_buffer) >= 10000:
                break
        else:
            if i >= total_steps:
                break

        with torch.no_grad():
            actions = policy(obs)

        # Retaining original logic: only log first 'total_steps' of obs_data to avoid memory bloat
        if i < total_steps:
            for j in range(obs.shape[0]):
                obs_data.append({
                    "obs": obs[j].cpu().numpy(),
                    "priv_obs": priv_obs.cpu().numpy() if hasattr(low_env, 'privileged_obs_buf') else None,
                    "action": actions[j].cpu().numpy()
                })

        obs, priv_obs, rews, dones, infos = env.step(actions.detach())
        episode_reward_tmp += rews
        episode_length_tmp += torch.ones(obs.shape[0], device=obs.device)

        # Append positions for potential saving later
        base_positions.append(env.base_pos.cpu().numpy())
        foot_positions.append(env.foot_positions.cpu().numpy())
        joint_positions.append(env.dof_pos.cpu().numpy())

        print(f"step {i}")

        cmd_yaw_vels.append(env.commands[robot_index, 2].item())
        actual_yaw_vels.append(env.base_ang_vel[robot_index, 2].item())
        cmd_heading.append(env.commands[robot_index, 3].item())
        current_headings.append(env._get_cur_heading()[robot_index].item())
        base_x_positions.append(env.base_pos[robot_index, 0].item())
        base_y_positions.append(env.base_pos[robot_index, 1].item())
        cmd_x_vels.append(env.commands[robot_index, 0].item())
        cmd_y_vels.append(env.commands[robot_index, 1].item())
        actual_x_vels.append(env.base_lin_vel[robot_index, 0].item())
        actual_y_vels.append(env.base_lin_vel[robot_index, 1].item())

        data_log.append({
            "obs": obs.cpu().numpy(),
            "desired_heading": env.commands[robot_index, 3].item(),
            "actual_heading": env._get_cur_heading()[robot_index].item()
        })

        if MOVE_CAMERA:
            camera_position += camera_vel * low_env.dt
            low_env.set_camera(camera_position, camera_position + camera_direction)
        if infos["episode"]:
            num_episodes = torch.sum(low_env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
            episode_reward_buf.extend(episode_reward_tmp[low_env.reset_buf].cpu().numpy().tolist())
            episode_reward_tmp[low_env.reset_buf] = 0.
            episode_length_buf.extend(episode_length_tmp[low_env.reset_buf].cpu().numpy().tolist())
            episode_length_tmp[low_env.reset_buf] = 0.

        i += 1

    if "push_door" in args.task:
        finish_count = env.finish_count
        success_count = env.success_count
        data["finish_count"] = finish_count
        data["success_count"] = success_count
        data["success_rate"] = success_count / finish_count if finish_count > 0 else 0.0
        print(f"finish count: {finish_count}, success count: {success_count}, success rate: {data['success_rate']:.3f}")

        data["avg_ttc"] = env.total_success_ttc / success_count if success_count > 0 else 0.0
        data["avg_cot"] = env.total_success_cot / success_count if success_count > 0 else 0.0
        print(f"Metrics -> TTC: {data['avg_ttc']:.2f}s | CoT: {data['avg_cot']:.3f}")

    suffix = ""
    if "push_door" in args.task:
        suffix = "_left_ood" if (args.left and args.ood) else "_left" if args.left else "_ood" if args.ood else ""
    
    # Append robustness parameters to the filename
    if args.friction != 1.0: suffix += f"_fric{args.friction}"
    if args.mass_offset != 0.0: suffix += f"_mass{args.mass_offset}"
    if args.push_vel != 0.0: suffix += f"_push{args.push_vel}"
    if args.armature != 0.005: suffix += f"_arm{args.armature}"

    # ===============================
    # ===== UNIFIED DATA SAVING =====
    # ===============================
    
    # Bundle generic arrays, trajectories, logs, and action pairs 
    data["obs_action"] = obs_data
    data["data_log"] = data_log
    data["cmd_yaw_vels"] = np.array(cmd_yaw_vels)
    data["actual_yaw_vels"] = np.array(actual_yaw_vels)
    data["cmd_heading"] = np.array(cmd_heading)
    data["actual_heading"] = np.array(current_headings)
    data["cmd_vx"] = np.array(cmd_x_vels)
    data["cmd_vy"] = np.array(cmd_y_vels)
    data["actual_vx"] = np.array(actual_x_vels)
    data["actual_vy"] = np.array(actual_y_vels)
    data["base_x_positions"] = np.array(base_x_positions)
    data["base_y_positions"] = np.array(base_y_positions)
    
    # Conditionally add time-series buffers
    if args.save_timeseries:
        data["base_positions"] = np.array(base_positions)
        data["foot_positions"] = np.array(foot_positions)
        data["joint_positions"] = np.array(joint_positions)
        print("[✓] Included time-series data in the unified output")

    # Finalize directory and save
    run = train_cfg.runner.load_run
    export_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'isaacgym_recordings', train_cfg.runner.experiment_name, run)
    os.makedirs(export_dir, exist_ok=True)
    unified_output_file = os.path.join(export_dir, f"{checkpoint_name}_unified_data{suffix}.npy")
    np.save(unified_output_file, data)
    print(f"[✓] Saved ALL unified metrics, tracking, and logs to {unified_output_file}")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    MOVE_CAMERA = False

    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument('--ood', action='store_true', default=False, help='Whether to run the OOD version of the task')
    custom_parser.add_argument('--friction', type=float, default=1.0, help='Friction coefficient')
    custom_parser.add_argument('--mass_offset', type=float, default=0.0, help='Mass offset')
    custom_parser.add_argument('--push_vel', type=float, default=0.0, help='Push velocity')
    custom_parser.add_argument('--armature', type=float, default=0.005)
    
    # NEW ARGUMENT: Only save large positional arrays when this flag is present
    custom_parser.add_argument('--save_timeseries', action='store_true', default=False, help='Save high-frequency time series data (positions, joints, etc.) to the unified output file')
    
    # 2. Extract ONLY the custom args, leaving the rest in "remaining_argv"
    custom_args, remaining_argv = custom_parser.parse_known_args()
    
    # 3. Trick get_args() by temporarily removing our custom args from sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv
    
    # 4. Now call the standard get_args() - it will only see the default legged_gym args
    args = get_args()
    
    # 5. Inject our custom args into the main args object so play() can use them
    args.ood = custom_args.ood
    args.friction = custom_args.friction
    args.mass_offset = custom_args.mass_offset
    args.push_vel = custom_args.push_vel
    args.armature = custom_args.armature
    args.save_timeseries = custom_args.save_timeseries
    
    play(args)