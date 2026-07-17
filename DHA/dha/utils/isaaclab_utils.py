import torch

import os
import glob
import dha
import numpy as np
from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.DynamicsAutoEncoder import DAE
from dha.nn.ControlledDynamicsAutoEncoder import ControlledDAE
from dha.nn.ControlledEquivDynamicsAutoencoder import ControlledEquivDAE
from dha.utils.mysc import class_from_name
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
from dha.data.DhaDynamicsRecording import DhaDynamicsRecording
from morpho_symm.data.DynamicsRecording import DynamicsRecording
from typing import Iterable, List, Optional, Union
from dha.utils.mysc import safe_standardize



import escnn
from escnn.nn import FieldType
from morpho_symm.utils.algebra_utils import permutation_matrix
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens, Representation, Group

import re

def get_kinematic_three_rep_two(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(2, dtype=int)}
    gens = [permutation_matrix([1, 0])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three

def compute_joint_pos_obs(q_js_ms_rel, q0_isaaclab, q0, joint_order_indices):
    q_js_ms = q_js_ms_rel[:, joint_order_indices] + q0_isaaclab[joint_order_indices] + q0[7:]  # Add offset to the measurements from UMich
    cos_q_js, sin_q_js = torch.cos(q_js_ms), torch.sin(q_js_ms)  # convert from angle to unit circle parametrization
    # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
    q_js_unit_circle_t = torch.stack([cos_q_js, sin_q_js], axis=2)
    q_js_unit_circle_t = q_js_unit_circle_t.reshape(q_js_unit_circle_t.shape[0], -1)
    joint_pos_S1 = q_js_unit_circle_t  # Joints in angle not unit circle representation
    joint_pos = q_js_ms  # Joints in angle representation
    return joint_pos_S1, joint_pos

def compute_joint_pos_obs_batched(q_js_ms_rel, q0_isaaclab, q0, joint_order_indices):
    q_js_ms = q_js_ms_rel[:, :, joint_order_indices] + q0_isaaclab[joint_order_indices].unsqueeze(0).unsqueeze(0) + q0[7:].unsqueeze(0).unsqueeze(0)  # Add offset to the measurements from UMich
    cos_q_js, sin_q_js = torch.cos(q_js_ms), torch.sin(q_js_ms)  # convert from angle to unit circle parametrization
    # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
    q_js_unit_circle_t = torch.stack([cos_q_js, sin_q_js], axis=3)
    q_js_unit_circle_t = q_js_unit_circle_t.reshape(q_js_unit_circle_t.shape[0], q_js_unit_circle_t.shape[1], -1)
    joint_pos_S1 = q_js_unit_circle_t  # Joints in angle not unit circle representation
    joint_pos = q_js_ms  # Joints in angle representation
    return joint_pos_S1, joint_pos

def get_state_action_from_obs(obs, joint_order_indices, q0):
    """
    Takes an observation from the observation class and extracts the system state and action vectors.

    Puts the state in the correct form for use in the DAE model.

    State vector is composed as: $x = [q, \dot q, z, v, o, \omega] \in \mathbb R^{46}$
    """

    # Define the default joint positions in Isaaclab
    q0_isaaclab = torch.tensor([0.10000000149011612, -0.10000000149011612, 0.10000000149011612, -0.10000000149011612, -0.800000011920929, -0.800000011920929, -0.800000011920929, -0.800000011920929, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716], device=obs.device, dtype=obs.dtype) #TODO this is hardcoded, if the defaults change then I have to change this too

    base_vel = obs[:, :3]
    base_ang_vel = obs[:, 3:6]
    projected_gravity = obs[:, 6:9]
    velocity_commands_xy = obs[:, 9:11] # Rep: Rd for xy, euler xyz for heading? idk
    velocity_commands_z = obs[:, 11].unsqueeze(-1) # Rep: Rd for xy, euler xyz for heading? idk
    joint_pos_rel = obs[:, 12:24]
    joint_vel = obs[:, 24:36]
    action_joint_pos = obs[:, 36:48]

    # Get the joint positions and velocities
    joint_pos_S1, _ = compute_joint_pos_obs(joint_pos_rel, q0_isaaclab, q0, joint_order_indices)

    # Reorder joint velocities
    joint_vel = joint_vel[:, joint_order_indices]

    # Parametrize past action joint positions
    a_joint_pos_S1, _ = compute_joint_pos_obs(action_joint_pos, q0_isaaclab, q0, joint_order_indices)

    state_obs = [joint_pos_S1, joint_vel, base_vel, base_ang_vel, projected_gravity, a_joint_pos_S1]
    action_obs = [velocity_commands_xy, velocity_commands_z]

    x = torch.cat(state_obs, dim=1).to(dtype=obs.dtype)
    u = torch.cat(action_obs, dim=1).to(dtype=obs.dtype)

    return x, u

def get_state_action_from_obs_batched(robot_name, obs, action, joint_order_indices, q0):
    """
    Takes an observation from the observation class and extracts the system state and action vectors.

    Puts the state in the correct form for use in the DAE model.

    State vector is composed as: $x = [q, \dot q, v, \omega, a] \in \mathbb R^{46}$
    """

    if robot_name == "mini_cheetah":
        # Define the default joint positions in Isaaclab
        q0_isaaclab = torch.tensor([0.10000000149011612, -0.10000000149011612, 0.10000000149011612, -0.10000000149011612, -0.800000011920929, -0.800000011920929, -0.800000011920929, -0.800000011920929, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716], device=obs.device, dtype=obs.dtype) #TODO this is hardcoded, if the defaults change then I have to change this too

        # Assume obs is ['joint_pos_S1', 'joint_vel', 'base_vel', 'base_ang_vel', 'projected_gravity', 'a_joint_pos_S1', 'velocity_commands_xy', 'velocity_commands_z']
        base_vel = obs[:, :, :3]
        base_ang_vel = obs[:, :, 3:6]
        projected_gravity = obs[:, :, 6:9]
        velocity_commands_xy = obs[:, :, 9:11] # Rep: Rd for xy, euler xyz for heading? idk
        velocity_commands_z = obs[:, :, 11].unsqueeze(-1) # Rep: Rd for xy, euler xyz for heading? idk
        joint_pos_rel = obs[:, :, 12:24]
        joint_vel = obs[:, :, 24:36]
        action_joint_pos = obs[:, :, 36:48]

        # Assume action is ['current_action_S1']
        current_action_S1, _ = compute_joint_pos_obs_batched(action, q0_isaaclab, q0, joint_order_indices)

        # Get the joint positions and velocities
        joint_pos_S1, _ = compute_joint_pos_obs_batched(joint_pos_rel, q0_isaaclab, q0, joint_order_indices)

        # Reorder joint velocities
        joint_vel = joint_vel[:, :, joint_order_indices]

        # Parametrize past action joint positions
        a_joint_pos_S1, _ = compute_joint_pos_obs_batched(action_joint_pos, q0_isaaclab, q0, joint_order_indices)

        state_obs = [joint_pos_S1, joint_vel, base_vel, base_ang_vel, projected_gravity, a_joint_pos_S1, velocity_commands_xy, velocity_commands_z]
        action_obs = [current_action_S1]

    else: # a1
        joint_order = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': -1.396,  # -80 deg
            'FR_thigh_joint': -1.396,
            'RL_thigh_joint': -1.396,
            'RR_thigh_joint': -1.396,
            'FL_calf_joint': 2.356,    # 135 deg
            'FR_calf_joint': 2.356,
            'RL_calf_joint': 2.356,
            'RR_calf_joint': 2.356,
        }
        default_dof_pos = torch.tensor([default_joint_angles[j] for j in joint_order], device=obs.device, dtype=obs.dtype)  # (12,)

        joint_pos_rel = obs[:, :, 9:21]  # 12D
        joint_pos = joint_pos_rel + default_dof_pos.unsqueeze(0)  # Add the default joint angles to the relative positions

        # joint_vel, actions
        joint_vel = obs[:, :, 21:33]  # 12D
        prev_actions = obs[:, :, 33:45]    # 12D

        # 其他观测量（含 projected_gravity, projected_forward_vec, command, etc.）
        projected_gravity = obs[:, :, 0:3]
        projected_forward_vec = obs[:, :, 3:6]
        xy_commands = obs[:, :, 6:8]
        z_commands = obs[:, :, 8:9]  # 1D, euler_z
        clock_inputs = obs[:, :, 45:47]

        state_obs = [projected_gravity, projected_forward_vec, xy_commands, z_commands, joint_pos, joint_vel, prev_actions, clock_inputs]
        action_obs = [action]

    x = torch.cat(state_obs, dim=2).to(dtype=obs.dtype)
    u = torch.cat(action_obs, dim=2).to(dtype=obs.dtype)

    return x, u

def quat_to_euler_torch(quaternions):
    """
    Converts quaternions to Euler angles (XYZ convention) using PyTorch.

    Args:
        quaternions (torch.Tensor): Quaternions tensor of shape (..., 4).

    Returns:
        torch.Tensor: Euler angles tensor of shape (..., 3).
    """

    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.pi / 2, torch.asin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)

def extract_trained_model_info(state_dict, model_dir) -> (int, int, bool, int):
    """Extracts model information from a state_dict."""
    layers = 0
    hidden_units = 0
    obs_state_dim = 0
    has_bias = False

    for key in state_dict.keys():
        if ".obs_fn.net" in key:
            if "model.obs_fn.net.block_" in key and "weight" in key:
                layers += 1
            if "E-DAE" in model_dir or "EC-DAE" in model_dir:
                if "model.obs_fn.net.block_0.linear_0" in key and "matrix" in key:
                    state_dim = state_dict[key].shape[1]
            else:
                if "model.obs_fn.net.block_0" in key and "weight" in key:
                    state_dim = state_dict[key].shape[1]
            if "linear_0" in key and ("weight" in key or "matrix" in key):
                hidden_units = state_dict[key].shape[0]
            if 'bias' in key and not has_bias:
                has_bias = True
            if "head" in key and ("weight" in key or "matrix" in key):
                obs_state_dim = state_dict[key].shape[0]

    layers += 1  # Add one for the head layer

    return layers, hidden_units, has_bias, obs_state_dim, state_dim

def remove_state_dict_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def get_trained_dae_model(model_dir):
    """
    Load the trained DAE model.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        torch.nn.Module: The trained model.
    """
    ckpt_path = os.path.join(model_dir, "best.ckpt")

    # Load the model from the checkpoint
    checkpoint = torch.load(ckpt_path)

    # Extract the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Define the state representation
    # G is the symmetry group of the system
    if "mini_cheetah" in model_dir:
        robot, G = load_symmetric_system(robot_name="mini_cheetah")

        # Create the state representations
        gspace = escnn.gspaces.no_base_space(G)
        # Extract the representations from G.representations.items()
        rep_Q_js = G.representations['Q_js']
        rep_Rd = G.representations['R3']
        rep_TqQ_js = G.representations['TqQ_js']
        rep_z = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
        rep_z.name = "base_z"
        rep_xy = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[:2, :2].reshape((2, 2)) for h in G.elements if h != G.identity})
        rep_xy.name = "base_xy"
        rep_euler_xyz = G.representations['euler_xyz']
        rep_euler_z = group_rep_from_gens(G, rep_H={h: rep_euler_xyz(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
        rep_euler_z.name = "euler_z"

        # Define the state and action type using the extracted representations
        state_reps = [rep_Q_js, rep_TqQ_js, rep_Rd, rep_euler_xyz, rep_Rd, rep_Q_js, rep_xy, rep_euler_z] #['joint_pos_S1', 'joint_vel', 'base_vel', 'base_ang_vel', 'projected_gravity', 'a_joint_pos_S1', 'velocity_commands_xy', 'velocity_commands_z']
        state_type = FieldType(gspace, representations=state_reps)
        state_type.size = sum(rep.size for rep in state_reps) + rep_Q_js.size + rep_Rd.size  # Count duplicates twice
        state_type = FieldType(gspace, representations=state_reps)
        action_reps = [rep_Q_js]  # ['current_actions_S1']
        action_type = FieldType(gspace, representations=action_reps)
        action_type.size = sum(rep.size for rep in action_reps)

    else: # a1
        robot, G = load_symmetric_system(robot_name="a1")

        # Create the state representations
        gspace = escnn.gspaces.no_base_space(G)
        # Extract the representations from G.representations.items()
        rep_Rd = G.representations['R3']
        rep_TqQ_js = G.representations['TqQ_js']
        rep_kin_three = get_kinematic_three_rep_two(G)
        rep_xy = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[:2, :2].reshape((2, 2)) for h in G.elements if h != G.identity})
        rep_xy.name = "base_xy"
        rep_euler_xyz = G.representations['euler_xyz']
        rep_euler_z = group_rep_from_gens(G, rep_H={h: rep_euler_xyz(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
        rep_euler_z.name = "euler_z"

        # Define the state and action type using the extracted representations
        state_reps = [rep_Rd, rep_Rd, rep_xy, rep_euler_z, rep_TqQ_js, rep_TqQ_js, rep_TqQ_js, rep_kin_three] #['projected_gravity', 'projected_forward_vec', 'xy_commands', 'z_commands', 'joint_pos', 'joint_vel', 'prev_actions', 'clock_inputs'] # base pose
        state_type = FieldType(gspace, representations=state_reps)
        state_type.size = sum(rep.size for rep in state_reps) + rep_Rd.size + 2 * rep_TqQ_js.size  # Count duplicates twice
        state_type = FieldType(gspace, representations=state_reps)
        action_reps = [rep_TqQ_js]  # ['actions']
        action_type = FieldType(gspace, representations=action_reps)
        action_type.size = sum(rep.size for rep in action_reps)

    num_layers, num_hidden_units, bias, obs_state_dim, state_dim = extract_trained_model_info(state_dict, model_dir)

    dt = 0.02
    orth_w_match = re.search(r"Orth_w:([\d\.]+)", model_dir)
    orth_w = float(orth_w_match.group(1)) if orth_w_match else 0.0
    obs_pred_w_match = re.search(r"Obs_w:([\d\.]+)", model_dir)
    obs_pred_w = float(obs_pred_w_match.group(1)) if obs_pred_w_match else 1.0
    group_avg_trick = True
    state_dependent_obs_dyn = False
    enforce_constant_fn = True
    act_match = re.search(r"Act:([\d\.]+)", model_dir)
    activation = obs_pred_w_match.group(1) if act_match else 'ELU'
    batch_norm = False

    if not "E-DAE" in model_dir and not "EC-DAE" in model_dir:
        activation = class_from_name("torch.nn", activation)

    obs_fn_params = {'num_layers': num_layers, 'num_hidden_units': num_hidden_units, 'activation': activation, 'bias': bias, 'batch_norm': batch_norm}

    initial_rng_state = torch.get_rng_state()

    if "E-DAE" in model_dir:
        model = EquivDAE(
            state_rep=state_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            enforce_constant_fn=enforce_constant_fn,
        )
    elif "EC-DAE" in model_dir:
        model = ControlledEquivDAE(
            state_rep=state_type.representation,
            action_rep=action_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            enforce_constant_fn=enforce_constant_fn,
        )
    elif "C-DAE" in model_dir:
        model = ControlledDAE(
            state_dim=state_type.size,
            action_dim=action_type.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=enforce_constant_fn,
        )
    else:
        corr_w = 0.0
        model = DAE(
            state_dim=state_type.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            corr_w=corr_w,
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=enforce_constant_fn,
        )

    torch.set_rng_state(initial_rng_state)
    model.load_state_dict(remove_state_dict_prefix(state_dict, "model."))

    return model

def get_stats(model_path, device):

        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, model_path)
        norm_dir = os.path.join(model_dir, "state_mean_var.npy")
        # Load state_mean and state_var from the npy file
        norm_data = np.load(norm_dir, allow_pickle=True).item()

        # Extract state_mean and state_var values
        state_mean_values = norm_data["state_mean"]
        state_var_values = norm_data["state_var"]
        state_mean = torch.tensor(state_mean_values, device=device).float()
        state_std = torch.sqrt(torch.tensor(state_var_values, device=device)).float()

        # Extract action_mean and action_var values
        if "C-DAE" in model_path:
            # C-DAE model
            action_mean_values = norm_data["action_mean"]
            action_var_values = norm_data["action_var"]
            action_mean = torch.tensor(action_mean_values, device=device).float()
            action_std = torch.sqrt(torch.tensor(action_var_values, device=device)).float()
        else:
            action_mean, action_std = None, None

        return state_mean, state_std, action_mean, action_std

def get_pybullet_q0(device):
    import pybullet
    from pybullet_utils.bullet_client import BulletClient
    robot, G = load_symmetric_system(robot_name="mini_cheetah")
    bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
    robot.configure_bullet_simulation(bullet_client=bullet_client)
    # Get zero reference position.
    q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
    q0 = torch.tensor(q0).to(device)
    return q0

def get_joint_order_indices():
    usd_joint_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    joint_order_for_morphosymm = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

    joint_order_indices = [usd_joint_order.index(joint) for joint in joint_order_for_morphosymm]
    return joint_order_indices

def get_latent_state(observations, model_path, dae_model, joint_order_indices, q0, data_state_mean, data_state_std, data_action_mean, data_action_std):
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, u = get_state_action_from_obs(observations, joint_order_indices, q0)
            x_normed = (x - data_state_mean) / data_state_std
            u_normed = (u - data_action_mean) / data_action_std
            if "E-DAE" in model_path:
                # E-DAE model
                symmetric_x = dae_model.state_type(x_normed)
                s = dae_model.obs_fn(symmetric_x).tensor
            else:
                # DAE model
                s = dae_model.obs_fn(x_normed)
        return s, x_normed, u_normed

def normalize_state(state_mean, state_std, state, next_state):
    """
    Normalize the state tensors using the provided means and standard deviations.
    """
    # state_normed = (state - state_mean) / state_std
    # next_state_normed = (next_state - state_mean) / state_std
    state_normed = safe_standardize(state_normed, state_mean, state_std)
    next_state_normed = safe_standardize(next_state_normed, state_mean, state_std)
    return state_normed, next_state_normed

def normalize(state_mean, state_std, action_mean, action_std, state, action, next_state):
    """
    Normalize the state and action tensors using the provided means and standard deviations.
    """
    # state_normed = (state - state_mean) / state_std
    # action_normed = (action - action_mean) / action_std
    # next_state_normed = (next_state - state_mean) / state_std
    state_normed = safe_standardize(state, state_mean, state_std)
    action_normed = safe_standardize(action, action_mean, action_std)
    next_state_normed = safe_standardize(next_state, state_mean, state_std)
    return state_normed, action_normed, next_state_normed

def denormalize(state_mean, state_std, state_normed):
    """
    Denormalize the state tensor using the provided means and standard deviations.
    """
    state = state_normed * state_std + state_mean
    return state

def reshape_state_action(state_batched, action_batched, prediction_horizon):
    # Clip the state tensor to match the prediction horizon
    state = state_batched[:-prediction_horizon, :, :]

    # Get the next_state tensor
    next_state = torch.stack(
        [state_batched[i + 1:i + 1 + prediction_horizon, :, :] for i in range(state_batched.shape[0] - prediction_horizon)],
        dim=0
    )

    # Get the action tensor
    action = torch.stack(
        [action_batched[i:i + prediction_horizon, :, :] for i in range(action_batched.shape[0] - prediction_horizon)],
        dim=0
    )

    return state, action, next_state

def main():
    model_dir = "/home/edelia-iit.local/git/DynamicsHarmonicsAnalysis/dha/experiments/test/S:2025-05-16_16-16-41-OS:5-G:K4xC2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=711"

    dha_dir = os.path.dirname(dha.__file__)
    model_dir = os.path.join(dha_dir, model_dir)

    model = get_trained_dae_model(model_dir)
    A = model.obs_space_dynamics.transfer_op.weight
    A_bias = model.obs_space_dynamics.transfer_op.bias
    B = model.obs_space_dynamics.control_op.weight
    B_bias = model.obs_space_dynamics.control_op.bias
    print("Model loaded successfully!")
    print(model)

    if "C-DAE" in model_dir:
        # Try to compute latent dynamics
        obs = torch.randn(1, model.obs_state_dim)  # Example observation

        # Create a mapping from current joint order to morphosymm order
        joint_order_indices = get_joint_order_indices()

        # Create a variable to hold the q0 for the joint offset that is needed by the symmetry groups
        robot, G = load_symmetric_system(robot_name="mini_cheetah")
        from pybullet_utils.bullet_client import BulletClient
        import pybullet
        bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
        robot.configure_bullet_simulation(bullet_client=bullet_client)
        # Get zero reference position.

        q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
        q0 = torch.tensor(q0)

        state, action = get_state_action_from_obs(obs, joint_order_indices, q0)
        print(f"size of state: {state.size()}, size of action: {action.size()}")

        # Compute the next state given the latent state
        my_action = torch.randn(1, 1, model.action_dim)  # Example action
        next_state, next_latent_state = model.forecast(state, my_action)
        print(f"Next state: {next_state.shape}, Next latent state: {next_latent_state.shape}")
        assert torch.allclose(next_latent_state[:, 0, :], model.obs_fn(state)), "The first element in the predicted states shoudl be the current state."

if __name__ == "__main__":
    main()