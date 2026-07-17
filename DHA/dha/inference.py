import os
import glob
import torch
import dha
import numpy as np
from pathlib import Path

from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.DynamicsAutoEncoder import DAE
from dha.utils.mysc import class_from_name
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
import dha.utils.isaaclab_utils as isaaclab_utils

import escnn
from escnn.nn import FieldType
import re

# model_path = "experiments/test/S:2025-05-16_16-16-41-OS:5-G:K4xC2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=711" #711" 179 481 529
# model_path = "experiments/test/S:2025-05-16_16-16-41-OS:5-G:K4xC2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=514"

# model_path = "experiments/test/S:from_weishu-OS:3-G:C2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=962"
model_path = "experiments/test/S:from_weishu-OS:3-G:C2-H:5-EH:5_EC-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=881"

if "mini_cheetah" in model_path:
    terrains = ["curriculum"] #, "uneven_easy", "uneven_medium", "uneven_hard_squares"]
    modes = ["2025-05-16_16-16-41"]
    # modes = ["2025-05-16_16-22-18"]
    for terrain in terrains:
        for mode in modes:
            data_paths = list(Path(f"data/mini_cheetah/isaaclab_recordings/{terrain}/{mode}/raw_recording").glob("*.npy"))
else: # a1
    tasks = ["stand_dance_cyber"] #, "uneven_easy", "uneven_medium", "uneven_hard_squares"]
    # modes = ["2025-05-16_16-16-41"]
    # modes = "20250521_203857"
    modes = ["from_weishu"]
    for task in tasks:
        for mode in modes:
            data_paths = list(Path(f"data/a1/isaacgym_recordings/{task}/{mode}/raw_recording").glob("*.npy"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dha_dir = os.path.dirname(dha.__file__)
model_dir = os.path.join(dha_dir, model_path)

model = isaaclab_utils.get_trained_dae_model(model_dir).to(device)
model.eval()  # Set the model to evaluation mode

 # Get the normalization info for the DAE model
norm_dir = os.path.join(model_dir, "state_mean_var.npy")
# Load state_mean and state_var from the npy file
norm_data = np.load(norm_dir, allow_pickle=True).item()

# Extract state_mean and state_var values
state_mean, state_std, action_mean, action_std = isaaclab_utils.get_stats(model_path, device)

# Get some data to test the model
all_data = []
all_action_data = []
if "mini_cheetah" in model_path:
    action_label = 'actions'
    concat_axis = 1
else: # a1
    action_label = 'action'
    concat_axis = 0
for data_path in data_paths:
    assert data_path.exists(), f"Path {data_path.absolute()} does not exist"
    data = np.load(data_path, allow_pickle=True)
    all_data.append(np.array([traj['obs'] for traj in data]))
    all_action_data.append(np.array([traj[action_label] for traj in data]))
state_batched = np.concatenate(all_data, axis=concat_axis) # shape is (ep_length, num_envs, obs_dim)
action_batched = np.concatenate(all_action_data, axis=concat_axis)
# Reshape the data so that the first dimension is end to end
# obs = state_batched.transpose((1,0,2)).reshape(state_batched.shape[0] * state_batched.shape[1], -1)
# joint_angle_action = action_batched.transpose((1,0,2)).reshape(action_batched.shape[0] * action_batched.shape[1], -1) # this is the joint angle action
# Convert to torch tensors

if "a1" in model_path:
    num_timesteps = 1000
    num_envs = 32
    state_reshaped = state_batched.reshape(len(data_paths), num_timesteps, num_envs, state_batched.shape[-1])
    reordered_state = state_reshaped.transpose(0, 2, 1, 3)
    state_batched = reordered_state.reshape(num_timesteps, len(data_paths) * num_envs, state_batched.shape[-1])

    action_reshaped = action_batched.reshape(len(data_paths), num_timesteps, num_envs, action_batched.shape[-1])
    reordered_action = action_reshaped.transpose(0, 2, 1, 3)
    action_batched = reordered_action.reshape(num_timesteps, len(data_paths) * num_envs, action_batched.shape[-1])

obs = torch.tensor(state_batched, device=device).float()
joint_angle_action = torch.tensor(action_batched, device=device).float() # joint angle action

if "mini_cheetah" in model_path:
    # Get the zero reference position for mini cheetah
    q0 = isaaclab_utils.get_pybullet_q0(device)
    joint_order_indices = isaaclab_utils.get_joint_order_indices()
else:
    q0 = None
    joint_order_indices = None

# Extract the state_obs and action_obs (action is velocity commands)
# latent_state, state, action = utils.get_latent_state(obs, model_path, model, joint_order_indices, q0, state_mean, state_std, action_mean, action_std)

# Parse the prediction horizon from the string
prediction_horizon = int(re.search(r"H:(\d+)", model_path).group(1))

# Get the state, action, and next_state tensors
if "mini_cheetah" in model_path:
    robot_name = "mini_cheetah"
else: # a1
    robot_name = "a1"

state_batched, action_batched = isaaclab_utils.get_state_action_from_obs_batched(robot_name, obs, joint_angle_action, joint_order_indices, q0)

state, action, next_state = isaaclab_utils.reshape_state_action(state_batched, action_batched, prediction_horizon)

# Choose an env to look at
env_idx = 100

state = state[:, env_idx].squeeze(0)
action = action[:, :, env_idx].squeeze(0)
next_state = next_state[:, :, env_idx].squeeze(0)

# Normalize before passing to the model
if "C-DAE" in model_path:
    state_normed, action_normed, next_state_normed = isaaclab_utils.normalize(state_mean, state_std, action_mean, action_std, state, action, next_state)
else:
   state_normed, next_state_normed = isaaclab_utils.normalize_state(state_mean, state_std, state, next_state)

# Predict the next state using the DAE model
with torch.no_grad():
    if "C-DAE" in model_path:
        pred_dict = model(state_normed, action_normed, next_state_normed)
    else:
        pred_dict = model(state_normed, next_state_normed)

# Compare the predicted state with the actual state
predicted_state_normed = pred_dict['pred_state_traj'] # shape (batch, pred_horizon + 1, state_dim)
pred_next_state_normed = predicted_state_normed[:, 1:, :]

pred_next_state = isaaclab_utils.denormalize(state_mean, state_std, pred_next_state_normed)

# Compute the RMSE for this env idx
input(F"next state shape: {next_state.shape}, pred_next_state shape: {pred_next_state.shape}")
state_pred_rmse = torch.sqrt(torch.mean((next_state - pred_next_state) ** 2)) #, dim=(0, 1)))
input(state_pred_rmse)

# Compute the difference between predicted and actual latent state
#TODO this is not a denormalizable quantity, so i should not denormalize it
pred_latent_state = pred_dict['pred_obs_state_traj'][:, 1:, :]  # shape (batch, pred_horizon + 1, latent_dim)
if "E-DAE" in model_path or "EC-DAE" in model_path:
    next_latent_state = []
    for i in range(next_state_normed.shape[1]):
        next_latent_state_i = model.obs_fn(model.state_type(next_state_normed[:, i, :]))  # shape (batch, latent_dim)
        next_latent_state.append(next_latent_state_i.tensor)
    next_latent_state = torch.stack(next_latent_state, dim=1)  # shape (batch, pred_horizon + 1, latent_dim)
else:
    next_latent_state = model.obs_fn(next_state_normed)  # shape (batch, latent_dim)
input(f"pred_latent_state shape: {pred_latent_state.shape}, next_latent_state shape: {next_latent_state.shape}")
obs_state_pred_rmse = torch.sqrt(torch.mean((next_latent_state - pred_latent_state) ** 2))  # shape (batch, latent_dim)
input(obs_state_pred_rmse)

# Count the unstable eigvals of A
if "E-DAE" in model_path or "EC-DAE" in model_path:
    linear_layer = model.obs_space_dynamics.transfer_op
    eigenvalues_results = {}
    assembled_matrix = linear_layer._basisexpansion(linear_layer.weights)
    if assembled_matrix.shape[0] == assembled_matrix.shape[1]:
        try:
            # Calculate eigenvalues for the full assembled matrix
            eigvals = torch.linalg.eigvals(assembled_matrix.squeeze()).detach().cpu().numpy()
            eigenvalues_results["full_transfer_matrix"] = eigvals # Store with a descriptive key
        except Exception:
            pass
else:
    A = model.obs_space_dynamics.transfer_op.weight.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(A)
unstable_eigvals = np.sum(np.abs(eigvals) > 1)
input(f"Number of unstable eigenvalues for env {env_idx}: {unstable_eigvals}")

# plot the states together
import matplotlib.pyplot as plt

# Define the state observation names and their dimensions
if "mini_cheetah" in model_path:
    state_obs_names = [
        ("joint_pos_S1", 24),
        ("joint_vel", 12),
        ("base_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("a_joint_pos_S1", 24),
        ("velocity_commands_xy", 2),
        ("velocity_commands_z", 1)
    ]
else: # a1
    state_obs_names = [
        ("projected_gravity", 3),
        ("projected_forward_vec", 3),
        ("xy_commands", 2),
        ("z_commands", 1),
        ("joint_pos", 12),
        ("joint_vel", 12),
        ("prev_actions", 12),
        ("clock_inputs", 2)
    ]

# Start index for slicing
start_idx = 0

# Create a separate plot for each state observation
colors = plt.cm.tab10.colors  # Use a colormap for consistent colors
color_idx = 0  # Initialize color index

steps_ahead_to_plot = -1

for name, dim in state_obs_names:
    end_idx = start_idx + dim
    plt.figure(figsize=(12, 6))
    for i in range(dim):
        color = colors[i % len(colors)]  # Cycle through colors for each component
        plt.plot(
            next_state[:, steps_ahead_to_plot, start_idx + i].cpu().numpy(),
            label=f'Actual {name}[{i}]',
            alpha=0.5,
            color=color
        )
        plt.plot(
            pred_next_state[:, steps_ahead_to_plot, start_idx + i].cpu().numpy(),
            label=f'Predicted {name}[{i}]',
            linestyle='--',
            color=color,
            alpha=0.5
        )
    plt.title(f'Actual vs Predicted State: {name}')
    plt.xlabel('Time Step')
    plt.ylabel(f'{name} Value')
    plt.legend()
    plt.show()
    start_idx = end_idx