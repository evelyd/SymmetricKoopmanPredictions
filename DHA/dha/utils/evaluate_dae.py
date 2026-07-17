import os
import torch
import numpy as np

# Import your helper functions and configurations from utils.py
from utils import (
    get_trained_dae_model_from_pt, 
    load_normalization_stats, 
    safe_standardize,
    # ecdae_koopman_cfg, 
    # cdae_koopman_cfg
)

def evaluate_offline_data(model_dir: str, data_path: str, koopman_cfg: dict, task: str, dae_type: str, left_data_path: str = None, horizon: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Model
    # koopman_cfg = ecdae_koopman_cfg if dae_type.lower() == "ecdae" else cdae_koopman_cfg
    model_pt_path = os.path.join(model_dir, "dae_model_20000.pt") # Adjust filename if necessary
    
    print(f"Loading {dae_type} model from {model_pt_path}...")
    model = get_trained_dae_model_from_pt(model_pt_path, koopman_cfg, task=dae_type.lower(), dt=0.02, device=device)
    model.eval()

    # 3. Load and Format the Offline Data
    print(f"Loading rollout data from {data_path}...")
    loaded_data = np.load(data_path, allow_pickle=True)
    loaded_data_left = np.load(left_data_path, allow_pickle=True) if left_data_path is not None else None
    
    # Handle numpy's wrapping of lists into 0-d object arrays
    if isinstance(loaded_data, np.ndarray) and loaded_data.ndim == 1 and isinstance(loaded_data[0], dict):
        loaded_data = loaded_data.tolist()
        left_data = loaded_data_left.tolist() if loaded_data_left is not None else None
    elif loaded_data.shape == ():
        loaded_data = loaded_data.item()
        left_data = loaded_data_left.item() if loaded_data_left is not None else None

    if isinstance(loaded_data, list):
        # Extract the flattened lists of arrays
        obs_array = np.array([d["obs"] for d in loaded_data])
        action_array = np.array([d["action"] for d in loaded_data])
        left_obs_array = np.array([d["obs"] for d in left_data]) if left_data is not None else None
        left_action_array = np.array([d["action"] for d in left_data]) if left_data is not None else None
        # Check if the data was flattened by the 'for j in range(obs.shape[0])' loop
        if obs_array.ndim == 2:
            # Shape is currently (total_steps * num_envs, dim)
            # IMPORTANT: Set this to match the --num_envs you used during play_and_save.py (default is 32)
            num_envs = 32 
            total_steps = obs_array.shape[0] // num_envs
            
            # Reshape back to (total_steps, num_envs, dim)
            obs_array = obs_array.reshape(total_steps, num_envs, -1)
            action_array = action_array.reshape(total_steps, num_envs, -1)
            if left_obs_array is not None:
                left_obs_array = left_obs_array.reshape(total_steps, num_envs, -1)
            if left_action_array is not None:
                left_action_array = left_action_array.reshape(total_steps, num_envs, -1)
            # Transpose to (num_envs, total_steps, dim) to group continuous trajectories together
            obs_array = obs_array.transpose(1, 0, 2)
            action_array = action_array.transpose(1, 0, 2)
            if left_obs_array is not None:
                left_obs_array = left_obs_array.transpose(1, 0, 2)
            if left_action_array is not None:
                left_action_array = left_action_array.transpose(1, 0, 2)
            
        elif obs_array.ndim == 3:
            # If you saved it as (total_steps, num_envs, dim) directly without the inner loop
            obs_array = obs_array.transpose(1, 0, 2)
            action_array = action_array.transpose(1, 0, 2)
            if left_obs_array is not None:
                left_obs_array = left_obs_array.transpose(1, 0, 2)
            if left_action_array is not None:
                left_action_array = left_action_array.transpose(1, 0, 2)
            
        raw_obs = torch.tensor(obs_array, dtype=torch.float32, device=device)
        raw_actions = torch.tensor(action_array, dtype=torch.float32, device=device)
        raw_left_obs = torch.tensor(left_obs_array, dtype=torch.float32, device=device) if left_obs_array is not None else None
        raw_left_actions = torch.tensor(left_action_array, dtype=torch.float32, device=device) if left_action_array is not None else None
        
    else:
        # Fallback if it was saved as a standard dictionary of arrays
        raw_obs = torch.tensor(loaded_data["obs"], dtype=torch.float32, device=device)
        raw_actions = torch.tensor(loaded_data["action"], dtype=torch.float32, device=device)
        raw_left_obs = torch.tensor(loaded_data_left["obs"], dtype=torch.float32, device=device) if loaded_data_left is not None else None
        raw_left_actions = torch.tensor(loaded_data_left["action"], dtype=torch.float32, device=device) if loaded_data_left is not None else None
        if raw_obs.shape[0] > raw_obs.shape[1] and len(raw_obs.shape) == 3:
            raw_obs = raw_obs.transpose(0, 1) 
            raw_actions = raw_actions.transpose(0, 1)
            if raw_left_obs is not None:
                raw_left_obs = raw_left_obs.transpose(0, 1)
            if raw_left_actions is not None:
                raw_left_actions = raw_left_actions.transpose(0, 1)

    num_envs, num_steps, state_dim = raw_obs.shape
    _, _, action_dim = raw_actions.shape

    # 1. Dynamically get the required state dimension from the loaded Koopman model
    if hasattr(model, 'obs_state_type'):
        dae_state_dim = model.state_type.size  # For EquivDAE
    else:
        dae_state_dim = model.state_dim        # For standard DAE

    full_obs_dim = raw_obs.shape[2]
        
    print(f"Full RL obs dim: {full_obs_dim} | DAE expects state dim: {dae_state_dim}")

    # 2. Slice the raw observation to keep only the current state frame.
    # Note: If your Isaac Gym appends the current frame at the end of the history buffer instead 
    # of the beginning, change this to raw_obs = raw_obs[:, :, -dae_state_dim:]
    raw_obs = raw_obs[:, :, :dae_state_dim] 
    if raw_left_obs is not None:
        raw_left_obs = raw_left_obs[:, :, :dae_state_dim]

    # 2. Load Normalization Stats
    # # Calculate stats directly from your true rollout data
    # state_mean = raw_obs[:, :, :dae_state_dim].mean(dim=(0, 1))
    # state_std = raw_obs[:, :, :dae_state_dim].std(dim=(0, 1))
    # input(f"Calculated normalization stats from data. State mean: {state_mean.cpu().numpy()}, State std: {state_std.cpu().numpy()}. Press Enter to continue...")
    
    # # Prevent division by zero
    # state_std[state_std < 1e-5] = 1.0
    # #TODO get the normalization stats from the model itself instead of loading from a separate file, to avoid mismatch bugs. We can save them as part of the model.pt file during training.


    # 3. Handle the hardcoded '35' fallback bug from utils.py
    # 1. Slice to get the correct DAE input dimensions
    dae_raw_obs = raw_obs[:, :, :dae_state_dim]
    
    # 2. Calculate the uncentered variance (mean of squares) exactly as done in training
    # Variance approx = Sum of Squares / Count
    state_variance_approx = torch.mean(dae_raw_obs**2, dim=(0, 1))
    
    # 3. Add epsilon and take square root
    epsilon = 1e-8
    state_scale = torch.sqrt(state_variance_approx + epsilon)
    input(f"state std: {state_scale.cpu().numpy()}. Press Enter to continue...")
    
    # 4. Handle constant values to prevent division by zero
    state_scale[state_scale == 0] = 1.0
    
    # 5. Normalize WITHOUT subtracting the mean
    norm_obs = dae_raw_obs / state_scale

    # 4. Create Sliding Windows
    # We need windows of length horizon + 1 (1 for initial state, H for next_states)
    states_list, next_states_list, actions_list = [], [], []
    
    for t in range(num_steps - horizon):
        states_list.append(norm_obs[:, t, :])
        next_states_list.append(norm_obs[:, t+1 : t+1+horizon, :])
        actions_list.append(raw_actions[:, t : t+horizon, :])

    # Concatenate along the batch dimension
    states = torch.cat(states_list, dim=0)           # (Batch, state_dim)
    next_states = torch.cat(next_states_list, dim=0) # (Batch, horizon, state_dim)
    actions = torch.cat(actions_list, dim=0)         # (Batch, horizon, action_dim)

    print(f"Total evaluation samples (Batch size): {states.shape[0]}")

    # 5. Run Forward Pass and Compute Loss
    print("Running model forward pass...")
    with torch.no_grad():
        # The forward pass returns a dictionary with all trajectories
        outputs = model(state=states, action=actions, next_state=next_states)
        
        # compute_loss_and_metrics requires these specific inputs from the output dict
        # We pass a dummy tensor for pred_obs_state_one_step as it's handled via kwargs or ignored if not strict
        loss, metrics = model.compute_loss_and_metrics(
            state=states,
            action=actions,
            next_state=next_states,
            pred_state_traj=outputs["pred_state_traj"],
            rec_state_traj=outputs["rec_state_traj"],
            obs_state_traj=outputs["obs_state_traj"],
            pred_obs_state_traj=outputs["pred_obs_state_traj"],
            pred_obs_state_one_step=outputs.get("pred_obs_state_one_step", None) 
        )

   # 6. Print Results
    print("\n" + "="*80)
    print("Offline Evaluation Metrics")
    print("="*80)
    
    # loss is usually a scalar, but let's be safe
    loss_val = loss.mean().item() if isinstance(loss, torch.Tensor) else float(loss)
    print(f"Total Loss        : {loss_val:.4f}")
    print("-" * 80)
    
    # Iterate dynamically through the metrics dictionary
    for metric_name, metric_val in metrics.items():
        if isinstance(metric_val, torch.Tensor):
            if metric_val.numel() == 1:
                # It's a single scalar value
                print(f"{metric_name:<20}: {metric_val.item():.4f}")
            else:
                # It's a multi-dimensional tensor. We want the mean per-horizon-step.
                # Usually shape is (Batch, Horizon) or just (Horizon) depending on the metric function.
                # If it has more than 1 dimension, we mean across all dims except the LAST one.
                if metric_val.dim() > 1:
                    # Average over batch/time dimensions, keep the last dimension (horizon)
                    per_step_vals = metric_val.mean(dim=tuple(range(metric_val.dim() - 1))).detach().cpu().numpy()
                else:
                    per_step_vals = metric_val.detach().cpu().numpy()
                
                mean_val = per_step_vals.mean()
                formatted_vals = "[" + ", ".join([f"{v:.4f}" for v in per_step_vals]) + "]"
                print(f"{metric_name:<20}: Mean = {mean_val:.4f} | Per-step = {formatted_vals}")
        else:
            # Fallback
            try:
                print(f"{metric_name:<20}: {float(metric_val):.4f}")
            except (ValueError, TypeError):
                print(f"{metric_name:<20}: {metric_val}")
                
    print("="*80)

    #TODO is it computing the same metric that the training loss did?
    #TODO are we comparing loss over different horizions?


if __name__ == "__main__":
    # CDAE online next latent model and data paths
    CDAE_MODEL_DIR = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/logs/push_door_cyber_cdae_online_next_latent/2026-01-24-14-56-16_"
    CDAE_DATA_PATH = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/isaacgym_recordings/push_door_cyber_cdae_online_next_latent/2026-01-24-14-56-16_/model_20000_obs_action.npy" # Assuming it's saved here
    
    # ECDAE online next latent model and data paths
    ECDAE_MODEL_DIR = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/logs/push_door_cyber_emlp_ecdae_online_next_latent/2026-02-15-19-22-09_"
    ECDAE_DATA_PATH = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/isaacgym_recordings/push_door_cyber_emlp_ecdae_online_next_latent/2026-02-15-19-22-09_/model_20000_obs_action.npy" # Assuming it's saved here
    ECDAE_LEFT_DATA_PATH = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/isaacgym_recordings/push_door_cyber_emlp_ecdae_online_next_latent/2026-02-15-19-22-09_/model_20000_obs_action_left.npy" # Assuming it's saved here
    
    cdae_model_cfg = {
        "name": 'cdae',
        "equivariant": False,
        "activation": 'ELU',
        "num_layers": 5,
        "num_hidden_units": 128,
        "batch_norm": False,
        "obs_pred_w": 1.0,
        "orth_w": 0.0,
        "corr_w": 0.0,
        "bias": True,
        "constant_function": True,
        "num_mini_batches": 8,
        "mini_batch_size": 256,
        "beta_initial": 0.4,
        "beta_annealing_steps": 20000,
    }

    ecdae_model_cfg = {
        "name": 'ecdae',
        "equivariant": True,
        "activation": 'ELU',
        "num_layers": 5,
        "num_hidden_units": 128,
        "batch_norm": False,
        "obs_pred_w": 1.0,
        "orth_w": 0.0,
        "corr_w": 0.0,
        "bias": True,
        "constant_function": True,
        "num_mini_batches": 8,
        "mini_batch_size": 256,
        "beta_initial": 0.4,
        "beta_annealing_steps": 20000,
        "group_avg_trick": True,
        "state_dependent_obs_dyn": False,
    }

    robot_cfg = {
            "name": "a1",
            "lr": 1e-3,
            "max_epochs": 200,
            "obs_state_ratio": 3,
            "state_obs": ['projected_gravity', 'projected_forward_vec', 'joint_pos', 'prev_actions', 'phase_input', 'base_pos', 'door_bottom_corner_pos', 'door_normal_vec', 'lr_vec'],
            "action_obs": ['actions'],
            "state_dim": 3 + 3 + 12 + 12 + 2 + 3 + 3 + 3 + 2,
            "action_dim": 12,
            "pred_horizon": 5,
            "frames_per_state": 1,
    }

    cdae_koopman_cfg = {
        "model": cdae_model_cfg,
        "robot": robot_cfg,
        }

    ecdae_koopman_cfg = {
        "model": ecdae_model_cfg,
        "robot": robot_cfg,
    }

    evaluate_offline_data(
        model_dir=CDAE_MODEL_DIR,
        data_path=CDAE_DATA_PATH,
        koopman_cfg=cdae_koopman_cfg,
        task="push_door",
        dae_type="cDAE",
        horizon=5
    )

    evaluate_offline_data(
        model_dir=ECDAE_MODEL_DIR,
        data_path=ECDAE_DATA_PATH,
        left_data_path=ECDAE_LEFT_DATA_PATH,
        koopman_cfg=ecdae_koopman_cfg,
        task="push_door",
        dae_type="ecDAE",
        horizon=5
    )