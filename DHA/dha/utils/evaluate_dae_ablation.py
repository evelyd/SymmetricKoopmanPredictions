import os
import torch
import numpy as np

# Import your helper functions and configurations from utils.py
from utils import (
    get_trained_dae_model_from_pt, 
    safe_standardize,
)
from dha.utils.mysc import batched_to_flat_trajectory

def evaluate_offline_data(model_dir: str, data_path: str, koopman_cfg: dict, task: str, dae_type: str, left_data_path: str = None, horizon: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"EVALUATING: {dae_type} on {task}")
    print(f"{'='*80}")
    print(f"Using device: {device}")

    # 1. Load the Model
    model_pt_path = os.path.join(model_dir, "dae_model_20000.pt")
    print(f"Loading {dae_type} model from {model_pt_path}...")
    model = get_trained_dae_model_from_pt(model_pt_path, koopman_cfg, task=dae_type.lower(), dt=0.02, device=device)
    model.eval()

    # 2. Load the Offline Data
    def load_and_reshape(path):
        if path is None: return None, None
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.ndim == 1 and isinstance(loaded[0], dict):
            loaded = loaded.tolist()
        elif loaded.shape == ():
            loaded = loaded.item()

        if isinstance(loaded, list):
            obs_array = np.array([d["obs"] for d in loaded])
            action_array = np.array([d["action"] for d in loaded])
            if obs_array.ndim == 2:
                num_envs = 32 
                total_steps = obs_array.shape[0] // num_envs
                obs_array = obs_array.reshape(total_steps, num_envs, -1).transpose(1, 0, 2)
                action_array = action_array.reshape(total_steps, num_envs, -1).transpose(1, 0, 2)
            elif obs_array.ndim == 3:
                obs_array = obs_array.transpose(1, 0, 2)
                action_array = action_array.transpose(1, 0, 2)
            return torch.tensor(obs_array, dtype=torch.float32, device=device), \
                   torch.tensor(action_array, dtype=torch.float32, device=device)
        else:
            r_obs = torch.tensor(loaded["obs"], dtype=torch.float32, device=device)
            r_act = torch.tensor(loaded["action"], dtype=torch.float32, device=device)
            if r_obs.shape[0] > r_obs.shape[1] and len(r_obs.shape) == 3:
                r_obs, r_act = r_obs.transpose(0, 1), r_act.transpose(0, 1)
            return r_obs, r_act

    raw_obs, raw_actions = load_and_reshape(data_path)
    raw_left_obs, raw_left_actions = load_and_reshape(left_data_path)

    # Determine State Dims
    dae_state_dim = model.state_type.size if hasattr(model, 'obs_state_type') else model.state_dim
    print(f"Full RL obs dim: {raw_obs.shape[2]} | DAE expects state dim: {dae_state_dim}")

    # ------------------------------------------------------------------
    # 3. Calculate Normalizer safely (Fixing the 1e-4 Explosion)
    # ------------------------------------------------------------------
    dae_raw_obs = raw_obs[:, :, :dae_state_dim]
    
    # State Normalization
    state_variance_approx = torch.mean(dae_raw_obs**2, dim=(0, 1))
    state_scale = torch.sqrt(state_variance_approx)
    state_scale[state_scale < 1e-4] = 1.0  # CRITICAL: Prevent division by tiny numbers
    
    # Action Normalization
    action_variance_approx = torch.mean(raw_actions**2, dim=(0, 1))
    action_scale = torch.sqrt(action_variance_approx)
    action_scale[action_scale < 1e-4] = 1.0
    
    # Apply normalizer
    norm_obs = dae_raw_obs / state_scale
    norm_actions = raw_actions / action_scale
    
    if raw_left_obs is not None:
        norm_left_obs = raw_left_obs[:, :, :dae_state_dim] / state_scale
        norm_left_actions = raw_left_actions / action_scale

    # ------------------------------------------------------------------
    # 4. Windowing Function (Now using normalized actions)
    # ------------------------------------------------------------------
    def create_windows(n_obs, n_act):
        num_steps = n_obs.shape[1]
        s_list, ns_list, a_list = [], [], []
        for t in range(num_steps - horizon):
            s_list.append(n_obs[:, t, :])
            ns_list.append(n_obs[:, t+1 : t+1+horizon, :])
            a_list.append(n_act[:, t : t+horizon, :]) # Normalized actions used here!
        return torch.cat(s_list, dim=0), torch.cat(ns_list, dim=0), torch.cat(a_list, dim=0)

    # Format Right Data
    states, next_states, actions = create_windows(norm_obs, norm_actions)
    gt_state_traj = torch.cat([states.unsqueeze(1), next_states], dim=1)
    
    # 5. Evaluate Baseline on Right Data
    with torch.no_grad():
        outputs_right = model(state=states, action=actions, next_state=next_states)
        baseline_right_mse = torch.nn.functional.mse_loss(outputs_right["pred_state_traj"], gt_state_traj).item()

    baseline_left_mse = None
    ablated_no_inv_mse = None
    ablated_only_inv_mse = None

    # 6. Evaluate on Left Data (and Ablation)
    if raw_left_obs is not None:
        print(f"\nProcessing Left-Side Test Data...")
        left_states, left_next_states, left_actions = create_windows(norm_left_obs, raw_left_actions)
        gt_left_state_traj = torch.cat([left_states.unsqueeze(1), left_next_states], dim=1)

        # Baseline Eval on Left Data
        with torch.no_grad():
            outputs_left = model(state=left_states, action=left_actions, next_state=left_next_states)
            baseline_left_mse = torch.nn.functional.mse_loss(outputs_left["pred_state_traj"], gt_left_state_traj).item()

        # Perform Ablations if ecDAE
        if dae_type.lower() == "ecdae":
            print("\nExecuting Eigendecomposition Ablations...")
            
            # Extract A, B, and Bias
            identity_a = torch.eye(model.obs_state_type.size, device=device)
            A = model.obs_space_dynamics.transfer_op(model.obs_state_type(identity_a)).tensor.detach().T

            identity_b = torch.eye(model.action_type.size, device=device)
            B = model.obs_space_dynamics.control_op(model.action_type(identity_b)).tensor.detach().T
            
            bias = torch.zeros(model.obs_state_type.size, device=device)
            if hasattr(model.obs_space_dynamics, 'bias') and model.obs_space_dynamics.bias is not None:
                if hasattr(model.obs_space_dynamics.bias, 'tensor'):
                    bias = model.obs_space_dynamics.bias.tensor.detach().squeeze()

            # Eigendecomposition
            A_np = A.cpu().numpy()
            eigvals, eigvecs = np.linalg.eig(A_np)
            
            # Find the invariant mode
            target_idx = np.argmin(np.abs(np.abs(eigvals) - 1.0))
            print(f"  --> Found invariant mode at λ = {eigvals[target_idx]:.4f}")
            
            # --- Ablation 1: Zero out the invariant mode ---
            eigvals_no_inv = eigvals.copy()
            eigvals_no_inv[target_idx] = 0.0
            A_no_inv_np = np.real(eigvecs @ np.diag(eigvals_no_inv) @ np.linalg.inv(eigvecs))
            A_no_inv = torch.tensor(A_no_inv_np, dtype=torch.float32, device=device)

            # --- Ablation 2: Zero out EVERYTHING EXCEPT the invariant mode ---
            eigvals_only_inv = np.zeros_like(eigvals)
            eigvals_only_inv[target_idx] = eigvals[target_idx]
            A_only_inv_np = np.real(eigvecs @ np.diag(eigvals_only_inv) @ np.linalg.inv(eigvecs))
            A_only_inv = torch.tensor(A_only_inv_np, dtype=torch.float32, device=device)

            # Helper function to rollout a trajectory given a specific A matrix
            def rollout_ablation(A_matrix):
                with torch.no_grad():
                    z_t = model.obs_fn(model.pre_process_state(state=left_states)).tensor
                    traj = [z_t]
                    
                    for t in range(horizon):
                        u_t = left_actions[:, t, :]
                        z_t = z_t @ A_matrix.T + u_t @ B.T + bias
                        traj.append(z_t)
                        
                    traj = torch.stack(traj, dim=1)
                    geom = model.obs_state_type(batched_to_flat_trajectory(traj))
                    pred_state_geom = model.inv_obs_fn(geom)
                    pred_state_traj = model.post_process_state(pred_state_geom)

                    return torch.nn.functional.mse_loss(pred_state_traj, gt_left_state_traj).item()

            # Execute both rollouts
            ablated_no_inv_mse = rollout_ablation(A_no_inv)
            ablated_only_inv_mse = rollout_ablation(A_only_inv)

    # 7. Print Final Summary Table
    print("\n" + "="*80)
    print(f"               FINAL RESULTS: {dae_type} ({horizon}-step Prediction MSE)")
    print("="*80)
    print(f"  Right-Side (Trained) Data Baseline   : {baseline_right_mse:.4f}")
    if baseline_left_mse is not None:
        print(f"  Left-Side  (Zero-Shot) Data Baseline : {baseline_left_mse:.4f}")
        
    if ablated_no_inv_mse is not None and ablated_only_inv_mse is not None:
        print(f"  Left-Side  Data ABLATED (No Inv)     : {ablated_no_inv_mse:.4f}")
        print(f"  Left-Side  Data ABLATED (Only Inv)   : {ablated_only_inv_mse:.4f}")
        
        diff_no_inv = ((ablated_no_inv_mse - baseline_left_mse) / baseline_left_mse) * 100
        diff_only_inv = ((ablated_only_inv_mse - baseline_left_mse) / baseline_left_mse) * 100
        print(f"\n  Ablation Impact (Removed Invariant)  : {diff_no_inv:+.2f}% Error Increase")
        print(f"  Ablation Impact (Isolated Invariant) : {diff_only_inv:+.2f}% Error Increase")
    print("="*80 + "\n")


if __name__ == "__main__":
    # CDAE online next latent model and data paths
    CDAE_MODEL_DIR = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/logs/push_door_cyber_cdae_online_next_latent/2026-01-24-14-56-16_"
    CDAE_DATA_PATH = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/isaacgym_recordings/push_door_cyber_cdae_online_next_latent/2026-01-24-14-56-16_/model_20000_obs_action.npy" 
    
    # ECDAE online next latent model and data paths
    ECDAE_MODEL_DIR = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/logs/push_door_cyber_emlp_ecdae_online_next_latent/2026-02-15-19-22-09_"
    ECDAE_DATA_PATH = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/isaacgym_recordings/push_door_cyber_emlp_ecdae_online_next_latent/2026-02-15-19-22-09_/model_20000_obs_action.npy"
    ECDAE_LEFT_DATA_PATH = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/isaacgym_recordings/push_door_cyber_emlp_ecdae_online_next_latent/2026-02-15-19-22-09_/model_20000_obs_action_left.npy" 
    
    cdae_model_cfg = {
        "name": 'cdae', "equivariant": False, "activation": 'ELU', "num_layers": 5, "num_hidden_units": 128,
        "batch_norm": False, "obs_pred_w": 1.0, "orth_w": 0.0, "corr_w": 0.0, "bias": True, "constant_function": True,
        "num_mini_batches": 8, "mini_batch_size": 256, "beta_initial": 0.4, "beta_annealing_steps": 20000,
    }

    ecdae_model_cfg = {
        "name": 'ecdae', "equivariant": True, "activation": 'ELU', "num_layers": 5, "num_hidden_units": 128,
        "batch_norm": False, "obs_pred_w": 1.0, "orth_w": 0.0, "corr_w": 0.0, "bias": True, "constant_function": True,
        "num_mini_batches": 8, "mini_batch_size": 256, "beta_initial": 0.4, "beta_annealing_steps": 20000,
        "group_avg_trick": True, "state_dependent_obs_dyn": False,
    }

    robot_cfg = {
            "name": "a1", "lr": 1e-3, "max_epochs": 200, "obs_state_ratio": 3,
            "state_obs": ['projected_gravity', 'projected_forward_vec', 'joint_pos', 'prev_actions', 'phase_input', 'base_pos', 'door_bottom_corner_pos', 'door_normal_vec', 'lr_vec'],
            "action_obs": ['actions'], "state_dim": 3 + 3 + 12 + 12 + 2 + 3 + 3 + 3 + 2,
            "action_dim": 12, "pred_horizon": 5, "frames_per_state": 1,
    }

    cdae_koopman_cfg = {"model": cdae_model_cfg, "robot": robot_cfg}
    ecdae_koopman_cfg = {"model": ecdae_model_cfg, "robot": robot_cfg}

    # Evaluate standard CDAE
    evaluate_offline_data(
        model_dir=CDAE_MODEL_DIR,
        data_path=CDAE_DATA_PATH,                  # CDAE's own right-side data
        left_data_path=ECDAE_LEFT_DATA_PATH,       # <--- CROSS-EVALUATION HERE!
        koopman_cfg=cdae_koopman_cfg,
        task="push_door",
        dae_type="cDAE",
        horizon=5
    )

    # Evaluate ECDAE (Includes Left-Side Ablation test)
    evaluate_offline_data(
        model_dir=ECDAE_MODEL_DIR,
        data_path=ECDAE_DATA_PATH,                 # ECDAE's own right-side data
        left_data_path=ECDAE_LEFT_DATA_PATH,       # ECDAE's own left-side data
        koopman_cfg=ecdae_koopman_cfg,
        task="push_door",
        dae_type="ecDAE",
        horizon=5
    )