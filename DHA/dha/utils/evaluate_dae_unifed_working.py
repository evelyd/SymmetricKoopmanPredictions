import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import escnn.nn.modules.basismanager.basisexpansion_singleblock as bes

# Dynamically sniff out the correct class name in your version of escnn
for name, obj in vars(bes).items():
    # Find any PyTorch Module defined in this file
    if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
        orig_forward = obj.forward

        # Create a closure to hold the original forward function
        def make_patched_forward(original_fn):
            def patched_forward(self, weights, *args, **kwargs):
                # If the basis and weights are on different devices, align them!
                if hasattr(self, 'sampled_basis') and self.sampled_basis.device != weights.device:
                    self.to(weights.device)
                return original_fn(self, weights, *args, **kwargs)
            return patched_forward

        # Apply the patch
        obj.forward = make_patched_forward(orig_forward)

# Import your helper functions and configurations from utils.py
from utils import get_trained_dae_model_from_pt
from dha.utils.mysc import batched_to_flat_trajectory

def load_and_reshape(path, device):
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

def evaluate_and_extract_metrics(model_dir: str, data_path: str, left_data_path: str, koopman_cfg: dict, task: str, dae_type: str, horizon: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating {dae_type}...")

    # Load Model
    model_files = [f for f in os.listdir(model_dir) if f.startswith('dae_model_') and f.endswith('.pt')]
    latest_model = max(model_files, key=lambda x: int(x.replace('dae_model_', '').replace('.pt', '')))
    model_pt_path = os.path.join(model_dir, latest_model)
    model = get_trained_dae_model_from_pt(model_pt_path, koopman_cfg, task=dae_type.lower(), dt=0.02, device=device)
    model.eval()

    # Load Data
    raw_obs, raw_actions = load_and_reshape(data_path, device)
    raw_left_obs, raw_left_actions = load_and_reshape(left_data_path, device)

    dae_state_dim = model.state_type.size if hasattr(model, 'obs_state_type') else model.state_dim
    dae_raw_obs = raw_obs[:, :, :dae_state_dim]

    # Normalization
    state_variance_approx = torch.mean(dae_raw_obs**2, dim=(0, 1))
    state_scale = torch.sqrt(state_variance_approx)
    state_scale[state_scale < 1e-4] = 1.0

    action_variance_approx = torch.mean(raw_actions**2, dim=(0, 1))
    action_scale = torch.sqrt(action_variance_approx)
    action_scale[action_scale < 1e-4] = 1.0

    norm_obs = dae_raw_obs / state_scale
    norm_actions = raw_actions / action_scale
    if raw_left_obs is not None:
        norm_left_obs = raw_left_obs[:, :, :dae_state_dim] / state_scale
        norm_left_actions = raw_left_actions / action_scale

    # Windowing Helper
    def create_windows(n_obs, n_act):
        num_steps = n_obs.shape[1]
        s_list, ns_list, a_list = [], [], []
        for t in range(num_steps - horizon):
            s_list.append(n_obs[:, t, :])
            ns_list.append(n_obs[:, t+1 : t+1+horizon, :])
            a_list.append(n_act[:, t : t+horizon, :])
        return torch.cat(s_list, dim=0), torch.cat(ns_list, dim=0), torch.cat(a_list, dim=0)

    # ---------------------------------------------------------
    # RIGHT SIDE EVALUATION (Training Distribution)
    # ---------------------------------------------------------
    states, next_states, actions = create_windows(norm_obs, norm_actions)
    gt_state_traj = torch.cat([states.unsqueeze(1), next_states], dim=1)

    with torch.no_grad():
        outputs = model(state=states, action=actions, next_state=next_states)
        baseline_right_mse = torch.nn.functional.mse_loss(outputs["pred_state_traj"], gt_state_traj).item()

        # Get Recon and 1-Step Latent Loss from internal metrics function
        _, metrics = model.compute_loss_and_metrics(
            state=states, action=actions, next_state=next_states,
            pred_state_traj=outputs["pred_state_traj"],
            rec_state_traj=outputs["rec_state_traj"],
            obs_state_traj=outputs["obs_state_traj"],
            pred_obs_state_traj=outputs["pred_obs_state_traj"],
            pred_obs_state_one_step=outputs.get("pred_obs_state_one_step", None)
        )

        # Average the per-step metrics
        recon_mse = metrics['state_rec_loss'].mean().item()
        latent_1step_mse = metrics['obs_pred_loss'].mean().item()

    # ---------------------------------------------------------
    # LEFT SIDE EVALUATION & ABLATION (Zero-Shot Symmetry)
    # ---------------------------------------------------------
    baseline_left_mse, ablated_right_mse, ablated_left_mse, ablated_impact_pct_r, ablated_impact_pct, invariant_lambda = None, None, None, None, None, None

    if raw_left_obs is not None:
        left_states, left_next_states, left_actions = create_windows(norm_left_obs, raw_left_actions)
        gt_left_state_traj = torch.cat([left_states.unsqueeze(1), left_next_states], dim=1)

        with torch.no_grad():
            outputs_left = model(state=left_states, action=left_actions, next_state=left_next_states)
            baseline_left_mse = torch.nn.functional.mse_loss(outputs_left["pred_state_traj"], gt_left_state_traj).item()

        if dae_type.lower() == "ecdae":
            identity_a = torch.eye(model.obs_state_type.size, device=device)
            A = model.obs_space_dynamics.transfer_op(model.obs_state_type(identity_a)).tensor.detach().T
            identity_b = torch.eye(model.action_type.size, device=device)
            B = model.obs_space_dynamics.control_op(model.action_type(identity_b)).tensor.detach().T

            bias = torch.zeros(model.obs_state_type.size, device=device)
            if hasattr(model.obs_space_dynamics, 'bias') and model.obs_space_dynamics.bias is not None:
                if hasattr(model.obs_space_dynamics.bias, 'tensor'):
                    bias = model.obs_space_dynamics.bias.tensor.detach().squeeze()
                elif isinstance(model.obs_space_dynamics.bias, torch.Tensor):
                    bias = model.obs_space_dynamics.bias.detach().squeeze()

            A_np = A.cpu().numpy()
            eigvals, eigvecs = np.linalg.eig(A_np)

            target_idx = np.argmin(np.abs(np.abs(eigvals) - 1.0))
            invariant_lambda = eigvals[target_idx]

            eigvals_ablated = eigvals.copy()
            eigvals_ablated[target_idx] = 0.0
            A_ablated_np = np.real(eigvecs @ np.diag(eigvals_ablated) @ np.linalg.inv(eigvecs))
            A_ablated = torch.tensor(A_ablated_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                z_t_r = model.obs_fn(model.pre_process_state(state=states)).tensor
                ablated_obs_traj_r = [z_t_r]
                for t in range(horizon):
                    u_t_r = actions[:, t, :]
                    z_t_r = z_t_r @ A_ablated.T + u_t_r @ B.T + bias
                    ablated_obs_traj_r.append(z_t_r)

                ablated_obs_traj_r = torch.stack(ablated_obs_traj_r, dim=1)
                ablated_obs_geom_r = model.obs_state_type(batched_to_flat_trajectory(ablated_obs_traj_r))
                ablated_pred_state_traj_r = model.post_process_state(model.inv_obs_fn(ablated_obs_geom_r))
                ablated_right_mse = torch.nn.functional.mse_loss(ablated_pred_state_traj_r, gt_state_traj).item()

                ablated_impact_pct_r = ((ablated_right_mse - baseline_right_mse) / baseline_right_mse) * 100

                z_t = model.obs_fn(model.pre_process_state(state=left_states)).tensor
                ablated_obs_traj = [z_t]
                for t in range(horizon):
                    u_t = left_actions[:, t, :]
                    z_t = z_t @ A_ablated.T + u_t @ B.T + bias
                    ablated_obs_traj.append(z_t)

                ablated_obs_traj = torch.stack(ablated_obs_traj, dim=1)
                ablated_obs_geom = model.obs_state_type(batched_to_flat_trajectory(ablated_obs_traj))
                ablated_pred_state_traj = model.post_process_state(model.inv_obs_fn(ablated_obs_geom))
                ablated_left_mse = torch.nn.functional.mse_loss(ablated_pred_state_traj, gt_left_state_traj).item()

                ablated_impact_pct = ((ablated_left_mse - baseline_left_mse) / baseline_left_mse) * 100

    return {
        "recon_mse": recon_mse,
        "latent_1step_mse": latent_1step_mse,
        "baseline_right_mse": baseline_right_mse,
        "baseline_left_mse": baseline_left_mse,
        "invariant_lambda": invariant_lambda,
        "ablated_left_mse": ablated_left_mse,
        "ablated_right_mse": ablated_right_mse,
        "ablated_impact_pct_l": ablated_impact_pct,
        "ablated_impact_pct_r": ablated_impact_pct_r
    }

# =====================================================================
# NEW LONG-TRAJECTORY PLOTTING FUNCTIONS
# =====================================================================

def save_trajectory_plots(y_true, preds_cdae, preds_ecdae, save_dir, feature_indices, feature_names):
    """Saves long trajectory comparison plots without displaying them."""
    os.makedirs(save_dir, exist_ok=True)
    horizons = [1, 5, 10]

    # This ensures the X-axis plots the full T-step trajectory (e.g., 1000 steps)
    timesteps = np.arange(y_true.shape[0])

    for h in horizons:
        if h not in preds_cdae or h not in preds_ecdae:
            continue

        fig, axes = plt.subplots(len(feature_indices), 1, figsize=(16, 4 * len(feature_indices)))
        if len(feature_indices) == 1: axes = [axes]

        for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
            ax = axes[i]

            # Ground truth (full length)
            ax.plot(timesteps, y_true[:, idx], label='True (.npy)', color='black', linestyle='-', linewidth=2)

            # Predictions (padded to align with the ground truth timeline)
            ax.plot(timesteps, preds_cdae[h][:, idx], label=f'cDAE ({h}-step ahead)', color='blue', alpha=0.8, linewidth=1.5)
            ax.plot(timesteps, preds_ecdae[h][:, idx], label=f'ecDAE ({h}-step ahead)', color='red', alpha=0.8, linewidth=1.5)

            ax.set_title(f'{h}-Step Ahead Prediction over Time: {name}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Normalized Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'long_trajectory_comp_horizon_{h}.png')
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # Prevent display
    print(f"--> Saved long trajectory plots to {save_dir}")

def generate_comparison_plots(cdae_dir, ecdae_dir, data_path, cdae_cfg, ecdae_cfg, save_dir, max_steps=1000):
    """Evaluates the models dynamically over a long, continuous trajectory using a sliding window."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGenerating {max_steps}-step continuous trajectory plots for horizons [1, 5, 10]...")

    # Load Models
    def load_latest_model(model_dir, cfg, task_name):
        files = [f for f in os.listdir(model_dir) if f.startswith('dae_model_') and f.endswith('.pt')]
        latest = max(files, key=lambda x: int(x.replace('dae_model_', '').replace('.pt', '')))
        model = get_trained_dae_model_from_pt(os.path.join(model_dir, latest), cfg, task=task_name, dt=0.02, device=device)
        model.eval()
        return model

    model_cdae = load_latest_model(cdae_dir, cdae_cfg, "cdae")
    model_ecdae = load_latest_model(ecdae_dir, ecdae_cfg, "ecdae")

    # Load and normalize data
    raw_obs, raw_actions = load_and_reshape(data_path, device)
    dae_state_dim = model_cdae.state_type.size if hasattr(model_cdae, 'obs_state_type') else model_cdae.state_dim
    dae_raw_obs = raw_obs[:, :, :dae_state_dim]

    state_scale = torch.sqrt(torch.mean(dae_raw_obs**2, dim=(0, 1)))
    state_scale[state_scale < 1e-4] = 1.0
    action_scale = torch.sqrt(torch.mean(raw_actions**2, dim=(0, 1)))
    action_scale[action_scale < 1e-4] = 1.0

    norm_obs = dae_raw_obs / state_scale
    norm_actions = raw_actions / action_scale

    # Grab the very first environment run
    env_idx = 0
    obs_seq = norm_obs[env_idx]
    act_seq = norm_actions[env_idx]

    # Limit trajectory length
    T = min(max_steps, obs_seq.shape[0])
    obs_seq = obs_seq[:T]
    act_seq = act_seq[:T]

    preds_cdae = {}
    preds_ecdae = {}
    horizons = [1, 5, 10]

    for h in horizons:
        if T <= h: continue

        # Build batched sliding windows for the continuous trajectory
        s_list, ns_list, a_list = [], [], []
        for t in range(T - h):
            s_list.append(obs_seq[t])
            ns_list.append(obs_seq[t+1 : t+1+h])
            a_list.append(act_seq[t : t+h])

        states = torch.stack(s_list)
        next_states = torch.stack(ns_list)
        actions = torch.stack(a_list)

        with torch.no_grad():
            out_cdae = model_cdae(state=states, action=actions, next_state=next_states)
            out_ecdae = model_ecdae(state=states, action=actions, next_state=next_states)

        # Extract the h-step prediction for every window
        pred_c_traj = out_cdae["pred_state_traj"][:, -1, :].cpu().numpy()
        pred_e_traj = out_ecdae["pred_state_traj"][:, -1, :].cpu().numpy()

        # Pad the first 'h' steps with NaNs so the arrays align perfectly with the T-step ground truth timeline
        state_dim = obs_seq.shape[-1]
        padded_c = np.full((T, state_dim), np.nan)
        padded_e = np.full((T, state_dim), np.nan)

        padded_c[h:] = pred_c_traj
        padded_e[h:] = pred_e_traj

        preds_cdae[h] = padded_c
        preds_ecdae[h] = padded_e

    # Real target trajectory as a standard numpy array (Full length T)
    y_true = obs_seq.cpu().numpy()

    # --- DENORMALIZE ---
    scale_np = state_scale.cpu().numpy()

    # Denormalize ground truth
    y_true = y_true * scale_np

    # Denormalize predictions
    for h in horizons:
        if h in preds_cdae:
            preds_cdae[h] = preds_cdae[h] * scale_np
        if h in preds_ecdae:
            preds_ecdae[h] = preds_ecdae[h] * scale_np

    # --- CORRECTED INDICES ---
    # Based on config: gravity(3) + fwd_vec(3) + joint_pos(12)
    # Joint block starts at index 6.
    # FL(0,1,2), FR(3,4,5), RL(6,7,8), RR(9,10,11)
    feature_indices = [13, 14, 16, 17]
    feature_names = ["Rear Left Hip Pitch", "Rear Left Knee", "Rear Right Hip Pitch", "Rear Right Knee"]

    save_trajectory_plots(y_true, preds_cdae, preds_ecdae, save_dir, feature_indices, feature_names)
# =====================================================================
# =====================================================================


def print_latex_table(cdae_res, ecdae_res, cdae_res_std=None, ecdae_res_std=None):
    # Helper to format complex numbers safely
    def fmt_lambda(val, std=None):
        if val is None: return "N/A"
        base = f"{val.real:.3f}" if abs(val.imag) < 1e-4 else f"{val.real:.3f} + {val.imag:.3f}i"
        if std is not None:
            std_str = f"{std.real:.3f}" if abs(std.imag) < 1e-4 else f"{std.real:.3f} + {std.imag:.3f}i"
            return f"${base} \\pm {std_str}$"
        return f"${base}$"

    # Helper to format standard values with std and math mode
    def fmt_val(val, std=None, prefix="", suffix=""):
        if val is None: return "N/A"
        if std is not None:
            return f"${prefix}{val:.3f} \\pm {std:.3f}{suffix}$"
        return f"${prefix}{val:.3f}{suffix}$"

    # Format CDAE
    c_recon = fmt_val(cdae_res['recon_mse'], cdae_res_std['recon_mse'] if cdae_res_std else None)
    c_1step = fmt_val(cdae_res['latent_1step_mse'], cdae_res_std['latent_1step_mse'] if cdae_res_std else None)
    c_5step_r = fmt_val(cdae_res['baseline_right_mse'], cdae_res_std['baseline_right_mse'] if cdae_res_std else None)
    c_5step_l = fmt_val(cdae_res['baseline_left_mse'], cdae_res_std['baseline_left_mse'] if cdae_res_std else None)

    # Format ECDAE
    e_recon = fmt_val(ecdae_res['recon_mse'], ecdae_res_std['recon_mse'] if ecdae_res_std else None)
    e_1step = fmt_val(ecdae_res['latent_1step_mse'], ecdae_res_std['latent_1step_mse'] if ecdae_res_std else None)
    e_5step_r = fmt_val(ecdae_res['baseline_right_mse'], ecdae_res_std['baseline_right_mse'] if ecdae_res_std else None)
    e_5step_l = fmt_val(ecdae_res['baseline_left_mse'], ecdae_res_std['baseline_left_mse'] if ecdae_res_std else None)

    e_lambda = fmt_lambda(ecdae_res['invariant_lambda'], ecdae_res_std.get('invariant_lambda') if ecdae_res_std else None)

    e_ablate_r = fmt_val(ecdae_res.get('ablated_right_mse'), ecdae_res_std.get('ablated_right_mse') if ecdae_res_std else None)
    e_ablate_l = fmt_val(ecdae_res.get('ablated_left_mse'), ecdae_res_std.get('ablated_left_mse') if ecdae_res_std else None)

    # Impact percentages
    e_impact_r = fmt_val(ecdae_res.get('ablated_impact_pct_r'), ecdae_res_std.get('ablated_impact_pct_r') if ecdae_res_std else None, prefix="+", suffix="\\%")
    e_impact_l = fmt_val(ecdae_res.get('ablated_impact_pct_l'), ecdae_res_std.get('ablated_impact_pct_l') if ecdae_res_std else None, prefix="+", suffix="\\%")

    latex_str = f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Standard cDAE}} & \\textbf{{Equivariant ecDAE}}\\\\
\\midrule
\\textbf{{Reconstruction MSE}} & {c_recon} & \\textbf{{{e_recon}}} \\\\
\\textbf{{1-Step Latent MSE}}  & \\textbf{{{c_1step}}} & {e_1step} \\\\
\\textbf{{5-Step MSE (Right)}} & {c_5step_r} & \\textbf{{{e_5step_r}}} \\\\
\\textbf{{5-Step MSE (Left)}}  & {c_5step_l} & \\textbf{{{e_5step_l}}} \\\\
\\textbf{{Invariant $\\lambda$}} & N/A & \\textbf{{{e_lambda}}} \\\\
\\textbf{{Ablated MSE (Right)}}& N/A & {e_ablate_r} ({e_impact_r}) \\\\
\\textbf{{Ablated MSE (Left)}} & N/A & {e_ablate_l} ({e_impact_l}) \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Quantitative comparison of the standard continuous Koopman Autoencoder (cDAE) and the symmetry-constrained Equivariant Koopman Autoencoder (ecDAE). Error metrics indicate Mean Squared Error (MSE).}}
\\label{{tab:dae_comparison}}
\\end{{table}}
"""
    print("\n" + "="*80)
    print(" GENERATED LATEX TABLE ")
    print("="*80)
    print(latex_str)


if __name__ == "__main__":

    home_dir = os.path.expanduser("~")
    # Define your paths
    cdae_task = "push_door_cyber_cdae_online_next_latent"
    ecdae_task = "push_door_cyber_emlp_ecdae_online_next_latent"
    cdae_model_dirs = sorted([d for d in os.listdir(f"{home_dir}/git/koopman_symmloco/legged_gym/logs/{cdae_task}/") if "202" in d and not d.startswith("2026-03-03-")])
    ecdae_model_dirs = sorted([d for d in os.listdir(f"{home_dir}/git/koopman_symmloco/legged_gym/logs/{ecdae_task}/") if "202" in d and not d.startswith("2026-03-03-")])

    # Get the first .pt model file found
    def get_model_file(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.startswith('dae_model_') and f.endswith('.pt')]
        if not model_files:
            return None
        # Return the one with the highest iteration number to match evaluate_and_extract_metrics
        return max(model_files, key=lambda x: int(x.replace('dae_model_', '').replace('.pt', '')))

    # Define your configs
    cdae_model_cfg = {"name": 'cdae', "equivariant": False, "activation": 'ELU', "num_layers": 5, "num_hidden_units": 128, "batch_norm": False, "obs_pred_w": 1.0, "orth_w": 0.0, "corr_w": 0.0, "bias": True, "constant_function": True, "num_mini_batches": 8, "mini_batch_size": 256, "beta_initial": 0.4, "beta_annealing_steps": 20000}
    ecdae_model_cfg = {"name": 'ecdae', "equivariant": True, "activation": 'ELU', "num_layers": 5, "num_hidden_units": 128, "batch_norm": False, "obs_pred_w": 1.0, "orth_w": 0.0, "corr_w": 0.0, "bias": True, "constant_function": True, "num_mini_batches": 8, "mini_batch_size": 256, "beta_initial": 0.4, "beta_annealing_steps": 20000, "group_avg_trick": True, "state_dependent_obs_dyn": False}
    robot_cfg = {"name": "a1", "lr": 1e-3, "max_epochs": 200, "obs_state_ratio": 3, "state_obs": ['projected_gravity', 'projected_forward_vec', 'joint_pos', 'prev_actions', 'phase_input', 'base_pos', 'door_bottom_corner_pos', 'door_normal_vec', 'lr_vec'], "action_obs": ['actions'], "state_dim": 3+3+12+12+2+3+3+3+2, "action_dim": 12, "pred_horizon": 5, "frames_per_state": 1}

    res_cdae_agg = {"recon_mse": [], "latent_1step_mse": [], "baseline_right_mse": [], "baseline_left_mse": []}
    res_ecdae_agg = {"recon_mse": [], "latent_1step_mse": [], "baseline_right_mse": [], "baseline_left_mse": [], "invariant_lambda": [], "ablated_left_mse": [], "ablated_right_mse": [], "ablated_impact_pct_l": [], "ablated_impact_pct_r": []}

    for i, (cdae_seed, ecdae_seed) in enumerate(zip(cdae_model_dirs, ecdae_model_dirs)):
        print(f"Evaluating cDAE model from seed {cdae_seed} and ecDAE model from seed {ecdae_seed}...")
        cdae_model_path = f"{home_dir}/git/koopman_symmloco/legged_gym/logs/{cdae_task}/{cdae_seed}"
        ecdae_model_path = f"{home_dir}/git/koopman_symmloco/legged_gym/logs/{ecdae_task}/{ecdae_seed}"
        cdae_data_path = f"{home_dir}/git/koopman_symmloco/legged_gym/isaacgym_recordings/{cdae_task}/{cdae_seed}/model_{get_model_file(cdae_model_path).replace('dae_model_', '').replace('.pt', '')}_obs_action.npy"
        ecdae_data_path = f"{home_dir}/git/koopman_symmloco/legged_gym/isaacgym_recordings/{ecdae_task}/{ecdae_seed}/model_{get_model_file(ecdae_model_path).replace('dae_model_', '').replace('.pt', '')}_obs_action.npy"
        ecdae_left_data_path = f"{home_dir}/git/koopman_symmloco/legged_gym/isaacgym_recordings/{ecdae_task}/{ecdae_seed}/model_{get_model_file(ecdae_model_path).replace('dae_model_', '').replace('.pt', '')}_obs_action_left.npy"

        # --- PLOTTING HOOK ---
        # Only run the plotting loop for the first seed to prevent spamming your file system
        if i == 0:
            plot_save_dir = f"{home_dir}/git/koopman_symmloco/legged_gym/logs/trajectory_plots"
            generate_comparison_plots(
                cdae_dir=cdae_model_path,
                ecdae_dir=ecdae_model_path,
                data_path=ecdae_data_path,
                cdae_cfg={"model": cdae_model_cfg, "robot": robot_cfg},
                ecdae_cfg={"model": ecdae_model_cfg, "robot": robot_cfg},
                save_dir=plot_save_dir,
                max_steps=1000
            )
        # ---------------------

        res_cdae = evaluate_and_extract_metrics(cdae_model_path, cdae_data_path, ecdae_left_data_path, {"model": cdae_model_cfg, "robot": robot_cfg}, "push_door", "cDAE", 5)
        res_ecdae = evaluate_and_extract_metrics(ecdae_model_path, ecdae_data_path, ecdae_left_data_path, {"model": ecdae_model_cfg, "robot": robot_cfg}, "push_door", "ecDAE", 5)

        # Print results for this pair of models
        print(f"\nResults for cDAE (seed {cdae_seed}):")
        for k, v in res_cdae.items():
            print(f"  {k}: {v}")
        print(f"\nResults for ecDAE (seed {ecdae_seed}):")
        for k, v in res_ecdae.items():
            print(f"  {k}: {v}")

        # Append the results
        res_cdae_agg["recon_mse"].append(res_cdae['recon_mse'])
        res_cdae_agg["latent_1step_mse"].append(res_cdae['latent_1step_mse'])
        res_cdae_agg["baseline_right_mse"].append(res_cdae['baseline_right_mse'])
        res_cdae_agg["baseline_left_mse"].append(res_cdae['baseline_left_mse'])

        res_ecdae_agg["recon_mse"].append(res_ecdae['recon_mse'])
        res_ecdae_agg["latent_1step_mse"].append(res_ecdae['latent_1step_mse'])
        res_ecdae_agg["baseline_right_mse"].append(res_ecdae['baseline_right_mse'])
        res_ecdae_agg["baseline_left_mse"].append(res_ecdae['baseline_left_mse'])
        res_ecdae_agg["invariant_lambda"].append(res_ecdae['invariant_lambda'])
        res_ecdae_agg["ablated_left_mse"].append(res_ecdae['ablated_left_mse'])
        res_ecdae_agg["ablated_right_mse"].append(res_ecdae['ablated_right_mse'])
        res_ecdae_agg["ablated_impact_pct_l"].append(res_ecdae['ablated_impact_pct_l'])
        res_ecdae_agg["ablated_impact_pct_r"].append(res_ecdae['ablated_impact_pct_r'])

    # Get the average metrics across seeds
    res_cdae_agg_std = {k: np.std(v) for k, v in res_cdae_agg.items()}
    res_ecdae_agg_std = {k: np.std(v) for k, v in res_ecdae_agg.items()}

    res_cdae_agg = {k: np.mean(v) for k, v in res_cdae_agg.items()}
    res_ecdae_agg = {k: np.mean(v) for k, v in res_ecdae_agg.items()}

    # Output formatted LaTeX Table
    print_latex_table(res_cdae_agg, res_ecdae_agg, res_cdae_agg_std, res_ecdae_agg_std)