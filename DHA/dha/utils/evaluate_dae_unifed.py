import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import re

import escnn.nn.modules.basismanager.basisexpansion_singleblock as bes

# Import the running standard scaler used during training
from rsl_rl.storage import RunningStdScaler

# Dynamically sniff out the correct class name in your version of escnn
for name, obj in vars(bes).items():
    if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
        orig_forward = obj.forward
        def make_patched_forward(original_fn):
            def patched_forward(self, weights, *args, **kwargs):
                if hasattr(self, 'sampled_basis') and self.sampled_basis.device != weights.device:
                    self.to(weights.device)
                return original_fn(self, weights, *args, **kwargs)
            return patched_forward
        obj.forward = make_patched_forward(orig_forward)

from utils import get_trained_dae_model_from_pt
from dha.utils.mysc import batched_to_flat_trajectory

# =====================================================================
# ROBUST FILE FINDER
# =====================================================================
def get_highest_iter_file(directory, prefix="", suffix=""):
    if not os.path.exists(directory):
        return None
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)]
    if not files:
        return None
    def extract_iter(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else -1
    return max(files, key=extract_iter)

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

def get_std_scales(normalizer, state_dim, action_dim, device):
    """Safely extracts ONLY the standard deviation scaling factors."""
    zero_obs = torch.zeros(1, state_dim, device=device)
    one_obs = torch.ones(1, state_dim, device=device)
    zero_act = torch.zeros(1, action_dim, device=device)
    one_act = torch.ones(1, action_dim, device=device)

    norm_zero_obs, norm_zero_act = normalizer.normalize(zero_obs, zero_act)
    norm_one_obs, norm_one_act = normalizer.normalize(one_obs, one_act)

    std_obs = 1.0 / (norm_one_obs - norm_zero_obs)
    std_act = 1.0 / (norm_one_act - norm_zero_act)
    return std_obs.detach(), std_act.detach()

def evaluate_and_extract_metrics(model_dir: str, data_path: str, left_data_path: str, koopman_cfg: dict, task: str, dae_type: str, horizon: int = 5, forced_normalizer_dict=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating {dae_type} in {model_dir.split('/')[-1]}...")

    latest_model = get_highest_iter_file(model_dir, prefix='dae_model_', suffix='.pt')
    model_pt_path = os.path.join(model_dir, latest_model)

    checkpoint = torch.load(model_pt_path, map_location=device)
    model = get_trained_dae_model_from_pt(model_pt_path, koopman_cfg, task=dae_type.lower(), dt=0.02, device=device)

    if 'dae_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['dae_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    dae_state_dim = model.state_type.size if hasattr(model, 'obs_state_type') else model.state_dim
    action_dim = koopman_cfg["robot"]["action_dim"]

    raw_obs, raw_actions = load_and_reshape(data_path, device)
    raw_left_obs, raw_left_actions = load_and_reshape(left_data_path, device)

    dae_raw_obs = raw_obs[:, :, :dae_state_dim]

    # --- STRICT STD-ONLY NORMALIZATION ---
    if forced_normalizer_dict is not None or 'normalizer_state_dict' in checkpoint:
        normalizer = RunningStdScaler(dae_state_dim, action_dim, device=device)
        if forced_normalizer_dict is not None:
            normalizer.load_state_dict(forced_normalizer_dict)
        else:
            normalizer.load_state_dict(checkpoint['normalizer_state_dict'])

        state_std, action_std = get_std_scales(normalizer, dae_state_dim, action_dim, device)
        norm_obs = dae_raw_obs / state_std
        norm_actions = raw_actions / action_std
        if raw_left_obs is not None:
            norm_left_obs = raw_left_obs[:, :, :dae_state_dim] / state_std
            norm_left_actions = raw_left_actions / action_std
    else:
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

    def create_windows(n_obs, n_act):
        num_steps = n_obs.shape[1]
        s_list, ns_list, a_list = [], [], []
        for t in range(num_steps - horizon):
            s_list.append(n_obs[:, t, :])
            ns_list.append(n_obs[:, t+1 : t+1+horizon, :])
            a_list.append(n_act[:, t : t+horizon, :])
        return torch.cat(s_list, dim=0), torch.cat(ns_list, dim=0), torch.cat(a_list, dim=0)

    states, next_states, actions = create_windows(norm_obs, norm_actions)
    gt_state_traj = torch.cat([states.unsqueeze(1), next_states], dim=1)

    with torch.no_grad():
        outputs = model(state=states, action=actions, next_state=next_states)
        baseline_right_mse = torch.nn.functional.mse_loss(outputs["pred_state_traj"], gt_state_traj).item()

        _, metrics = model.compute_loss_and_metrics(
            state=states, action=actions, next_state=next_states,
            pred_state_traj=outputs["pred_state_traj"],
            rec_state_traj=outputs["rec_state_traj"],
            obs_state_traj=outputs["obs_state_traj"],
            pred_obs_state_traj=outputs["pred_obs_state_traj"],
            pred_obs_state_one_step=outputs.get("pred_obs_state_one_step", None)
        )

        recon_mse = metrics['state_rec_loss'].mean().item()
        latent_1step_mse = metrics['obs_pred_loss'].mean().item()

    baseline_left_mse, ablated_right_mse, ablated_left_mse, invariant_lambda = None, None, None, None
    only_inv_right_mse, only_inv_left_mse = None, None

    if raw_left_obs is not None:
        left_states, left_next_states, left_actions = create_windows(norm_left_obs, norm_left_actions)
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

            eigvals_only_inv = np.zeros_like(eigvals)
            eigvals_only_inv[target_idx] = eigvals[target_idx]
            A_only_inv_np = np.real(eigvecs @ np.diag(eigvals_only_inv) @ np.linalg.inv(eigvecs))
            A_only_inv = torch.tensor(A_only_inv_np, dtype=torch.float32, device=device)

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
                ablated_pred_state_traj_r = ablated_pred_state_traj_r.view(states.shape[0], horizon + 1, -1)
                ablated_right_mse = torch.nn.functional.mse_loss(ablated_pred_state_traj_r, gt_state_traj).item()

                z_t = model.obs_fn(model.pre_process_state(state=left_states)).tensor
                ablated_obs_traj = [z_t]
                for t in range(horizon):
                    u_t = left_actions[:, t, :]
                    z_t = z_t @ A_ablated.T + u_t @ B.T + bias
                    ablated_obs_traj.append(z_t)
                ablated_obs_traj = torch.stack(ablated_obs_traj, dim=1)
                ablated_obs_geom = model.obs_state_type(batched_to_flat_trajectory(ablated_obs_traj))
                ablated_pred_state_traj = model.post_process_state(model.inv_obs_fn(ablated_obs_geom))
                ablated_pred_state_traj = ablated_pred_state_traj.view(left_states.shape[0], horizon + 1, -1)
                ablated_left_mse = torch.nn.functional.mse_loss(ablated_pred_state_traj, gt_left_state_traj).item()

                z_t_r_only = model.obs_fn(model.pre_process_state(state=states)).tensor
                only_inv_obs_traj_r = [z_t_r_only]
                for t in range(horizon):
                    u_t_r = actions[:, t, :]
                    z_t_r_only = z_t_r_only @ A_only_inv.T + u_t_r @ B.T + bias
                    only_inv_obs_traj_r.append(z_t_r_only)
                only_inv_obs_traj_r = torch.stack(only_inv_obs_traj_r, dim=1)
                only_inv_obs_geom_r = model.obs_state_type(batched_to_flat_trajectory(only_inv_obs_traj_r))
                only_inv_pred_state_traj_r = model.post_process_state(model.inv_obs_fn(only_inv_obs_geom_r))
                only_inv_pred_state_traj_r = only_inv_pred_state_traj_r.view(states.shape[0], horizon + 1, -1)
                only_inv_right_mse = torch.nn.functional.mse_loss(only_inv_pred_state_traj_r, gt_state_traj).item()

                z_t_l_only = model.obs_fn(model.pre_process_state(state=left_states)).tensor
                only_inv_obs_traj_l = [z_t_l_only]
                for t in range(horizon):
                    u_t_l = left_actions[:, t, :]
                    z_t_l_only = z_t_l_only @ A_only_inv.T + u_t_l @ B.T + bias
                    only_inv_obs_traj_l.append(z_t_l_only)
                only_inv_obs_traj_l = torch.stack(only_inv_obs_traj_l, dim=1)
                only_inv_obs_geom_l = model.obs_state_type(batched_to_flat_trajectory(only_inv_obs_traj_l))
                only_inv_pred_state_traj_l = model.post_process_state(model.inv_obs_fn(only_inv_obs_geom_l))
                only_inv_pred_state_traj_l = only_inv_pred_state_traj_l.view(left_states.shape[0], horizon + 1, -1)
                only_inv_left_mse = torch.nn.functional.mse_loss(only_inv_pred_state_traj_l, gt_left_state_traj).item()

    return {
        "recon_mse": recon_mse,
        "latent_1step_mse": latent_1step_mse,
        "baseline_right_mse": baseline_right_mse,
        "baseline_left_mse": baseline_left_mse,
        "invariant_lambda": invariant_lambda,
        "ablated_left_mse": ablated_left_mse,
        "ablated_right_mse": ablated_right_mse,
        "only_inv_left_mse": only_inv_left_mse,
        "only_inv_right_mse": only_inv_right_mse
    }

def save_trajectory_plots(y_true, preds_cdae, preds_ecdae, preds_ecdae_only_inv, preds_ecdae_no_inv, save_dir, feature_indices, feature_names):
    os.makedirs(save_dir, exist_ok=True)
    horizons = [1, 5, 10]
    timesteps = np.arange(y_true.shape[0])

    for h in horizons:
        if h not in preds_cdae or h not in preds_ecdae:
            continue

        fig, axes = plt.subplots(len(feature_indices), 1, figsize=(16, 4 * len(feature_indices)))
        if len(feature_indices) == 1: axes = [axes]

        for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
            ax = axes[i]

            ax.plot(timesteps, y_true[:, idx], label='True (.npy)', color='black', linestyle='-', linewidth=2)
            ax.plot(timesteps, preds_cdae[h][:, idx], label=f'cDAE ({h}-step ahead)', color='blue', alpha=0.7, linewidth=1.5)
            ax.plot(timesteps, preds_ecdae[h][:, idx], label=f'ecDAE ({h}-step ahead)', color='red', alpha=0.7, linewidth=1.5)

            if h in preds_ecdae_only_inv:
                ax.plot(timesteps, preds_ecdae_only_inv[h][:, idx], label=f'ecDAE ONLY Inv ({h}-step)', color='green', alpha=0.9, linewidth=2, linestyle='--')

            if h in preds_ecdae_no_inv:
                ax.plot(timesteps, preds_ecdae_no_inv[h][:, idx], label=f'ecDAE NO Inv ({h}-step)', color='orange', alpha=0.9, linewidth=2, linestyle=':')

            ax.set_title(f'{h}-Step Ahead Prediction over Time: {name}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Unnormalized Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'long_trajectory_comp_horizon_{h}.png')
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    print(f"--> Saved long trajectory plots to {save_dir}")

def generate_comparison_plots(cdae_dir, ecdae_dir, data_path, cdae_cfg, ecdae_cfg, save_dir, max_steps=1000, cdae_forced_norm=None, ecdae_forced_norm=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGenerating {max_steps}-step continuous trajectory plots for horizons [1, 5, 10]...")

    def load_model_and_prep_data(model_dir, cfg, task_name, forced_norm):
        latest_file = get_highest_iter_file(model_dir, prefix='dae_model_', suffix='.pt')
        model_pt_path = os.path.join(model_dir, latest_file)

        checkpoint = torch.load(model_pt_path, map_location=device)
        model = get_trained_dae_model_from_pt(model_pt_path, cfg, task=task_name, dt=0.02, device=device)

        if 'dae_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['dae_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        raw_obs, raw_actions = load_and_reshape(data_path, device)
        dae_state_dim = model.state_type.size if hasattr(model, 'obs_state_type') else model.state_dim
        dae_raw_obs = raw_obs[:, :, :dae_state_dim]

        # --- STRICT STD-ONLY NORMALIZATION ---
        if forced_norm is not None or 'normalizer_state_dict' in checkpoint:
            normalizer = RunningStdScaler(cfg["robot"]["state_dim"], cfg["robot"]["action_dim"], device=device)
            if forced_norm is not None:
                normalizer.load_state_dict(forced_norm)
            else:
                normalizer.load_state_dict(checkpoint['normalizer_state_dict'])

            state_std, action_std = get_std_scales(normalizer, dae_state_dim, cfg["robot"]["action_dim"], device)
            norm_obs = dae_raw_obs / state_std
            norm_actions = raw_actions / action_std
            std_np = state_std.cpu().numpy()
            def unnorm_fn(tensor): return tensor * std_np
        else:
            state_std = torch.sqrt(torch.mean(dae_raw_obs**2, dim=(0, 1)))
            state_std[state_std < 1e-4] = 1.0
            action_std = torch.sqrt(torch.mean(raw_actions**2, dim=(0, 1)))
            action_std[action_std < 1e-4] = 1.0

            norm_obs = dae_raw_obs / state_std
            norm_actions = raw_actions / action_std
            std_np = state_std.cpu().numpy()
            def unnorm_fn(tensor): return tensor * std_np

        return model, norm_obs, norm_actions, unnorm_fn, dae_raw_obs, dae_state_dim

    model_cdae, norm_obs_cdae, norm_act_cdae, unnorm_cdae, raw_obs, dae_state_dim = load_model_and_prep_data(cdae_dir, cdae_cfg, "cdae", cdae_forced_norm)
    model_ecdae, norm_obs_ecdae, norm_act_ecdae, unnorm_ecdae, _, _ = load_model_and_prep_data(ecdae_dir, ecdae_cfg, "ecdae", ecdae_forced_norm)

    env_idx = 0
    T = min(max_steps, norm_obs_cdae.shape[1])
    # Ground truth remains strictly unnormalized for visual plotting
    y_true = raw_obs[env_idx, :T, :].cpu().numpy()

    obs_seq_c = norm_obs_cdae[env_idx, :T]
    act_seq_c = norm_act_cdae[env_idx, :T]
    obs_seq_e = norm_obs_ecdae[env_idx, :T]
    act_seq_e = norm_act_ecdae[env_idx, :T]

    identity_a = torch.eye(model_ecdae.obs_state_type.size, device=device)
    A = model_ecdae.obs_space_dynamics.transfer_op(model_ecdae.obs_state_type(identity_a)).tensor.detach().T
    identity_b = torch.eye(model_ecdae.action_type.size, device=device)
    B = model_ecdae.obs_space_dynamics.control_op(model_ecdae.action_type(identity_b)).tensor.detach().T

    bias = torch.zeros(model_ecdae.obs_state_type.size, device=device)
    if hasattr(model_ecdae.obs_space_dynamics, 'bias') and model_ecdae.obs_space_dynamics.bias is not None:
        if hasattr(model_ecdae.obs_space_dynamics.bias, 'tensor'):
            bias = model_ecdae.obs_space_dynamics.bias.tensor.detach().squeeze()
        elif isinstance(model_ecdae.obs_space_dynamics.bias, torch.Tensor):
            bias = model_ecdae.obs_space_dynamics.bias.detach().squeeze()

    A_np = A.cpu().numpy()
    eigvals, eigvecs = np.linalg.eig(A_np)
    target_idx = np.argmin(np.abs(np.abs(eigvals) - 1.0))

    eigvals_only_inv = np.zeros_like(eigvals)
    eigvals_only_inv[target_idx] = eigvals[target_idx]
    A_only_inv_np = np.real(eigvecs @ np.diag(eigvals_only_inv) @ np.linalg.inv(eigvecs))
    A_only_inv = torch.tensor(A_only_inv_np, dtype=torch.float32, device=device)

    eigvals_no_inv = eigvals.copy()
    eigvals_no_inv[target_idx] = 0.0
    A_no_inv_np = np.real(eigvecs @ np.diag(eigvals_no_inv) @ np.linalg.inv(eigvecs))
    A_no_inv = torch.tensor(A_no_inv_np, dtype=torch.float32, device=device)

    preds_cdae = {}
    preds_ecdae = {}
    preds_ecdae_only_inv = {}
    preds_ecdae_no_inv = {}
    horizons = [1, 5, 10]

    for h in horizons:
        if T <= h: continue

        s_list_c, ns_list_c, a_list_c = [], [], []
        s_list_e, ns_list_e, a_list_e = [], [], []

        for t in range(T - h):
            s_list_c.append(obs_seq_c[t])
            ns_list_c.append(obs_seq_c[t+1 : t+1+h])
            a_list_c.append(act_seq_c[t : t+h])

            s_list_e.append(obs_seq_e[t])
            ns_list_e.append(obs_seq_e[t+1 : t+1+h])
            a_list_e.append(act_seq_e[t : t+h])

        states_c = torch.stack(s_list_c)
        next_states_c = torch.stack(ns_list_c)
        actions_c = torch.stack(a_list_c)

        states_e = torch.stack(s_list_e)
        next_states_e = torch.stack(ns_list_e)
        actions_e = torch.stack(a_list_e)

        with torch.no_grad():

            # --- BASELINE CDAE ROLLOUT (Using native forecast) ---
            pred_states_c, _ = model_cdae.forecast(states_c, actions_c, n_steps=h)
            pred_c_traj = pred_states_c[:, -1, :].cpu().numpy()

            # --- BASELINE ECDAE EXPLICIT ROLLOUT (Matches Ablations perfectly) ---
            z_t_e = model_ecdae.obs_fn(model_ecdae.pre_process_state(state=states_e)).tensor
            e_obs_traj = [z_t_e]
            for t in range(h):
                u_t = actions_e[:, t, :]
                z_t_e = z_t_e @ A.T + u_t @ B.T + bias
                e_obs_traj.append(z_t_e)
            e_obs_traj = torch.stack(e_obs_traj, dim=1)
            e_obs_geom = model_ecdae.obs_state_type(batched_to_flat_trajectory(e_obs_traj))
            e_pred_state_traj = model_ecdae.post_process_state(model_ecdae.inv_obs_fn(e_obs_geom))
            e_pred_state_traj = e_pred_state_traj.view(states_e.shape[0], h + 1, -1)
            pred_e_traj = e_pred_state_traj[:, -1, :].cpu().numpy()

            # --- ONLY Invariant mode rollout ---
            z_t_only = model_ecdae.obs_fn(model_ecdae.pre_process_state(state=states_e)).tensor
            only_inv_obs_traj = [z_t_only]
            for t in range(h):
                u_t = actions_e[:, t, :]
                z_t_only = z_t_only @ A_only_inv.T + u_t @ B.T + bias
                only_inv_obs_traj.append(z_t_only)

            only_inv_obs_traj = torch.stack(only_inv_obs_traj, dim=1)
            only_inv_obs_geom = model_ecdae.obs_state_type(batched_to_flat_trajectory(only_inv_obs_traj))
            only_inv_pred_state_traj = model_ecdae.post_process_state(model_ecdae.inv_obs_fn(only_inv_obs_geom))
            only_inv_pred_state_traj = only_inv_pred_state_traj.view(states_e.shape[0], h + 1, -1)
            pred_e_only_inv_traj = only_inv_pred_state_traj[:, -1, :].cpu().numpy()

            # --- NO Invariant mode rollout ---
            z_t_no = model_ecdae.obs_fn(model_ecdae.pre_process_state(state=states_e)).tensor
            no_inv_obs_traj = [z_t_no]
            for t in range(h):
                u_t = actions_e[:, t, :]
                z_t_no = z_t_no @ A_no_inv.T + u_t @ B.T + bias
                no_inv_obs_traj.append(z_t_no)

            no_inv_obs_traj = torch.stack(no_inv_obs_traj, dim=1)
            no_inv_obs_geom = model_ecdae.obs_state_type(batched_to_flat_trajectory(no_inv_obs_traj))
            no_inv_pred_state_traj = model_ecdae.post_process_state(model_ecdae.inv_obs_fn(no_inv_obs_geom))
            no_inv_pred_state_traj = no_inv_pred_state_traj.view(states_e.shape[0], h + 1, -1)
            pred_e_no_inv_traj = no_inv_pred_state_traj[:, -1, :].cpu().numpy()

        padded_c = np.full((T, dae_state_dim), np.nan)
        padded_e = np.full((T, dae_state_dim), np.nan)
        padded_e_only_inv = np.full((T, dae_state_dim), np.nan)
        padded_e_no_inv = np.full((T, dae_state_dim), np.nan)

        padded_c[h:] = pred_c_traj
        padded_e[h:] = pred_e_traj
        padded_e_only_inv[h:] = pred_e_only_inv_traj
        padded_e_no_inv[h:] = pred_e_no_inv_traj

        preds_cdae[h] = unnorm_cdae(padded_c)
        preds_ecdae[h] = unnorm_ecdae(padded_e)
        preds_ecdae_only_inv[h] = unnorm_ecdae(padded_e_only_inv)
        preds_ecdae_no_inv[h] = unnorm_ecdae(padded_e_no_inv)

    feature_indices = [13, 14, 16, 17]
    feature_names = ["Rear Left Hip Pitch", "Rear Left Knee", "Rear Right Hip Pitch", "Rear Right Knee"]

    save_trajectory_plots(y_true, preds_cdae, preds_ecdae, preds_ecdae_only_inv, preds_ecdae_no_inv, save_dir, feature_indices, feature_names)

def print_latex_table(cdae_res, ecdae_res, cdae_res_std=None, ecdae_res_std=None):
    def fmt_lambda(val, std=None):
        if val is None: return "N/A"
        base = f"{val.real:.3f}" if abs(val.imag) < 1e-4 else f"{val.real:.3f} + {val.imag:.3f}i"
        if std is not None:
            std_str = f"{std.real:.3f}" if abs(std.imag) < 1e-4 else f"{std.real:.3f} + {std.imag:.3f}i"
            return f"${base} \\pm {std_str}$"
        return f"${base}$"

    def fmt_val(val, std=None, prefix="", suffix=""):
        if val is None: return "N/A"
        if std is not None:
            return f"${prefix}{val:.3f} \\pm {std:.3f}{suffix}$"
        return f"${prefix}{val:.3f}{suffix}$"

    c_recon = fmt_val(cdae_res['recon_mse'], cdae_res_std['recon_mse'] if cdae_res_std else None)
    c_1step = fmt_val(cdae_res['latent_1step_mse'], cdae_res_std['latent_1step_mse'] if cdae_res_std else None)
    c_5step_r = fmt_val(cdae_res['baseline_right_mse'], cdae_res_std['baseline_right_mse'] if cdae_res_std else None)
    c_5step_l = fmt_val(cdae_res['baseline_left_mse'], cdae_res_std['baseline_left_mse'] if cdae_res_std else None)

    e_recon = fmt_val(ecdae_res['recon_mse'], ecdae_res_std['recon_mse'] if ecdae_res_std else None)
    e_1step = fmt_val(ecdae_res['latent_1step_mse'], ecdae_res_std['latent_1step_mse'] if ecdae_res_std else None)
    e_5step_r = fmt_val(ecdae_res['baseline_right_mse'], ecdae_res_std['baseline_right_mse'] if ecdae_res_std else None)
    e_5step_l = fmt_val(ecdae_res['baseline_left_mse'], ecdae_res_std['baseline_left_mse'] if ecdae_res_std else None)

    e_lambda = fmt_lambda(ecdae_res['invariant_lambda'], ecdae_res_std.get('invariant_lambda') if ecdae_res_std else None)

    e_ablate_r = fmt_val(ecdae_res.get('ablated_right_mse'), ecdae_res_std.get('ablated_right_mse') if ecdae_res_std else None)
    e_ablate_l = fmt_val(ecdae_res.get('ablated_left_mse'), ecdae_res_std.get('ablated_left_mse') if ecdae_res_std else None)

    e_only_inv_r = fmt_val(ecdae_res.get('only_inv_right_mse'), ecdae_res_std.get('only_inv_right_mse') if ecdae_res_std else None)
    e_only_inv_l = fmt_val(ecdae_res.get('only_inv_left_mse'), ecdae_res_std.get('only_inv_left_mse') if ecdae_res_std else None)

    latex_str = f"""\\begin{{table}}[h]
\\vspace{{-0.25cm}}
\\centering
\\caption{{MSE comparison of cDAE and the symmetry-constrained ecDAE for \\emph{{push door}}, with ablation of invariant mode, averaged over seeds.}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{cDAE}} & \\textbf{{ecDAE}}\\\\
\\midrule
\\textbf{{$\\mathcal L_{{sr}}$ MSE}} & {c_recon} & \\textbf{{{e_recon}}} \\\\
\\textbf{{1-Step $\\mathcal L_{{lp}}$ MSE}}  & \\textbf{{{c_1step}}} & {e_1step} \\\\
\\textbf{{5-Step $\\mathcal L_{{sp}}$ MSE (Right)}} & {c_5step_r} & \\textbf{{{e_5step_r}}} \\\\
\\textbf{{5-Step $\\mathcal L_{{sp}}$ MSE (Left)}}  & {c_5step_l} & \\textbf{{{e_5step_l}}} \\\\
\\textbf{{Invariant $\\lambda$}} & N/A & \\textbf{{{e_lambda}}} \\\\
\\midrule
\\multicolumn{{3}}{{c}}{{\\textbf{{Ablation}}}}\\\\
\\midrule
\\textbf{{No Inv $|\\lambda|$ $\\mathcal L_{{sp}}$ MSE (Right)}}& N/A & {e_ablate_r} \\\\
\\textbf{{No Inv $|\\lambda|$ $\\mathcal L_{{sp}}$ MSE (Left)}} & N/A & {e_ablate_l} \\\\
\\textbf{{Only Inv $|\\lambda|$ $\\mathcal L_{{sp}}$ MSE (Right)}}& N/A & {e_only_inv_r} \\\\
\\textbf{{Only Inv $|\\lambda|$ $\\mathcal L_{{sp}}$ MSE (Left)}} & N/A & {e_only_inv_l} \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:dae_comparison}}
\\vspace{{-0.5cm}}
\\end{{table}}
"""
    print("\n" + "="*80)
    print(" GENERATED LATEX TABLE ")
    print("="*80)
    print(latex_str)

def get_reference_normalizer_dict(base_dir, reference_string):
    matching_dirs = [d for d in os.listdir(base_dir) if reference_string in d]
    if not matching_dirs:
        raise ValueError(f"Could not find a directory containing '{reference_string}' in {base_dir}")

    ref_dir = os.path.join(base_dir, matching_dirs[0])
    latest_pt = get_highest_iter_file(ref_dir, prefix="dae_model_", suffix=".pt")
    if not latest_pt:
        raise FileNotFoundError(f"Could not find a .pt file in {ref_dir}")

    checkpoint = torch.load(os.path.join(ref_dir, latest_pt), map_location="cpu")
    if 'normalizer_state_dict' not in checkpoint:
        raise KeyError(f"No 'normalizer_state_dict' found in {latest_pt}")

    return checkpoint['normalizer_state_dict']

if __name__ == "__main__":

    home_dir = os.path.expanduser("~")
    cdae_task = "push_door_cyber_cdae_online_next_latent"
    ecdae_task = "push_door_cyber_emlp_ecdae_online_next_latent"

    cdae_base_dir = f"{home_dir}/git/koopman_symmloco/legged_gym/logs/{cdae_task}/"
    ecdae_base_dir = f"{home_dir}/git/koopman_symmloco/legged_gym/logs/{ecdae_task}/"

    # --- EXTRACT THE GLOBAL REFERENCE NORMALIZERS ---
    print("Extracting global reference normalizers from 2026-03-03-16 runs...")
    ref_norm_cdae = get_reference_normalizer_dict(cdae_base_dir, "2026-03-03-16")
    ref_norm_ecdae = get_reference_normalizer_dict(ecdae_base_dir, "2026-03-03-16")

    cdae_model_dirs = sorted([d for d in os.listdir(cdae_base_dir) if "202" in d and not d.startswith("2026-03-03")])
    ecdae_model_dirs = sorted([d for d in os.listdir(ecdae_base_dir) if "202" in d and not d.startswith("2026-03-03")])

    cdae_model_cfg = {"name": 'cdae', "equivariant": False, "activation": 'ELU', "num_layers": 5, "num_hidden_units": 128, "batch_norm": False, "obs_pred_w": 1.0, "orth_w": 0.0, "corr_w": 0.0, "bias": True, "constant_function": True, "num_mini_batches": 8, "mini_batch_size": 256, "beta_initial": 0.4, "beta_annealing_steps": 20000}
    ecdae_model_cfg = {"name": 'ecdae', "equivariant": True, "activation": 'ELU', "num_layers": 5, "num_hidden_units": 128, "batch_norm": False, "obs_pred_w": 1.0, "orth_w": 0.0, "corr_w": 0.0, "bias": True, "constant_function": True, "num_mini_batches": 8, "mini_batch_size": 256, "beta_initial": 0.4, "beta_annealing_steps": 20000, "group_avg_trick": True, "state_dependent_obs_dyn": False}
    robot_cfg = {"name": "a1", "lr": 1e-3, "max_epochs": 200, "obs_state_ratio": 3, "state_obs": ['projected_gravity', 'projected_forward_vec', 'joint_pos', 'prev_actions', 'phase_input', 'base_pos', 'door_bottom_corner_pos', 'door_normal_vec', 'lr_vec'], "action_obs": ['actions'], "state_dim": 3+3+12+12+2+3+3+3+2, "action_dim": 12, "pred_horizon": 5, "frames_per_state": 1}

    res_cdae_agg = {"recon_mse": [], "latent_1step_mse": [], "baseline_right_mse": [], "baseline_left_mse": []}
    res_ecdae_agg = {"recon_mse": [], "latent_1step_mse": [], "baseline_right_mse": [], "baseline_left_mse": [], "invariant_lambda": [], "ablated_left_mse": [], "ablated_right_mse": [], "only_inv_left_mse": [], "only_inv_right_mse": []}

    plots_generated = False

    for i, (cdae_seed, ecdae_seed) in enumerate(zip(cdae_model_dirs, ecdae_model_dirs)):

        cdae_model_path = os.path.join(cdae_base_dir, cdae_seed)
        ecdae_model_path = os.path.join(ecdae_base_dir, ecdae_seed)

        cdae_data_dir = f"{home_dir}/git/koopman_symmloco/legged_gym/isaacgym_recordings/{cdae_task}/{cdae_seed}"
        ecdae_data_dir = f"{home_dir}/git/koopman_symmloco/legged_gym/isaacgym_recordings/{ecdae_task}/{ecdae_seed}"

        cdae_data_file = get_highest_iter_file(cdae_data_dir, suffix="_obs_action.npy")
        ecdae_data_file = get_highest_iter_file(ecdae_data_dir, suffix="_obs_action.npy")
        ecdae_left_data_file = get_highest_iter_file(ecdae_data_dir, suffix="_obs_action_left.npy")

        if not cdae_data_file or not ecdae_data_file or not ecdae_left_data_file:
            print(f"\nSkipping seed {cdae_seed} / {ecdae_seed} -> Missing required .npy files.")
            continue

        cdae_data_path = os.path.join(cdae_data_dir, cdae_data_file)
        ecdae_data_path = os.path.join(ecdae_data_dir, ecdae_data_file)
        ecdae_left_data_path = os.path.join(ecdae_data_dir, ecdae_left_data_file)

        if not plots_generated:
            plot_save_dir = f"{home_dir}/git/koopman_symmloco/legged_gym/logs/trajectory_plots"
            generate_comparison_plots(
                cdae_dir=cdae_model_path,
                ecdae_dir=ecdae_model_path,
                data_path=ecdae_data_path,
                cdae_cfg={"model": cdae_model_cfg, "robot": robot_cfg},
                ecdae_cfg={"model": ecdae_model_cfg, "robot": robot_cfg},
                save_dir=plot_save_dir,
                max_steps=1000,
                cdae_forced_norm=ref_norm_cdae,
                ecdae_forced_norm=ref_norm_ecdae
            )
            plots_generated = True

        res_cdae = evaluate_and_extract_metrics(cdae_model_path, cdae_data_path, ecdae_left_data_path, {"model": cdae_model_cfg, "robot": robot_cfg}, "push_door", "cDAE", 5, forced_normalizer_dict=ref_norm_cdae)
        res_ecdae = evaluate_and_extract_metrics(ecdae_model_path, ecdae_data_path, ecdae_left_data_path, {"model": ecdae_model_cfg, "robot": robot_cfg}, "push_door", "ecDAE", 5, forced_normalizer_dict=ref_norm_ecdae)

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
        res_ecdae_agg["only_inv_left_mse"].append(res_ecdae['only_inv_left_mse'])
        res_ecdae_agg["only_inv_right_mse"].append(res_ecdae['only_inv_right_mse'])

    if len(res_cdae_agg["recon_mse"]) > 0:
        res_cdae_agg_std = {k: np.std(v) for k, v in res_cdae_agg.items()}
        res_ecdae_agg_std = {k: np.std(v) for k, v in res_ecdae_agg.items()}

        res_cdae_agg = {k: np.mean(v) for k, v in res_cdae_agg.items()}
        res_ecdae_agg = {k: np.mean(v) for k, v in res_ecdae_agg.items()}

        print_latex_table(res_cdae_agg, res_ecdae_agg, res_cdae_agg_std, res_ecdae_agg_std)