# This version is modified for your case:
# - 35-dimensional observation
# - DAE-aug model (no action used)
# - Not using IMU task

import torch
import os
import re
import math
import numpy as np
import dha
from dha.utils.mysc import class_from_name
from dha.nn.DynamicsAutoEncoder import DAE
from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.ControlledDynamicsAutoEncoder import ControlledDAE
from dha.nn.ControlledEquivDynamicsAutoencoder import ControlledEquivDAE
from morpho_symm.utils.robot_utils import load_symmetric_system
from hydra import initialize, compose
from morpho_symm.nn.test_EMLP import get_kinematic_three_rep_two
import escnn
from escnn.nn import FieldType
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
from typing import Union
import legged_gym
import glob

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
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}

def load_normalization_stats(model_path: str, device: torch.device):
    """
    从给定 model_path 下加载 state_mean_var.npy 文件，返回 PyTorch 格式的 mean 和 std。
    """
    norm_path = os.path.join(model_path, "state_mean_var.npy")
    if not os.path.exists(norm_path):
        print(f"[Warning] Normalization file not found at: {norm_path}")
        # fallback to default
        return torch.zeros(35, device=device), torch.ones(35, device=device)

    norm_data = np.load(norm_path, allow_pickle=True).item()
    state_mean = torch.tensor(norm_data["state_mean"], device=device).float()
    state_var = torch.tensor(norm_data["state_var"], device=device).float()
    state_std = torch.sqrt(state_var)
    return state_mean, state_std

def safe_standardize(x_normed: Union[torch.Tensor, np.ndarray], mean: Union[torch.Tensor, np.ndarray], std: Union[torch.Tensor, np.ndarray]):
    mask = std > 0
    if isinstance(x_normed, torch.Tensor):
        x_normed = x_normed.clone()
    if x_normed.ndim == 2:
        x_normed[:, mask] = (x_normed[:, mask] - mean[mask]) / std[mask]
    elif x_normed.ndim == 3:
        x_normed[:, :, mask] = (x_normed[:, :, mask] - mean[mask]) / std[mask]
    return x_normed

def get_trained_dae_model_from_pt(model_path: str, cfg: dict, task: str, dt: float = 0.02, device: torch.device = torch.device('cpu')):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # 1. Initialize the correct model architecture purely from the cfg
    model = initialize_dae_model(cfg, task, dt, device)

    # 2. Load the saved weights from the .pt file
    loaded_data = torch.load(model_path, map_location=device)

    if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
        state_dict = loaded_data['state_dict']
    elif isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
        state_dict = loaded_data['model_state_dict']
    else:
        state_dict = loaded_data

    cleaned_state_dict = remove_state_dict_prefix(state_dict, "model.")

    # 3. Handle ESCNN exports
    is_exported = any(k.endswith(".weight") for k in cleaned_state_dict.keys() if "obs_fn" in k)

    if is_exported and ("e" in task.lower()):
        print("[Info] Detected an exported escnn checkpoint. Exporting internal modules...")

        def export_escnn_modules(mod):
            mod.eval()
            for name, child in mod.named_children():
                if hasattr(child, 'export') and callable(child.export):
                    try:
                        setattr(mod, name, child.export())
                        continue
                    except Exception:
                        pass
                export_escnn_modules(child)

        export_escnn_modules(model)

    # 4. Smart Key Mapping for escnn formatting differences
    def normalize_key(key):
        """Removes escnn suffixes and normalizes module paths to match the checkpoint format."""
        # Strip the escnn dimension suffixes like ": in=43-out=128"
        key = re.sub(r': in=\d+-out=\d+', '', key)
        # Re-map net_head to net.head
        key = key.replace('net_head.', 'net.head.')
        return key

    # Create a lookup dictionary of the loaded weights using the normalized keys
    normalized_loaded_dict = {normalize_key(k): v for k, v in cleaned_state_dict.items()}

    # Build a new state dict that strictly uses the keys the current model expects
    final_state_dict = {}
    expected_keys = model.state_dict().keys()

    for expected_key in expected_keys:
        norm_key = normalize_key(expected_key)
        if norm_key in normalized_loaded_dict:
            final_state_dict[expected_key] = normalized_loaded_dict[norm_key]

    # 5. Load the aligned weights
    # strict=False is used safely here to ignore the missing `act_0.A` / `act_0.Ainv` mathematical buffers
    model.load_state_dict(final_state_dict, strict=False)

    return model

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
    if "push_door" in model_dir:
        state_reps = [rep_Rd, rep_Rd, rep_TqQ_js, rep_TqQ_js, rep_kin_three, rep_Rd, rep_Rd, rep_Rd, rep_kin_three]  #['projected_gravity', 'projected_forward_vec', 'joint_pos', 'prev_actions', 'phase_input', 'base_pos', 'door_bottom_corner_pos', 'door_normal_vec', 'lr_vec']
        state_type = FieldType(gspace, representations=state_reps)
        state_type.size = sum(rep.size for rep in state_reps) + 4 * rep_Rd.size + rep_TqQ_js.size + rep_kin_three.size # Count duplicates twice
    else:
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

def initialize_dae_model(cfg, task: str, dt: int, device: torch.device) -> torch.nn.Module:
    """
    Initializes a Koopman model based on the provided configuration (from train_cfg.koopman_model).
    Can also load pre-trained weights if cfg.load_path is specified.

    Args:
        cfg: The Koopman model configuration object from train_cfg.
        task (str): The DAE task string.
        dt (float): The environment's delta time (from environment).
        device (torch.device): The torch device (e.g., 'cuda:0', 'cpu').

    Returns:
        torch.nn.Module: An initialized Koopman model (new or loaded).
    """

    # --- START FIX: Temporarily force CPU instantiation ---
    # IsaacGym globally sets default tensors to CUDA, which breaks escnn's
    # internal CPU-based geometric basis caching. We force standard CPU tensors here.
    original_tensor_type = torch.tensor([]).type()
    torch.set_default_tensor_type('torch.FloatTensor')

    try:
        # Define the state representation
        # G is the symmetry group of the system
        robot, G = load_symmetric_system(robot_name=cfg["robot"]["name"])

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

        # Create dict to define which obs match which representations
        obs_rep_dict = {
            'projected_gravity': rep_Rd,
            'projected_forward_vec': rep_Rd,
            'xy_commands': rep_xy,
            'z_commands': rep_euler_z,
            'joint_pos': rep_TqQ_js,
            'joint_vel': rep_TqQ_js,
            'prev_actions': rep_TqQ_js,
            'clock_inputs': rep_kin_three,
            'phase_input': rep_kin_three,
            'base_pos': rep_Rd,
            'door_bottom_corner_pos': rep_Rd,
            'door_normal_vec': rep_Rd,
            'lr_vec': rep_kin_three,
            'actions': rep_TqQ_js,
        }

        state_reps = []
        action_reps = []
        for state_obs in cfg["robot"]["state_obs"]:
            if state_obs in obs_rep_dict:
                state_reps.append(obs_rep_dict[state_obs])
            else:
                raise ValueError(f"Observation '{state_obs}' not found in the defined representations.")
        for action_obs in cfg["robot"]["action_obs"]:
            if action_obs in obs_rep_dict:
                action_reps.append(obs_rep_dict[action_obs])
            else:
                raise ValueError(f"Action '{action_obs}' not found in the defined representations.")

        state_type = FieldType(gspace, representations=state_reps)
        action_type = FieldType(gspace, representations=action_reps)

        state_dim = cfg["robot"]["state_dim"]
        action_dim = cfg["robot"]["action_dim"]

        # Ensure that with duplicate reps the size matches the expected dimensions
        state_type.size = state_dim
        action_type.size = action_dim

        obs_state_dim = math.ceil(cfg["robot"]["obs_state_ratio"] * state_dim)
        num_hidden_neurons = cfg["model"]["num_hidden_units"]
        if obs_state_dim > num_hidden_neurons:
            num_hidden_neurons = 2 ** math.ceil(math.log2(obs_state_dim))

        activation = cfg["model"]["activation"]

        if not cfg["model"]["equivariant"]:
            activation = class_from_name("torch.nn", activation)

        obs_fn_params = {'num_layers': cfg["model"]["num_layers"], 'num_hidden_units': cfg["model"]["num_hidden_units"], 'activation': activation, 'bias': cfg["model"]["bias"], 'batch_norm': cfg["model"]["batch_norm"]}

        initial_rng_state = torch.get_rng_state()

        if "edae" in task:
            model = EquivDAE(
                state_rep=state_type.representation,
                obs_state_dim=obs_state_dim,
                dt=dt,
                orth_w=cfg["model"]["orth_w"],
                obs_fn_params=obs_fn_params,
                group_avg_trick=cfg["model"]["group_avg_trick"],
                state_dependent_obs_dyn=cfg["model"]["state_dependent_obs_dyn"],
                enforce_constant_fn=cfg["model"]["constant_function"],
            )
        elif "ecdae" in task:
            model = ControlledEquivDAE(
                state_rep=state_type.representation,
                action_rep=action_type.representation,
                obs_state_dim=obs_state_dim,
                dt=dt,
                orth_w=cfg["model"]["orth_w"],
                obs_fn_params=obs_fn_params,
                group_avg_trick=cfg["model"]["group_avg_trick"],
                state_dependent_obs_dyn=cfg["model"]["state_dependent_obs_dyn"],
                enforce_constant_fn=cfg["model"]["constant_function"],
            )
        elif "cdae" in task:
            model = ControlledDAE(
                state_dim=state_dim,
                action_dim=action_dim,
                obs_state_dim=obs_state_dim,
                dt=dt,
                orth_w=cfg["model"]["orth_w"],
                obs_fn_params=obs_fn_params,
                enforce_constant_fn=cfg["model"]["constant_function"],
            )
        elif "dae" in task:
            model = DAE(
                state_dim=state_dim,
                obs_state_dim=obs_state_dim,
                dt=dt,
                obs_pred_w=cfg["model"]["obs_pred_w"],
                orth_w=cfg["model"]["orth_w"],
                obs_fn_params=obs_fn_params,
                enforce_constant_fn=cfg["model"]["constant_function"],
            )
        else:
            raise ValueError(f"Trying to create DAE model with unsupported task: {task}")

        torch.set_rng_state(initial_rng_state)

    finally:
        # --- END FIX: Safely restore whatever tensor type IsaacGym was using ---
        torch.set_default_tensor_type(original_tensor_type)

    # Put the fully constructed model on the specified device
    model.to(device)

    return model

def main():

    import matplotlib.pyplot as plt

    # Use LaTeX for text rendering for compatibility with papercept/IEEE
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # base experiments logs path
    logs_base_path = "/home/iit.local/edelia/git/koopman_symmloco/legged_gym/logs/"

    # List of base model directories to scan
    base_experiment_dirs = [
        logs_base_path + "stand_dance_cyber_cdae_online_next_latent/",
        logs_base_path + "stand_dance_cyber_emlp_ecdae_online_next_latent/",
        logs_base_path + "walk_slope_cyber_cdae_online_next_latent/",
        logs_base_path + "walk_slope_cyber_emlp_ecdae_online_next_latent/",
        logs_base_path + "push_door_cyber_cdae_online_next_latent/",
        logs_base_path + "push_door_cyber_emlp_ecdae_online_next_latent/"
    ]

    # Data structure to collect results for table averaging and last-run plotting
    collected_results = {}
    unique_types = []

    dha_dir = os.path.dirname(dha.__file__)

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
            "state_obs": ['projected_gravity', 'projected_forward_vec', 'xy_commands', 'z_commands', 'joint_pos', 'joint_vel', 'prev_actions', 'clock_inputs'],
            "action_obs": ['actions'],
            "state_dim": 3 + 3 + 3 + 12 + 12 + 12 + 2,
            "action_dim": 12,
            "pred_horizon": 5,
            "frames_per_state": 1,
    }

    push_door_robot_cfg = {
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

    push_door_cdae_koopman_cfg = {
        "model": cdae_model_cfg,
        "robot": push_door_robot_cfg,
    }

    push_door_ecdae_koopman_cfg = {
        "model": ecdae_model_cfg,
        "robot": push_door_robot_cfg,
    }

    # Step 1: Scan and collect results from all found run subdirectories
    for base_dir in base_experiment_dirs:
        if not os.path.exists(base_dir):
            print(f"[Warning] Base directory not found, skipping: {base_dir}")
            continue

        # Find task and type from base_dir name using regex
        match = re.search(r"logs/(.*?)_cyber", base_dir)
        task_name = match.group(1) if match else "unknown"
        dae_type = "ecDAE" if "ecdae" in base_dir else "cDAE"
        key = (task_name, dae_type)

        if dae_type not in unique_types: unique_types.append(dae_type)
        if key not in collected_results:
            collected_results[key] = { 'max_eig': [], 'cond_num': [], 'rank': [], 'state_dim': [], 'runs': [] }

        # Get all run subdirectories and sort them alphabetically
        run_subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if not run_subdirs:
            print(f"[Warning] No run subdirectories found in {base_dir}")
            continue

        for run_subdir in run_subdirs:
            run_dir_path = os.path.join(base_dir, run_subdir)

            # Find the dae_model_*.pt file. User's example has particular numbers like 30000.pt.
            # I will search for all files matching dae_model_*.pt and take the one with highest iteration number.
            model_files = glob.glob(os.path.join(run_dir_path, "dae_model_*.pt"))
            if not model_files:
                print(f"[Warning] No dae_model_*.pt file found in {run_dir_path}")
                continue

            # Logic to find "latest" model file with highest iteration
            latest_model_path = sorted(model_files, key=lambda f: int(re.search(r'dae_model_(\d+).pt', f).group(1)), reverse=True)[0]

            print(f"processing model in {latest_model_path} for averaging table...")
            koopman_cfg = ecdae_koopman_cfg if dae_type == "ecDAE" else cdae_koopman_cfg
            if "push_door" in latest_model_path:
                koopman_cfg = push_door_ecdae_koopman_cfg if dae_type == "ecDAE" else push_door_cdae_koopman_cfg

            try:
                model = get_trained_dae_model_from_pt(latest_model_path, koopman_cfg, task=dae_type.lower(), dt=0.02)
            except Exception as e:
                print(f"[Error] Failed to load model {latest_model_path}: {e}")
                continue

            # Reuse user's logic to extract matrices, eigenvalues, singular values, and rank
            if "ecdae" in latest_model_path:
                state_dim = model.obs_state_type.size
                action_dim = model.action_type.size
                identity_a = torch.eye(state_dim)
                input_tensor_a = model.obs_state_type(identity_a)
                a_matrix = model.obs_space_dynamics.transfer_op(input_tensor_a).tensor.detach().T
                identity_b = torch.eye(action_dim)
                input_tensor_b = model.action_type(identity_b)
                b_matrix = model.obs_space_dynamics.control_op(input_tensor_b).tensor.detach().T
            else:
                a_matrix = model.obs_space_dynamics.transfer_op.weight.detach()
                b_matrix = model.obs_space_dynamics.control_op.weight.detach()
                state_dim = a_matrix.shape[0]

            # Calculate eigenvalues of A
            eigvals = np.linalg.eigvals(a_matrix.numpy())
            max_eig_mag = np.abs(eigvals).max()
            collected_results[key]['max_eig'].append(max_eig_mag)

            # Controllability matrix and singular values
            controllability_matrix = b_matrix
            term = b_matrix
            for i in range(1, state_dim):
                term = a_matrix.T @ term
                controllability_matrix = torch.cat((controllability_matrix, term), dim=1)
            rank_c_val = torch.linalg.matrix_rank(controllability_matrix).item()
            svs_tensor = torch.linalg.svdvals(controllability_matrix)
            cond_number_val = (svs_tensor.max() / svs_tensor.min()).item()

            # Collect results for averaging table
            collected_results[key]['cond_num'].append(cond_number_val)
            collected_results[key]['rank'].append(rank_c_val)
            collected_results[key]['state_dim'].append(state_dim)

            # Collect details for last-run plotting
            collected_results[key]['runs'].append((run_dir_path, latest_model_path, a_matrix, b_matrix, eigvals, svs_tensor, rank_c_val, state_dim))

    # Explicit ordering for tasks
    ordered_tasks = ["stand_dance", "walk_slope", "push_door"]
    sorted_unique_types = sorted(unique_types)

    # Dictionary to format LaTeX safe printouts
    task_display_names = {
        "stand_dance": "Stand Dance",
        "walk_slope": "Walk Slope",
        "push_door": "Push Door",
        "unknown": "Unknown"
    }

    # Step 2: Print table with averaged values and standard deviations over found seeds
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l l c c c}")
    print("\\hline")
    # Added bolding to the title row headers
    print("\\textbf{Task} & \\textbf{DAE Type} & \\textbf{Avg Max} $|\\lambda(A)|$ & \\textbf{Avg Cond Number} $\\sigma(\\mathcal C)$ & \\textbf{Avg} $\\text{rank}(\\mathcal C)$ \\\\")
    print("\\hline")

    for task_name in ordered_tasks:
        for j, dae_type in enumerate(sorted_unique_types):
            key = (task_name, dae_type)
            if key in collected_results:
                res = collected_results[key]
                if res['runs']: # ensure some seeds were actually found
                    avg_max_eig = np.mean(res['max_eig'])
                    std_max_eig = np.std(res['max_eig'])

                    avg_cond_num = np.mean(res['cond_num'])
                    std_cond_num = np.std(res['cond_num'])

                    # Format Condition number with shared exponent
                    cond_mean_str = f"{avg_cond_num:.2e}"
                    cond_mean_man, cond_exp = cond_mean_str.split('e')
                    cond_std_man = std_cond_num / (10**int(cond_exp))
                    cond_str = f"({float(cond_mean_man):.2f} $\\pm$ {cond_std_man:.2f})e{cond_exp}"

                    avg_rank = np.mean(res['rank'])
                    std_rank = np.std(res['rank'])

                    # Assumption: state_dim is the same for all seeds of a task/type combo. Use the first one.
                    state_dim_for_label = res['state_dim'][0]
                    num_seeds = len(res['runs'])

                    # Ensure no underscores make it to LaTeX and only print bolded task on the first iteration (j=0)
                    pretty_task_name = task_display_names.get(task_name, task_name)
                    display_task = f"\\textbf{{{pretty_task_name}}}" if j == 0 else ""

                    # Formatted strings, with numbers exposed so they can be easily wrapped with \textbf{} if desired in the future
                    print(f"{display_task} & {dae_type} & {avg_max_eig:.4f} $\\pm$ {std_max_eig:.4f} & {cond_str} & {avg_rank:.1f} $\\pm$ {std_rank:.1f}/{state_dim_for_label} \\\\")
                else:
                    print(f"[Error] Data collected for key {key} but no runs stored?")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Average summary of $A$ matrix eigenvalues and controllability matrix singular values for each model over multiple seeds found in subdirectories.}")
    print("\\label{tab:a_eig_sv_summary_averaged}")
    print("\\end{table}")


    # Step 3: Create plots using data ONLY from the last (alphabetical) seed subdir for each model category
    n_rows = len(ordered_tasks)
    n_cols = len(sorted_unique_types)

    # Re-using user's subplot arrangement
    fig_a, axs_a = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 2.5*n_rows))
    fig_b, axs_b = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 2.5*n_rows))
    fig_eig, axs_eig = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows))
    fig_eig_mag, axs_eig_mag = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows))
    fig_sv, axs_sv = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows))

    # Helper function to get correct axis for any subplot grid shape
    def get_ax(axs, i, j):
        if n_rows == 1 and n_cols == 1:
            return axs
        elif n_rows == 1:
            return axs[j]
        elif n_cols == 1:
            return axs[i]
        else:
            return axs[i, j]

    # Helper dict for LaTeX task names
    task_latex = {
        "stand_dance": "stand dance",
        "walk_slope": "walk slope",
        "push_door": "push door",
        "unknown": "unknown"
    }

    # Iterate unique tasks and types for plotting
    for i, task_name in enumerate(ordered_tasks):
        for j, dae_type in enumerate(sorted_unique_types):
            key = (task_name, dae_type)
            if key in collected_results and collected_results[key]['runs']:
                # Identify data for the last run subdirectory. 'runs' contains detail tuples. Alphabetical sorting was done in Step 1 for run_subdirs.
                last_run_tuple = collected_results[key]['runs'][-1]
                run_dir_plot, model_file_plot, a_mat_plot, b_mat_plot, eigvals_plot, svs_tensor_plot, rank_c_plot, state_dim_plot = last_run_tuple
                print(f"[Plotting] Generating plots for last alphabetical run in: {model_file_plot}")

                # Use original plotting code, but with variables from last_run_tuple and updated titles
                # Plots are unchanged visually, but now are done for the correct seed.

                # A matrix
                ax = get_ax(axs_a, i, j)
                im = ax.imshow(a_mat_plot.numpy(), cmap='viridis', interpolation='nearest', aspect='equal')
                # Updated title to include "Last Seed" information
                ax.set_title(rf"{dae_type} $A$ for {task_latex.get(task_name, task_name)}", fontsize=13)
                if j == 0:
                    ax.set_ylabel("Output dim")
                if i == n_rows - 1:
                    ax.set_xlabel("Input dim")
                fig_a.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # B matrix
                ax = get_ax(axs_b, i, j)
                im = ax.imshow(b_mat_plot.numpy(), cmap='viridis', interpolation='nearest', aspect='equal')
                # Updated title
                ax.set_title(rf"{dae_type} $B$ for {task_latex.get(task_name, task_name)}", fontsize=13)
                if j == 0:
                    ax.set_ylabel("Output dim")
                if i == n_rows - 1:
                    ax.set_xlabel("Input dim")
                fig_b.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Eigenvalues
                ax = get_ax(axs_eig, i, j)
                ax.scatter(eigvals_plot.real, eigvals_plot.imag, color='blue', marker='+')
                # Updated title
                ax.set_title(rf"{dae_type} $\lambda(A)$ for {task_latex.get(task_name, task_name)}", fontsize=13)
                if j == 0:
                    ax.set_ylabel("Imag")
                if i == n_rows - 1:
                    ax.set_xlabel("Real")
                # Red dashed unit circle logic unchanged
                theta_circle = np.linspace(0, 2 * np.pi, 400)
                ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'r--')
                ax.axhline(0, color='gray', linewidth=0.5)
                ax.axvline(0, color='gray', linewidth=0.5)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_aspect('equal')

                # Eigenvalue magnitudes
                ax = get_ax(axs_eig_mag, i, j)
                # Calculating magnitude and sorting descending for descending plot
                mags_plot = np.abs(eigvals_plot)
                sorted_mags_plot = np.sort(mags_plot)[::-1]
                ax.plot(sorted_mags_plot, 'o-', color='purple', label='Magnitude')
                # Red dashed line for stability border
                ax.axhline(y=1.0, color='r', linestyle='--', label='$|\lambda| = 1$')
                # Updated title
                ax.set_title(rf"{dae_type} $|\lambda(A)|$ for {task_latex.get(task_name, task_name)}", fontsize=13)
                if j == 0:
                    ax.set_ylabel("Magnitude")
                if i == n_rows - 1:
                    ax.set_xlabel("Index")
                ax.grid(True, linestyle='--', alpha=0.5)
                if i == 0 and j == 0:
                    ax.legend(title=None, loc='lower left')

                # Controllability Singular values
                ax = get_ax(axs_sv, i, j)
                ax.semilogy(svs_tensor_plot.detach().numpy(), 'o-', label='Singular Values')
                # Updated title to show specific rank and state dim for that last seed
                ax.set_title(rf"{dae_type} $\sigma(C)$ for {task_latex.get(task_name, task_name)}", fontsize=13)
                if j == 0:
                    ax.set_ylabel('SV Magnitude (log)')
                if i == n_rows - 1:
                    ax.set_xlabel('SV Index')
                ax.grid(True, which="both", linestyle='--')
                # Numeric zero line unchanged
                ax.axhline(y=1e-8, color='r', linestyle='--', label='Num. Zero')
                ax.legend()


    # Step 4: Final logic for y-limits on singular values and saving all figures. Unchanged logic but now uses data from last seeds.
    # Set the same y-limits for all controllability SV plots to allow visual comparison
    all_svs_flat = torch.cat([collected_results[key]['runs'][-1][5].detach() for key in collected_results if collected_results[key]['runs']])
    min_sv_global = all_svs_flat.min().item()
    max_sv_global = all_svs_flat.max().item()
    for i in range(n_rows):
        for j in range(n_cols):
            ax = get_ax(axs_sv, i, j)
            ax.set_ylim([min_sv_global, max_sv_global])

    # Save all figures with descriptive filenames indicating averages table and last-seed plots are generated.
    fig_a.tight_layout()
    fig_b.tight_layout()
    fig_eig.tight_layout()
    fig_eig_mag.tight_layout()
    fig_sv.tight_layout()
    fig_a.savefig("all_a_matrices_last_seed.pdf", bbox_inches='tight', pad_inches=0.01)
    fig_b.savefig("all_b_matrices_last_seed.pdf", bbox_inches='tight', pad_inches=0.01)
    fig_eig.savefig("all_a_eigenvalues_last_seed.pdf", bbox_inches='tight', pad_inches=0.01)
    fig_eig_mag.savefig("all_a_eigenvalue_magnitudes_last_seed.pdf", bbox_inches='tight', pad_inches=0.01)
    fig_sv.savefig("all_controllability_svs_last_seed.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close(fig_a)
    plt.close(fig_b)
    plt.close(fig_eig)
    plt.close(fig_eig_mag)
    plt.close(fig_sv)

    print("Processing complete. LaTeX table printed and PDFs saved.")

if __name__ == "__main__":
    main()