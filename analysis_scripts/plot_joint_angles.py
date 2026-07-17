import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from legged_gym import LEGGED_GYM_ROOT_DIR

plt.rcParams.update({'font.family':'serif'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Plot joint angle trajectories.")
parser.add_argument('--offset-right', action='store_true',
                    help='Offset the right trajectory to align the phase with the left side.')
args = parser.parse_args()

# --- Configuration ---
BASE_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "isaacgym_recordings")
NUM_ENVS = 32
ENV_ID_TO_PLOT = 0
MIN_TIMESTEPS = 0  # Minimum timesteps to consider for evaluation
MAX_TIMESTEPS = 500  # Increased to 1000 to accommodate the second window
DOF_POS_SCALE = 1.0

# --- Offset Math ---
STEP_FREQ_HZ = 2.5
TIMESTEP_S = 4 * 0.005
# Left and right are typically 180 degrees (half a period) out of phase
HALF_PERIOD_S = (1.0 / STEP_FREQ_HZ) / 2.0
OFFSET_STEPS = int(HALF_PERIOD_S / TIMESTEP_S) if args.offset_right else 0

if args.offset_right:
    print(f"[*] Applying right-side offset of {OFFSET_STEPS} timesteps (Half-period: {HALF_PERIOD_S}s, dt: {TIMESTEP_S}s)\n")

# Paths based on hf_datasets/inference_data with specific dates to plot
HF_DATASETS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "../hf_datasets/inference_data/joint_angle_data")
TARGET_DATES = ["2026-02-01-20-59-56_", "2026-02-20-08-00-08_", "2026-02-21-08-59-03_", "2026-02-24-11-03-28_"]

def find_npy_files(base_dir, target_dates):
    """Find all .npy files in stand_dance* directories matching target dates."""
    methods_to_plot = {}
    
    if not os.path.exists(base_dir):
        print(f"[!] Directory not found: {base_dir}")
        return methods_to_plot
    
    # Scan for stand_dance* directories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path) or not item.startswith("stand_dance"):
            continue
        
        # Scan for 202* subdirectories
        for subitem in os.listdir(item_path):
            subitem_path = os.path.join(item_path, subitem)
            if not os.path.isdir(subitem_path) or not subitem.startswith("202"):
                continue
            
            # Check if this date matches our target dates
            date_match = None
            for target_date in target_dates:
                if subitem.startswith(target_date):
                    date_match = target_date
                    break
            
            if date_match:
                # Find .npy files in this directory
                for fname in os.listdir(subitem_path):
                    if fname.endswith("obs_action.npy"):
                        rel_path = os.path.relpath(os.path.join(subitem_path, fname), base_dir)
                        # Determine method name based on directory structure
                        if "emlp_ecdae" in item:
                            method = "SKooP"
                        elif "emlp" in item and "ecdae" not in item:
                            method = "PPOeqic"
                        elif "cdae_online" in item:
                            method = "SKooP-NoSym"
                        elif "cyber" in item:
                            method = "PPO"
                        else:
                            method = "Other"
                        
                        if method not in methods_to_plot:
                            methods_to_plot[method] = []
                        methods_to_plot[method].append(rel_path)
    
    return methods_to_plot

methods_to_plot = find_npy_files(HF_DATASETS_DIR, TARGET_DATES)
if not methods_to_plot:
    print("[!] No matching .npy files found. Falling back to hardcoded paths.")
    BASE_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "isaacgym_recordings")
    methods_to_plot = {
        "PPO": ["stand_dance_cyber/2026-02-01-20-59-56_/model_30000_obs_action.npy"],
        "PPOeqic": ["stand_dance_cyber_emlp/2026-02-18-13-38-36_/model_26100_obs_action.npy"],
        "SKooP-NoSym": ["stand_dance_cyber_cdae_online_next_latent/2026-02-21-08-59-03_/model_30000_obs_action.npy"],
        "SKooP": ["stand_dance_cyber_emlp_ecdae_online_next_latent/2026-02-24-11-03-28_/model_30000_obs_action.npy"]
    }
else:
    BASE_DIR = HF_DATASETS_DIR

# Enforce consistent plot order: PPO, PPOeqic, SKooP-NoSym, SKooP
ordered_methods = {}
for method_name in ["PPO", "PPOeqic", "SKooP-NoSym", "SKooP"]:
    if method_name in methods_to_plot:
        ordered_methods[method_name] = methods_to_plot[method_name]
methods_to_plot = ordered_methods

# --- Target Joints Configuration ---
JOINT_CONFIGS = {
    "Rear Knee": {
        "left_idx": 17,
        "right_idx": 20,
        "default_angle": 70 / 57.3,
    },
    "Rear Hip Pitch": {
        "left_idx": 16,
        "right_idx": 19,
        "default_angle": -45 / 57.3,
    },
    "Front Hip Pitch": {
        "left_idx": 10,
        "right_idx": 13,
        "default_angle": -45 / 57.3,
    },
    "Front Knee": {
        "left_idx": 11,
        "right_idx": 14,
        "default_angle": 70 / 57.3,
    }
}

# Grouping for Side-by-Side Subplots
JOINT_GROUPS = {
    "Rear Legs": {
        "joints": [("Rear Hip Pitch", JOINT_CONFIGS["Rear Hip Pitch"]),
                   ("Rear Knee", JOINT_CONFIGS["Rear Knee"])],
        "filename": "rear_legs_trajectories_new.pdf"
    },
    "Front Legs": {
        "joints": [("Front Hip Pitch", JOINT_CONFIGS["Front Hip Pitch"]),
                   ("Front Knee", JOINT_CONFIGS["Front Knee"])],
        "filename": "front_legs_trajectories_new.pdf"
    }
}

def load_and_extract(filepath, env_id, num_envs, joint_idx, default_angle, scale, min_steps, max_steps):
    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return None

    data = np.load(filepath, allow_pickle=True)
    env_data = data[env_id::num_envs][min_steps:max_steps]
    joint_trajectory = [(step['obs'][joint_idx] / scale) + default_angle for step in env_data]
    return joint_trajectory

def compute_and_print_rear_latex_table(methods_to_plot, joint_configs, base_dir, env_id, num_envs, scale, max_steps):
    """
    Computes quantitative gait metrics averaged over the Rear Legs across multiple seeds and multiple windows,
    and prints a LaTeX table with Mean ± Std.
    """
    timestep_s = 4 * 0.005
    eval_windows = [(100, 500), (600, 1000)]

    # Calculate the theoretical offset for a 180-degree phase shift at 2.5 Hz
    expected_offset_steps = int(((1.0 / 2.5) / 2.0) / timestep_s)

    # Filter configs for only Rear joints
    rear_joints = {name: config for name, config in joint_configs.items() if "Rear" in name}

    # Initialize LaTeX table string (Updated columns)
    latex_str = "\\begin{table}[h!]\n\\centering\n"
    latex_str += "\\resizebox{\\textwidth}{!}{\n"
    latex_str += "\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "\\textbf{Method} & \\textbf{Avg Phase MAE (rad)} & \\textbf{Avg Amplitude (rad)} & \\textbf{Avg Amp Diff (rad)} \\\\\n\\hline\n"

    for method_name, rel_paths in methods_to_plot.items():
        # Removed peak_var and gait_freq
        method_metrics = {'phase_mae': [], 'amplitude': [], 'amp_diff': []}

        for rel_path in rel_paths:
            full_path = os.path.join(base_dir, rel_path)

            run_sum = {'phase_mae': 0.0, 'amplitude': 0.0, 'amp_diff': 0.0}
            valid_evaluations_count = 0

            for joint_name, config in rear_joints.items():
                left_traj = load_and_extract(full_path, env_id, num_envs, config["left_idx"], config["default_angle"], scale, 0, max_steps)
                right_traj = load_and_extract(full_path, env_id, num_envs, config["right_idx"], config["default_angle"], scale, 0, max_steps)

                if left_traj is None or right_traj is None:
                    continue

                # Evaluate metrics for each window independently
                for start_idx, end_idx in eval_windows:
                    l_steady = np.array(left_traj[start_idx:end_idx])
                    r_steady = np.array(right_traj[start_idx:end_idx])

                    if len(l_steady) == 0 or len(r_steady) == 0:
                        continue

                    # 1. Amplitude (Average of left and right) & Amplitude Difference
                    amp_l = np.max(l_steady) - np.min(l_steady)
                    amp_r = np.max(r_steady) - np.min(r_steady)
                    amp_avg = (amp_l + amp_r) / 2.0
                    amp_diff = np.abs(amp_l - amp_r)

                    # 2. Phase Alignment Error (MAE)
                    l_aligned = np.array(left_traj[:-expected_offset_steps])
                    r_aligned = np.array(right_traj[expected_offset_steps:])

                    eval_start = max(0, start_idx)
                    eval_end = min(len(l_aligned), end_idx)

                    # Ensure we have aligned data in this range
                    if eval_end > eval_start:
                        mae = np.mean(np.abs(l_aligned[eval_start:eval_end] - r_aligned[eval_start:eval_end]))
                    else:
                        mae = 0.0

                    # Accumulate for this specific evaluation window
                    run_sum['phase_mae'] += mae
                    run_sum['amplitude'] += amp_avg
                    run_sum['amp_diff'] += amp_diff
                    valid_evaluations_count += 1

            # Average across joints and windows and store as a single datapoint for this run
            if valid_evaluations_count > 0:
                method_metrics['phase_mae'].append(run_sum['phase_mae'] / valid_evaluations_count)
                method_metrics['amplitude'].append(run_sum['amplitude'] / valid_evaluations_count)
                method_metrics['amp_diff'].append(run_sum['amp_diff'] / valid_evaluations_count)

        # Compute Mean and Std across all runs for this method
        if len(method_metrics['phase_mae']) > 0:
            means = {k: np.mean(v) for k, v in method_metrics.items()}
            stds = {k: np.std(v) for k, v in method_metrics.items()}

            latex_str += (f"{method_name} & "
                          f"${means['phase_mae']:.4f} \\pm {stds['phase_mae']:.4f}$ & "
                          f"${means['amplitude']:.4f} \\pm {stds['amplitude']:.4f}$ & "
                          f"${means['amp_diff']:.4f} \\pm {stds['amp_diff']:.4f}$ \\\\\n")
        else:
            latex_str += f"{method_name} & N/A & N/A & N/A \\\\\n"

    latex_str += "\\hline\n\\end{tabular}\n}\n"
    latex_str += "\\caption{Average Quantitative Gait Metrics for Rear Legs (Computed over windows 100-500 and 600-1000, $\\pm$ 1 Std Dev)}\n"
    latex_str += "\\label{tab:rear_gait_metrics}\n\\end{table}"

    print("\n" + "="*50)
    print("GENERATED LATEX TABLE")
    print("="*50)
    print(latex_str)
    print("="*50 + "\n")

# Run the metrics function
compute_and_print_rear_latex_table(methods_to_plot, JOINT_CONFIGS, BASE_DIR, ENV_ID_TO_PLOT, NUM_ENVS, DOF_POS_SCALE, MAX_TIMESTEPS)

# --- Plotting Loop ---
for group_name, group_data in JOINT_GROUPS.items():
    print(f"Generating side-by-side plots for: {group_name}...")

    # Create a 4x2 grid: 4 methods (rows) x 2 joints (columns)
    fig, axes = plt.subplots(4, 2, figsize=(16, 6), sharex=True)

    for row_idx, (method_name, rel_paths) in enumerate(methods_to_plot.items()):
        # Just use the FIRST run/seed for visualization purposes
        first_rel_path = rel_paths[0]
        full_path = os.path.join(BASE_DIR, first_rel_path)

        for col_idx, (joint_name, config) in enumerate(group_data["joints"]):
            ax = axes[row_idx, col_idx]

            left_traj = load_and_extract(full_path, ENV_ID_TO_PLOT, NUM_ENVS, config["left_idx"], config["default_angle"], DOF_POS_SCALE, MIN_TIMESTEPS, MAX_TIMESTEPS)
            right_traj = load_and_extract(full_path, ENV_ID_TO_PLOT, NUM_ENVS, config["right_idx"], config["default_angle"], DOF_POS_SCALE, MIN_TIMESTEPS, MAX_TIMESTEPS)

            if left_traj is not None and right_traj is not None:
                if OFFSET_STEPS > 0:
                    left_plot = left_traj[:-OFFSET_STEPS]
                    right_plot = right_traj[OFFSET_STEPS:]
                    time_steps = range(len(left_plot))
                else:
                    left_plot = left_traj
                    right_plot = right_traj
                    time_steps = range(len(left_plot))

                ax.plot(time_steps, left_plot, label=f"Left {joint_name}", color='blue', alpha=0.8, linewidth=1.5)

                right_label = f"Right {joint_name} (Offset by {OFFSET_STEPS})" if OFFSET_STEPS > 0 else f"Right {joint_name}"
                ax.plot(time_steps, right_plot, label=right_label, color='red', linestyle='--', alpha=0.8, linewidth=1.5)

            ax.set_title(f"{method_name} - {joint_name}", fontsize=16)
            ax.tick_params(axis="both", which="both", labelsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)
            if row_idx == len(list(methods_to_plot.items())) - 1 and col_idx==1:
                ax.legend(loc='upper right', fontsize=12, ncol=2)

    axes[3, 0].set_xlabel("Timestep", fontsize=14)
    axes[3, 1].set_xlabel("Timestep", fontsize=14)
    fig.supylabel("Joint Pos (rad)", fontsize=14, x=0.015)
    plt.tight_layout()

    filename = group_data["filename"].replace('.pdf', '_offset.pdf') if OFFSET_STEPS > 0 else group_data["filename"]
    plt.savefig("paper_plots/" + filename, dpi=300)
    plt.close()
    print(f"[✓] Saved as {filename}\n")