import os
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from legged_gym import LEGGED_GYM_ROOT_DIR

plt.rcParams.update({'font.family':'serif'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# --- CONFIGURATION ---

# Parse args early to catch headless mode before loading MuJoCo
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--checkpoint', type=str, default="30000")
parser.add_argument('--task', type=str, default="cyber2_stand_dance_emlp_ecdae_online_next_latent")
parser.add_argument('--load_run', type=str, default='latest')
parser.add_argument('--baseline_run', type=str, default='latest', help="Run folder for the baseline PPO model")
args, _ = parser.parse_known_args()

# Test grids
frictions = [0.4, 1.0]
mass_offsets = [0.0, 0.5]  # Added mass in kg (negative means lighter)
push_vels = [0.0, 0.3]
armatures = [0.0, 0.002, 0.005]

# Build the list of independent tests (we keep other params at baseline: F=0.8, M=1.0, P=0.0)
tests = []
for f in frictions: tests.append({'f': f, 'm': 1.0, 'p': 0.0, 'a': 0.0, 'name': f'Friction $\mu$={f}'})
for m in mass_offsets: 
    if m != 0.0: tests.append({'f': 0.8, 'm': m, 'p': 0.0, 'a': 0.0, 'name': f'Mass Offset {m}kg'})
for p in push_vels: 
    if p != 0.0: tests.append({'f': 0.8, 'm': 1.0, 'p': p, 'a': 0.0, 'name': f'Push Vel {p}m/s'})
for a in armatures:
    if a != 0.0: tests.append({'f': 0.8, 'm': 1.0, 'p': 0.0, 'a': a, 'name': f'Armature {a}'})

results = []

def compute_mses(data_dict):
    """Calculates Linear XY MSE and Angular Z (Yaw) MSE."""
    lin_xy_mse = np.mean((data_dict['cmd_vx'] - data_dict['actual_vx'])**2 + 
                         (data_dict['cmd_vy'] - data_dict['actual_vy'])**2)
    yaw_mse = np.mean((data_dict['cmd_yaw_vels'] - data_dict['actual_yaw_vels'])**2)
    return lin_xy_mse, yaw_mse

# Extract base names for baseline (PPO) and the new task (SKooP)
if "stand_dance" in args.task:
    baseline_task = "cyber2_stand_dance"
    exp_name_ppo = "stand_dance_cyber"
    suffix = args.task.replace("cyber2_stand_dance", "").lstrip("_")
    exp_name_skoop = "stand_dance_cyber" + ("_" + suffix if suffix else "")
elif "walk_slope" in args.task:
    baseline_task = "cyber2_walk_slope"
    exp_name_ppo = "walk_slope_cyber"
    suffix = args.task.replace("cyber2_walk_slope", "").lstrip("_")
    exp_name_skoop = "walk_slope_cyber" + ("_" + suffix if suffix else "")
elif "push_door" in args.task:
    baseline_task = "cyber2_push_door"
    exp_name_ppo = "push_door_cyber"
    suffix = args.task.replace("cyber2_push_door", "").lstrip("_")
    exp_name_skoop = "push_door_cyber" + ("_" + suffix if suffix else "")
else:
    raise ValueError(f"Unrecognized task name: {args.task}")

print(f"Baseline task identified as: {baseline_task}")
print(f"SKooP task identified as: {args.task}")
print("Starting MuJoCo Evaluation Suite...")

for t in tests:
    print(f"\n--- Running Test: {t['name']} ---")
    
    # 1. Run MuJoCo for PPO (Baseline)
    print("Running PPO baseline rollout...")
    subprocess.run([
        "python", "legged_gym/scripts/play_mujoco.py", "--task", baseline_task, "--headless",
        "--checkpoint", args.checkpoint, "--load_run", args.baseline_run,
        "--friction", str(t['f']), "--mass_offset", str(t['m']), "--push_vel", str(t['p'])
    ], check=True)
    
    # 2. Run MuJoCo for SKooP
    print("Running SKooP rollout...")
    subprocess.run([
        "python", "legged_gym/scripts/play_mujoco.py", "--task", args.task, "--headless",
        "--checkpoint", args.checkpoint, "--load_run", args.load_run,
        "--friction", str(t['f']), "--mass_offset", str(t['m']), "--push_vel", str(t['p'])
    ], check=True)
    
    # 3. Load Data
    LOG_DIR_PPO = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", exp_name_ppo, args.baseline_run)
    LOG_DIR_SKOOP = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", exp_name_skoop, args.load_run)
    
    ppo_file = os.path.join(LOG_DIR_PPO, f"{args.checkpoint}_yaw_heading_data_mujoco.npy")
    skoop_file = os.path.join(LOG_DIR_SKOOP, f"{args.checkpoint}_yaw_heading_data_mujoco.npy") 
    
    ppo_data = np.load(ppo_file, allow_pickle=True).item()
    skoop_data = np.load(skoop_file, allow_pickle=True).item()
    
    ppo_lin_mse, ppo_yaw_mse = compute_mses(ppo_data)
    skoop_lin_mse, skoop_yaw_mse = compute_mses(skoop_data)
    
    results.append({
        'name': t['name'],
        'ppo_lin': ppo_lin_mse, 'skoop_lin': skoop_lin_mse,
        'ppo_yaw': ppo_yaw_mse, 'skoop_yaw': skoop_yaw_mse
    })

    # Plot baseline comparison (Friction=0.8, Mass=1.0, Push=0.0)
    if t['f'] == 0.8 and t['m'] == 1.0 and t['p'] == 0.0:
        print("Generating Comparison Plots for Baseline Condition...")
        
        plt.figure(figsize=(6, 3))
        plt.plot(skoop_data['cmd_heading'], label='Commanded', linestyle='--', color='black', linewidth=2)
        plt.plot(ppo_data['actual_heading'], label='PPO Actual', alpha=0.8, color='blue')
        plt.plot(skoop_data['actual_heading'], label='SKooP Actual', alpha=0.8, color='red')
        plt.title('Heading Tracking in MuJoCo: PPO vs SKooP')
        plt.xlabel('Steps')
        plt.ylabel('Heading (rad)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR_SKOOP, f"{args.checkpoint}_ppo_vs_skoop_heading_comparison.pdf"))
        plt.close()

        plt.figure(figsize=(6, 3))
        plt.plot(skoop_data['cmd_yaw_vels'], label='Commanded', linestyle='--', color='black', linewidth=2)
        plt.plot(ppo_data['actual_yaw_vels'], label='PPO Actual', alpha=0.8, color='blue')
        plt.plot(skoop_data['actual_yaw_vels'], label='SKooP Actual', alpha=0.8, color='red')
        plt.title('Yaw Velocity Tracking in MuJoCo: PPO vs SKooP')
        plt.xlabel('Steps')
        plt.ylabel('Yaw Vel (rad/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR_SKOOP, f"{args.checkpoint}_ppo_vs_skoop_yaw_vel_comparison.pdf"))
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.plot(ppo_data['base_x_positions'], ppo_data['base_y_positions'], label='PPO', alpha=0.8, color='blue')
        plt.plot(skoop_data['base_x_positions'], skoop_data['base_y_positions'], label='SKooP', alpha=0.8, color='red')
        plt.title('Base Position Trajectory: PPO vs SKooP')
        plt.xlabel('Steps')
        plt.ylabel('Base Y Position (m)')
        plt.axis('equal')  # Set equal aspect ratio
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR_SKOOP, f"{args.checkpoint}_ppo_vs_skoop_base_pos_comparison.pdf"))
        plt.close()

# --- GENERATE LATEX TABLE ---
print("\n=== LATEX TABLE OUTPUT ===")
latex_str = r"""\begin{table}[h]
\centering
\caption{MuJoCo Tracking Performance (MSE): PPO vs SKooP}
\label{tab:ppo_vs_skoop_robustness}
\begin{tabular}{l c c c c}
\toprule
\textbf{Test Condition} & \multicolumn{2}{c}{\textbf{Linear XY MSE ($m/s)^2$}} & \multicolumn{2}{c}{\textbf{Angular Z MSE ($rad/s)^2$}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& PPO & SKooP & PPO & SKooP \\
\midrule
"""

for r in results:
    latex_str += f"{r['name']} & {r['ppo_lin']:.4f} & {r['skoop_lin']:.4f} & {r['ppo_yaw']:.4f} & {r['skoop_yaw']:.4f} \\\\\n"

latex_str += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex_str)