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
parser.add_argument('--task', type=str, default="cyber2_stand_dance")
parser.add_argument('--load_run', type=str, default='latest')
args, _ = parser.parse_known_args()

# Test grids
frictions = [0.4, 1.0]
mass_offsets = [0.0, 0.5]  # Added mass in kg (negative means lighter)
push_vels = [0.0, 0.3]
armatures = [0.002, 0.005]

# Build the list of independent tests (we keep other params at baseline: F=0.8, M=1.0, P=0.0)
tests = []
for f in frictions: tests.append({'f': f, 'm': 0.0, 'p': 0.0, 'a': 0.005, 'name': f'Friction $\mu$={f}'})
for m in mass_offsets: 
    if m != 0.0: tests.append({'f': 1.0, 'm': m, 'p': 0.0, 'a': 0.005, 'name': f'Mass Offset {m}kg'})
for p in push_vels: 
    if p != 0.0: tests.append({'f': 1.0, 'm': 0.0, 'p': p, 'a': 0.005, 'name': f'Push Vel {p}m/s'})
for a in armatures:
    if a != 0.0: tests.append({'f': 1.0, 'm': 0.0, 'p': 0.0, 'a': a, 'name': f'Armature {a}'})

results = []

def compute_mses(data_dict):
    """Calculates Linear XY MSE and Angular Z (Yaw) MSE."""
    lin_xy_mse = np.mean((data_dict['cmd_vx'] - data_dict['actual_vx'])**2 + 
                         (data_dict['cmd_vy'] - data_dict['actual_vy'])**2)
    yaw_mse = np.mean((data_dict['cmd_yaw_vels'] - data_dict['actual_yaw_vels'])**2)
    return lin_xy_mse, yaw_mse

print("Starting Sim-to-Sim Evaluation Suite...")

for t in tests:
    print(f"\n--- Running Test: {t['name']} ---")

    input(f"About to run test with friction={t['f']}, mass_offset={t['m']}, push_vel={t['p']}, armature={t['a']}. Press Enter to continue...")
    
    # 1. Run Isaac Gym
    print("Running Isaac Gym rollout...")
    print(f"task: {args.task}, load_run: {args.load_run}, checkpoint: {args.checkpoint}, friction: {t['f']}, mass_offset: {t['m']}, push_vel: {t['p']}. Press Enter to continue...")
    subprocess.run([
        "python", "legged_gym/scripts/play_and_save.py", "--task", args.task, "--headless",
        "--checkpoint", args.checkpoint, "--load_run", args.load_run,
        "--friction", str(t['f']), "--mass_offset", str(t['m']), "--push_vel", str(t['p'])
    ], check=True)
    
    # 2. Run MuJoCo
    print("Running MuJoCo rollout...")
    subprocess.run([
        "python", "legged_gym/scripts/play_mujoco.py", "--headless", "--task", args.task,
        "--checkpoint", args.checkpoint, "--load_run", args.load_run,
        "--friction", str(t['f']), "--mass_offset", str(t['m']), "--push_vel", str(t['p'])
    ], check=True)
    
    # 3. Load Data
    # Extract task base name and suffix
    task_parts = args.task.replace("cyber2_", "").split("_")
    # Reconstruct as: base_task + _cyber + suffix (if any)
    base_task = "_".join(task_parts[:-1]) if len(task_parts) > 1 and task_parts[-1] not in ["cyber"] else args.task.replace("cyber2_", "")
    
    # Extract the task type and any suffix after it
    if "stand_dance" in args.task:
        suffix = args.task.replace("cyber2_stand_dance", "").lstrip("_")
        exp_name = "stand_dance_cyber" + ("_" + suffix if suffix else "")
    elif "walk_slope" in args.task:
        suffix = args.task.replace("cyber2_walk_slope", "").lstrip("_")
        exp_name = "walk_slope_cyber" + ("_" + suffix if suffix else "")
    elif "push_door" in args.task:
        suffix = args.task.replace("cyber2_push_door", "").lstrip("_")
        exp_name = "push_door_cyber" + ("_" + suffix if suffix else "")
    else:
        raise ValueError(f"Unrecognized task name: {args.task}")
    LOG_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", exp_name, args.load_run)
    isaac_file = os.path.join(LOG_DIR, f"{args.checkpoint}_yaw_heading_data_isaacgym.npy")
    mujoco_file = os.path.join(LOG_DIR, f"{args.checkpoint}_yaw_heading_data_mujoco.npy") # MuJoCo script saves to same name
    
    # Note: You may want to modify your base scripts to save with distinct prefixes 
    # (e.g., isaac_yaw_heading.npy and mujoco_yaw_heading.npy) to avoid overwriting.
    # Assuming for this script you updated them to save as 'isaac_data.npy' and 'mujoco_data.npy'.
    isaac_data = np.load(isaac_file, allow_pickle=True).item()
    mujoco_data = np.load(mujoco_file, allow_pickle=True).item()
    
    isaac_lin_mse, isaac_yaw_mse = compute_mses(isaac_data)
    muj_lin_mse, muj_yaw_mse = compute_mses(mujoco_data)
    
    results.append({
        'name': t['name'],
        'isaac_lin': isaac_lin_mse, 'muj_lin': muj_lin_mse,
        'isaac_yaw': isaac_yaw_mse, 'muj_yaw': muj_yaw_mse
    })

    # Plot baseline comparison (Friction=0.8, Mass=1.0, Push=0.0)
    if t['f'] == 0.8 and t['m'] == 1.0 and t['p'] == 0.0 and t['a'] == 0.005:
        print("Generating Comparison Plots for Baseline...")
        
        plt.figure(figsize=(6, 3))
        plt.plot(mujoco_data['cmd_heading'], label='Commanded', linestyle='--', color='black', linewidth=2)
        plt.plot(isaac_data['actual_heading'], label='Isaac Gym Actual', alpha=0.8, color='blue')
        plt.plot(mujoco_data['actual_heading'], label='MuJoCo Actual', alpha=0.8, color='red')
        plt.title('Heading Tracking: Isaac Gym vs MuJoCo')
        plt.xlabel('Steps')
        plt.ylabel('Heading (rad)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR, f"{args.checkpoint}_sim_to_sim_heading_comparison.pdf"))
        plt.close()

        plt.figure(figsize=(6, 3))
        plt.plot(isaac_data['cmd_yaw_vels'], label='Commanded', linestyle='--', color='black', linewidth=2)
        plt.plot(isaac_data['actual_yaw_vels'], label='Isaac Gym Actual', alpha=0.8, color='blue')
        plt.plot(mujoco_data['actual_yaw_vels'], label='MuJoCo Actual', alpha=0.8, color='red')
        plt.title('Yaw Velocity Tracking: Isaac Gym vs MuJoCo')
        plt.xlabel('Steps')
        plt.ylabel('Yaw Vel (rad/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR, f"{args.checkpoint}_sim_to_sim_yaw_vel_comparison.pdf"))
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.plot(isaac_data['base_x_positions'], isaac_data['base_y_positions'], label='Isaac Gym', alpha=0.8, color='blue')
        plt.plot(mujoco_data['base_x_positions'], mujoco_data['base_y_positions'], label='MuJoCo', alpha=0.8, color='red')
        plt.title('Base Position Trajectory: Isaac Gym vs MuJoCo')
        plt.xlabel('Steps')
        plt.ylabel('Base Y Position (m)')
        plt.axis('equal')  # Set equal aspect ratio
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR, f"{args.checkpoint}_sim_to_sim_base_pos_comparison.pdf"))
        plt.close()

# --- GENERATE LATEX TABLE ---
print("\n=== LATEX TABLE OUTPUT ===")
latex_str = r"""\begin{table}[h]
\centering
\caption{Sim-to-Sim Tracking Performance (MSE) under Various Domain Randomizations}
\label{tab:sim2sim_robustness}
\begin{tabular}{l c c c c}
\toprule
\textbf{Test Condition} & \multicolumn{2}{c}{\textbf{Linear XY MSE ($m/s)^2$}} & \multicolumn{2}{c}{\textbf{Angular Z MSE ($rad/s)^2$}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& Isaac Gym & MuJoCo & Isaac Gym & MuJoCo \\
\midrule
"""

for r in results:
    latex_str += f"{r['name']} & {r['isaac_lin']:.4f} & {r['muj_lin']:.4f} & {r['isaac_yaw']:.4f} & {r['muj_yaw']:.4f} \\\\\n"

latex_str += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex_str)