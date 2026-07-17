import os
import re
import subprocess
import numpy as np
import sys

# Try to import LEGGED_GYM_ROOT_DIR, fallback to current dir if it fails
try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ImportError:
    print("Warning: Could not import legged_gym. Using current working directory.")
    LEGGED_GYM_ROOT_DIR = os.getcwd()

def load_methods_from_txt(file_path):
    """Reads the python-formatted list from the text file."""
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        sys.exit(1)

    with open(file_path, 'r') as f:
        content = f.read()

    # # Clean up any potential artifacts from copy-pasting
    # content = re.sub(r'\', '', content)

    local_vars = {}
    try:
        exec(content, {}, local_vars)
    except Exception as e:
        print(f"Error parsing the text file: {e}")
        sys.exit(1)

    if 'log_dirs_and_labels' not in local_vars:
        print("Error: 'log_dirs_and_labels' variable not found in the text file.")
        sys.exit(1)

    return local_vars['log_dirs_and_labels']

def main():
    txt_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "push_door.txt")
    log_dirs_and_labels = load_methods_from_txt(txt_file_path)

    results = {}

    for paths, label in log_dirs_and_labels:
        results[label] = {'normal': [], 'left': []}
        print(f"\nEvaluating Method: {label}")
        print("-" * 40)

        for path in paths:
            path = path.strip()
            if not path: continue

            # Isolate the exact run directory (e.g. 2025-06-24-11-36-27_)
            run_name = os.path.basename(os.path.normpath(path))

            # Identify the parent task mapping based on folder structure
            exp_name = os.path.basename(os.path.dirname(os.path.normpath(path)))
            task_name = exp_name.replace("push_door_cyber", "cyber2_push_door")

            for side in ['normal', 'left']:
                cmd = [
                    "python", "play_mujoco.py",
                    "--task", task_name,
                    "--load_run", run_name,
                    "--eval_success",
                    "--headless"
                ]
                if side == 'left':
                    cmd.append("--left")

                print(f"  -> Running: {run_name} ({side} door)")
                result = subprocess.run(cmd, capture_output=True, text=True)

                # Regex scrape the standard output for the specific success string
                # We specifically look for the run_name to avoid false matches
                regex_pattern = r'-> Success Rate for ' + re.escape(run_name) + r'.*?: ([\d\.]+)%'
                match = re.search(regex_pattern, result.stdout)

                if match:
                    sr = float(match.group(1))
                    results[label][side].append(sr)
                    print(f"     [✓] SR: {sr:.2f}%")
                else:
                    print(f"     [X] Error scraping output for {run_name}!")
                    print(f"         --- Script Output below ---")
                    if result.stderr.strip():
                        print(f"         STDERR: {result.stderr.strip()}")
                    if result.stdout.strip():
                        # Print the last 10 lines of stdout so it doesn't flood the terminal
                        out_lines = result.stdout.strip().split('\n')
                        print(f"         STDOUT (last 10 lines): \n         " + "\n         ".join(out_lines[-10:]))
                    results[label][side].append(0.0)

    # Generate LaTeX Table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Normal Door (\%) & Left Door (\%) \\",
        r"\midrule"
    ]

    # Publication-ready names mapped to the order in the text file
    NEW_NAMES = [
        "PPO",
        "PPOeqic",
        "SKooP-NoSym-NoPred",
        "SKooP-NoSym",
        "SKooP-NoPred",
        "SKooP"
    ]

    for i, (label, scores) in enumerate(results.items()):
        norm_avg = np.mean(scores['normal']) if scores['normal'] else 0.0
        norm_std = np.std(scores['normal']) if scores['normal'] else 0.0
        left_avg = np.mean(scores['left']) if scores['left'] else 0.0
        left_std = np.std(scores['left']) if scores['left'] else 0.0

        # Map to the new name based on index, fallback to original label if index goes out of bounds
        clean_label = NEW_NAMES[i] if i < len(NEW_NAMES) else label.title()

        latex_lines.append(f"{clean_label} & ${norm_avg:.1f} \\pm {norm_std:.1f}$ & ${left_avg:.1f} \\pm {left_std:.1f}$ \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Push Door Success Rates across Methods and Door Configurations}",
        r"\label{tab:push_door_success}",
        r"\end{table}"
    ])

    latex_output = "\n".join(latex_lines)

    # Print to console
    print("\n\n" + "="*40)
    print("LaTeX Results Table")
    print("="*40 + "\n")
    print(latex_output)

    # Save to a .tex file
    output_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "push_door_results_table.tex")
    with open(output_path, "w") as f:
        f.write(latex_output)
    print(f"\n[✓] LaTeX table successfully saved to: {output_path}")

if __name__ == "__main__":
    main()