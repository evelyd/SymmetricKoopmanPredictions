import os
import numpy as np
from collections import defaultdict
import pandas as pd
from legged_gym import LEGGED_GYM_ROOT_DIR

base_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "isaacgym_recordings")

results = []

for folder in os.listdir(base_dir):
    print(f"{folder}")
    if folder.startswith("push_door"):
    # if folder == "push_door_cyber":
        folder_path = os.path.join(base_dir, folder)
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                # Find right and left npy files
                npy_files = [f for f in os.listdir(subdir_path) if f.endswith('.npy')]
                right_file = next((f for f in npy_files if not f.endswith('_left.npy') and 'ood' not in f), None)
                left_file = next((f for f in npy_files if f.endswith('_left.npy') and 'ood' not in f), None)
                right_ood_file = next((f for f in npy_files if not f.endswith('_left.npy') and 'ood' in f), None)
                left_ood_file = next((f for f in npy_files if f.endswith('_left.npy') and 'ood' in f), None)
                if  right_file and left_file and "2025-06-26-10-45-16_" not in subdir and "2025-07-17-15-17-32_" not in subdir:
                # "2025-07-21-11-33-44_" not in subdir and "2025-07-21-11-33-54_" not in subdir:
                    print(f"  {subdir}: Found both right and left files.")
                    right_data = np.load(os.path.join(subdir_path, right_file), allow_pickle=True)
                    left_data = np.load(os.path.join(subdir_path, left_file), allow_pickle=True)
                    right_ood_data = np.load(os.path.join(subdir_path, right_ood_file), allow_pickle=True) if right_ood_file else None
                    left_ood_data = np.load(os.path.join(subdir_path, left_ood_file), allow_pickle=True) if left_ood_file else None
                    # Assume success_rate is stored in the last element as a dict
                    right_sr = right_data.item().get('success_rate', None)
                    left_sr = left_data.item().get('success_rate', None)
                    right_ood_sr = right_ood_data.item().get('success_rate', None)
                    print(f"    Right OOD SR: {right_ood_sr}")
                    left_ood_sr = left_ood_data.item().get('success_rate', None)
                    print(f"    Left OOD SR: {left_ood_sr}")

                    results.append({
                        'folder': folder,
                        'subdir': subdir,
                        'right_success_rate': right_sr,
                        'left_success_rate': left_sr,
                        'right_ood_success_rate': right_ood_sr,
                        'left_ood_success_rate': left_ood_sr
                    })
                    print(f"    Right SR: {right_sr}, Left SR: {left_sr}")
                    print(f"    Right OOD SR: {right_ood_sr}, Left OOD SR: {left_ood_sr}")

# Compute SI for each subdir and group results by folder

folder_stats = defaultdict(list)

for r in results:
    right_sr = r['right_success_rate']
    left_sr = r['left_success_rate']
    right_ood_sr = r['right_ood_success_rate']
    left_ood_sr = r['left_ood_success_rate']
    if right_sr is not None and left_sr is not None and (right_sr + left_sr) != 0:
        si = (2 * np.abs(right_sr - left_sr)) / (right_sr + left_sr) * 100
    else:
        si = None

    if right_ood_sr is not None and left_ood_sr is not None and (right_ood_sr + left_ood_sr) != 0:
        si_ood = (2 * np.abs(right_ood_sr - left_ood_sr)) / (right_ood_sr + left_ood_sr) * 100
        print(f"    OOD SI: {si_ood}")
    else:
        si_ood = None
    r['SI'] = si
    r['OOD SI'] = si_ood
    folder_stats[r['folder']].append(r)

# Prepare data for DataFrame
data = []
for folder, subdir_results in folder_stats.items():
    right_srs = [r['right_success_rate'] for r in subdir_results if r['right_success_rate'] is not None]
    left_srs = [r['left_success_rate'] for r in subdir_results if r['left_success_rate'] is not None]
    right_ood_srs = [r['right_ood_success_rate'] for r in subdir_results if r['right_ood_success_rate'] is not None]
    left_ood_srs = [r['left_ood_success_rate'] for r in subdir_results if r['left_ood_success_rate'] is not None]
    sis = [r['SI'] for r in subdir_results if r['SI'] is not None]
    ood_sis = [r['OOD SI'] for r in subdir_results if r['OOD SI'] is not None]
    if right_srs and left_srs and sis:
        right_mean = np.mean(right_srs) * 100
        right_std = np.std(right_srs) * 100
        left_mean = np.mean(left_srs) * 100
        left_std = np.std(left_srs) * 100
        ood_right_mean = np.mean(right_ood_srs) * 100
        ood_right_std = np.std(right_ood_srs) * 100
        ood_left_mean = np.mean(left_ood_srs) * 100
        ood_left_std = np.std(left_ood_srs) * 100
        si_mean = np.mean(sis)
        si_std = np.std(sis)
        si_ood_mean = np.mean(ood_sis)
        si_ood_std = np.std(ood_sis)
        data.append({
            "Folder": folder,
            "Right SR (mean ± std)": f"{right_mean:.2f} ± {right_std:.2f}",
            "Left SR (mean ± std)": f"{left_mean:.2f} ± {left_std:.2f}",
            "SI (mean ± std)": f"{si_mean:.2f} ± {si_std:.2f}",
            "OOD Right SR (mean ± std)": f"{ood_right_mean:.2f} ± {ood_right_std:.2f}",
            "OOD Left SR (mean ± std)": f"{ood_left_mean:.2f} ± {ood_left_std:.2f}",
            "OOD SI (mean ± std)": f"{si_ood_mean:.2f} ± {si_ood_std:.2f}"
        })

table_titles = [("push_door_cyber", "Baseline"),
                ("push_door_cyber_emlp", "EMLP"),
                ("push_door_cyber_cdae", "C-DAE"),
                ("push_door_cyber_cdae_online", "Online C-DAE"),
                ("push_door_cyber_emlp_ecdae", "EMLP + EC-DAE"),
                ("push_door_cyber_emlp_ecdae_online", "EMLP + Online EC-DAE")]

# Map folder names to display names using table_titles
folder_name_map = dict(table_titles)
# Create an ordered list of display names
ordered_display_names = [display for _, display in table_titles]

# Replace folder names with display names
for row in data:
    if row["Folder"] in folder_name_map:
        row["Folder"] = folder_name_map[row["Folder"]]

# Sort data according to the order in table_titles
data_sorted = [row for name in ordered_display_names for row in data if row["Folder"] == name]

df = pd.DataFrame(data_sorted)
print("\nSummary Table:")
print(df.to_string(index=False))

# Convert DataFrame to LaTeX table
latex_table = df.to_latex(index=False, caption="Summary Table", label="tab:summary_table")
print("\nLaTeX Table:")
print(latex_table)