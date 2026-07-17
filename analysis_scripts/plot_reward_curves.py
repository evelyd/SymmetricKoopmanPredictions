import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
from tqdm.auto import tqdm # For progress bars, helpful for long operations

plt.rcParams.update({'font.family':'serif'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def extract_scalars_from_events(log_dir):
    """Extract scalar values from TensorFlow event files in a directory."""
    scalars = {}
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(root, file)
                try:
                    event_acc = EventAccumulator(event_file)
                    event_acc.Reload()
                    for tag in event_acc.Tags()["scalars"]:
                        # Pre-allocate list if possible, though extend is fine for many small appends
                        if tag not in scalars:
                            scalars[tag] = []
                        events = event_acc.Scalars(tag)
                        # Appending tuples is generally efficient
                        scalars[tag].extend((e.step, e.value) for e in events)
                except Exception as e:
                    print(f"Warning: Could not process event file {event_file}: {e}")
    return scalars

# Your list of log directories and labels
def load_log_dirs_and_labels_from_training_data(task_name):
    """
    Loads experiment groups from ../hf_datasets/training_data/{task_name}.
    Each {task_name}_*/202* directory is treated as an experiment group.
    """
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "hf_datasets", "training_data", task_name)
    )
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Task directory not found: {base_dir}")

    log_dirs_and_labels = []
    for experiment_root in sorted(glob.glob(os.path.join(base_dir, f"{task_name}_*"))):
        if not os.path.isdir(experiment_root):
            continue
        experiment_group_label = os.path.basename(experiment_root).replace("_", " ")
        experiment_dirs_list = [
            d for d in sorted(glob.glob(os.path.join(experiment_root, "202*")))
            if os.path.isdir(d)
        ]
        if experiment_dirs_list:
            log_dirs_and_labels.append((experiment_dirs_list, experiment_group_label))
    return log_dirs_and_labels

# Example usage:
task_names = ['stand_dance', "walk_slope", "push_door"]  # Change as needed
log_dirs_and_labels_list = []
# Create a dictionary to store max_steps per task
max_steps_dict = {}
for task_name in task_names:
    log_dirs_and_labels_list.append(load_log_dirs_and_labels_from_training_data(task_name))
    if "stand_dance" in task_name or "walk_slope" in task_name:
        max_steps_dict[task_name] = 30000
    else:
        max_steps_dict[task_name] = 18000

# --- Define the list of scalar tags you want to plot ---
reward_scalar_tags_to_plot = [
    # 'Episode/rew_upright',
    # 'Episode/rew_tracking_lin_vel',
    'Train/mean_reward',
    # 'Loss/value_function',
]

# --- Define output directory for plots ---
output_dir = 'paper_plots'
os.makedirs(output_dir, exist_ok=True)

# --- Data Pre-processing: Load all scalars from all logs once ---
all_data_for_dataframe = []

print("--- Pre-loading all scalar data from log directories ---")

total_log_dirs = sum(len(experiment_dirs_list) for log_dirs_and_labels in log_dirs_and_labels_list for experiment_dirs_list, _ in log_dirs_and_labels)

with tqdm(total=total_log_dirs, desc="Loading Log Data") as pbar:
    for log_dirs_and_labels in log_dirs_and_labels_list:

        for experiment_dirs_list, experiment_group_label in log_dirs_and_labels:
            print(f"Processing experiment group: {experiment_group_label}")

            # Find the max_steps for the current task
            current_task = None
            for task_name in task_names:
                if task_name.replace('_', ' ') in experiment_group_label:
                    current_task = task_name
                    break

            max_steps = max_steps_dict.get(current_task, 15000) # Default to 15000 if not found

            for log_dir in experiment_dirs_list:

                # --- Extract original_seed from env.yaml ---
                original_seed = None
                env_yaml_path = os.path.join(log_dir, 'params', 'env.yaml')
                if os.path.exists(env_yaml_path):
                    try:
                        with open(env_yaml_path, 'r') as f:
                            env_params = yaml.load(f, Loader=yaml.UnsafeLoader)
                        if 'seed' in env_params:
                            original_seed = env_params['seed']
                        else:
                            original_seed = log_dir # Fallback
                    except Exception as e:
                        original_seed = log_dir # Fallback
                else:
                    original_seed = log_dir # Fallback

                # --- Extract ALL scalars using your function ---
                scalars_dict_for_this_log = extract_scalars_from_events(log_dir)

                # Prepare data for DataFrame creation
                for tag, data_points in scalars_dict_for_this_log.items():
                    if data_points: # Only add if there's actual data for this tag
                        for step, value in data_points:
                            if step <= max_steps:
                                all_data_for_dataframe.append({
                                    'step': step,
                                    'value': value,
                                    'tag': tag,
                                    'experiment_group': experiment_group_label,
                                    'original_seed': original_seed,
                                    'log_dir': log_dir # Potentially useful for debugging, but not directly for plot
                                })
                pbar.update(1)

print("--- Finished pre-loading data. Building combined DataFrame ---")
if not all_data_for_dataframe:
    print("No data collected for plotting.")
    # Do not exit, allow script to finish gracefully if no data
else:
    # Create the single, master DataFrame
    df_combined_all_scalars = pd.DataFrame(all_data_for_dataframe)
    print(f"Combined DataFrame created with {len(df_combined_all_scalars)} rows.")
    print(df_combined_all_scalars.head()) # Show a glimpse of the combined data

    # Set the legend
    legend_labels = ['PPO', 'SKooP-NoSym-NoPred', 'SKoop-NoSym', 'PPOeqic', 'SKooP-NoPred', 'SKooP']

    # --- Define smoothing window size ---
    SMOOTHING_WINDOW_SIZE = 500

    # --- Define a downsampling rate
    SUBSAMPLING_RATE = 20

    def smooth_and_resample_data(df, window_size, downsampling_rate):
        """
        Resamples and smooths the data onto a uniform timeline for each seed.
        """
        smoothed_data_list = []
        for (group, seed), group_df in df.groupby(['experiment_group', 'original_seed']):
            group_df = group_df.sort_values(by='step')
            resample_timeline = pd.Series(group_df['value'].values, index=pd.to_datetime(group_df['step'], unit='s'))
            resampled_data = resample_timeline.resample('1s').interpolate(method='linear')
            smoothed_values = resampled_data.rolling(
                window=f'{window_size}s', min_periods=1, center=True
            ).mean()
            smoothed_df = pd.DataFrame({
                'step': smoothed_values.index.astype(np.int64) // 10**9,
                'smoothed_value': smoothed_values.values
            })
            smoothed_df['experiment_group'] = group
            smoothed_df['original_seed'] = seed
            smoothed_data_list.append(smoothed_df)

        df_smoothed_full = pd.concat(smoothed_data_list, ignore_index=True)
        df_downsampled = df_smoothed_full[df_smoothed_full['step'] % downsampling_rate == 0].copy()
        return df_downsampled

    # --- New Plotting Logic for Single Plot with Rows ---
    num_tags = len(reward_scalar_tags_to_plot)
    num_tasks = len(task_names)
    fig, axes = plt.subplots(num_tags, num_tasks, figsize=(0.8 * 8 * num_tasks, 0.8 * 5 * num_tags), sharex=False, sharey=False) # Important: Don't share x-axis

    # Handle the case of single row/column
    if num_tags == 1 and num_tasks == 1:
        axes = np.array([[axes]])
    elif num_tags == 1:
        axes = axes.reshape(1, -1)
    elif num_tasks == 1:
        axes = axes.reshape(-1, 1)

    # --- Pre-process all data once to avoid redundant computations
    print("--- Processing and smoothing all data for unified plot ---")
    df_all_processed = pd.DataFrame()
    for current_reward_tag in tqdm(reward_scalar_tags_to_plot, desc="Smoothing Data"):
        df_tag = df_combined_all_scalars[df_combined_all_scalars['tag'] == current_reward_tag].copy()
        if not df_tag.empty:
            df_processed_tag = smooth_and_resample_data(df_tag, SMOOTHING_WINDOW_SIZE, SUBSAMPLING_RATE)
            df_processed_tag['tag'] = current_reward_tag # Add tag back
            df_all_processed = pd.concat([df_all_processed, df_processed_tag], ignore_index=True)

    print("--- Generating unified plot ---")
    for row_idx, current_reward_tag in enumerate(reward_scalar_tags_to_plot):
        for col_idx, task in enumerate(task_names):
            ax = axes[row_idx, col_idx]

            df_for_current = df_all_processed[
                (df_all_processed['tag'] == current_reward_tag) &
                (df_all_processed['experiment_group'].str.contains(task.replace('_', ' '), na=False, case=False, regex=False))
            ].copy()

            if df_for_current.empty:
                ax.set_title(f"No data for {task.replace('_', ' ').title()}")
                ax.axis('off')
                continue

            experiment_groups = df_for_current['experiment_group'].unique()
            legend_label_map = dict(zip(experiment_groups, legend_labels[:len(experiment_groups)]))

            # --- Map Colors ---
            palette = sns.color_palette("deep", n_colors=len(experiment_groups))
            color_map = dict(zip(experiment_groups, palette))

            # --- Map Linestyles ---
            non_solid_styles = ['--', '-.', ':', (0, (5, 5)), (0, (3, 1, 1, 1)), (0, (1, 10))]
            n_groups = len(experiment_groups)
            custom_linestyles = non_solid_styles[:n_groups - 1] + ['-']
            linestyle_map = dict(zip(experiment_groups, custom_linestyles))

            # --- Prepare Inset Axes for Zoom (0-3000) ---
            # Dynamically place the inset based on the task to avoid line and edge overlaps
            if "push_door" in task:
                # Move up and right (from [0.05, 0.55...])
                inset_bounds = [0.03, 0.62, 0.4, 0.35]
            elif "walk_slope" in task:
                # Move down and right (from [0.55, 0.15...])
                inset_bounds = [0.58, 0.03, 0.4, 0.35]
            else:
                # Stand Dance - keep original bottom-right placement
                inset_bounds = [0.55, 0.15, 0.4, 0.35]

            axins = ax.inset_axes(inset_bounds)

            y_min_zoom = float('inf')
            y_max_zoom = float('-inf')

            for group_name in experiment_groups:
                df_group = df_for_current[df_for_current['experiment_group'] == group_name]
                grouped_data = df_group.groupby('step')['smoothed_value'].agg(['mean', 'sem']).reset_index()

                group_label = legend_label_map.get(group_name, group_name)
                line_color = color_map.get(group_name)
                line_style = linestyle_map.get(group_name, '-')

                # Plot on Main Axis
                ax.plot(
                    grouped_data['step'],
                    grouped_data['mean'],
                    label=group_label,
                    color=line_color,
                    linestyle=line_style,
                    alpha=0.9
                )

                # Plot on Inset Axis
                axins.plot(
                    grouped_data['step'],
                    grouped_data['mean'],
                    color=line_color,
                    linestyle=line_style,
                    alpha=0.9
                )

                if "walk_slope" in task:
                    max_window_x = 5000
                else:
                    max_window_x = 3000

                # Track min/max specifically for the strictly 0-3000 step window
                zoom_data = grouped_data[(grouped_data['step'] >= 0) & (grouped_data['step'] <= max_window_x)]
                if not zoom_data.empty:
                    y_min_zoom = min(y_min_zoom, zoom_data['mean'].min())
                    y_max_zoom = max(y_max_zoom, zoom_data['mean'].max())

            # --- Format Inset Axes ---
            # Strictly enforce the 0 to max_window_x limits
            axins.set_xlim(0, max_window_x)

            if y_min_zoom != float('inf') and y_max_zoom != float('-inf'):
                margin = (y_max_zoom - y_min_zoom) * 0.05
                if margin == 0: margin = 0.1

                zoom_bottom = y_min_zoom - margin
                zoom_top = y_max_zoom + margin
                axins.set_ylim(zoom_bottom, zoom_top)

                # Lock ticks exactly to 0 and 3000 on the X axis, and min/max on the Y axis
                axins.set_xticks([0, max_window_x])
                axins.set_yticks([zoom_bottom, zoom_top])

                import matplotlib.ticker as ticker
                axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            axins.grid(True, linestyle='--', alpha=0.4)
            # axins.tick_params(axis='both', which='major', labelsize=8)
            axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False) # for removing ax labels on inset

            ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.2)

            # --- Set titles and labels for Main Axis ---
            if row_idx == 0:
                ax.set_title(task.replace('_', ' ').title(), fontsize=20)
            if col_idx == 0:
                ax.set_ylabel("Mean Episode Return" if "reward" in current_reward_tag else "Value Function Loss", fontsize=18)
            if row_idx == num_tags - 1:
                ax.set_xlabel('Training Iterations', fontsize=18)

            # --- Set main x-axis limits dynamically ---
            x_limit = max_steps_dict.get(task, 15000)
            ax.set_xlim(0, x_limit)

            # --- Main Y-axis scaling ---
            y_max_main = df_for_current['smoothed_value'].max()
            ax.set_ylim(0, y_max_main * 1.05)

            ax.grid(True, linestyle='--', alpha=0.6)

    # --- Setup Global Legend Below Figure ---
    # Extract handles and labels from the first axes that was populated
    handles, labels = axes.flat[0].get_legend_handles_labels()

    # Place it at the bottom center of the entire figure layout, spread across a single row
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.02), fontsize=18, frameon=False)

    # Adjust layout. Make room at the bottom for the new global legend using the `rect` parameter
    plt.tight_layout(rect=[0, 0.08, 0.98, 0.96])

    plot_save_path_mean_ci = os.path.join(output_dir, "combined_all_metrics_no_inlay_ax_labels.pdf")
    plt.savefig(plot_save_path_mean_ci, dpi=300)
    plt.close()

    print("\n--- Unified plot generated and saved ---")