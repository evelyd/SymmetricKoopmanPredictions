#!/bin/bash

# Usage: ./generate_data.sh <task>

if [ -z "$1" ]; then
    echo "Usage: $0 <task>"
    exit 1
fi

task="$1"

# Find all subdirectories with the task in their name
for subdir in logs/*"${task}"*/; do

    # if [[ "$(basename "$subdir")" != "push_door_cyber_cdae_online_next_latent" ]] && [[ "$(basename "$subdir")" != "push_door_cyber_emlp_ecdae_online_next_latent" ]]; then
    #     continue
    # fi
    # if [[ "$(basename "$subdir")" != "stand_dance_cyber_emlp_ecdae_online_next_latent" ]]; then
    #     continue
    # fi

    # if [ "$subdir" != *"rff"* ]; then
    #     continue
    # fi

    # if [[ "$(basename "$subdir")" == "push_door_cyber_cdae" ]]; then
    #     ckpt=20100
    # elif [ "$task" == "push_door" ]; then
    #     ckpt=20000
    # else
    #     ckpt=30000
    # fi

    if [[ "$subdir" =~ _cyber_(.*)/$ ]]; then
        suffix="_${BASH_REMATCH[1]}"
    else
        suffix=""
    fi
    echo "Processing directory: $subdir with suffix: $suffix"
    # Iterate through all matching directories for this task+suffix
    for dir in "${subdir}"202*/; do
        echo "$(basename "$dir")"

        # Skip other subdirs, only do one from each method
        # if [[ "$(basename "$dir")" != "2026-02-01-20-59-56_" ]] && [[ "$(basename "$dir")" != "2026-02-22-23-40-22_" ]] && [[ "$(basename "$dir")" != "2026-02-20-08-00-08_" ]] && [[ "$(basename "$dir")" != "2026-02-25-12-17-07_" ]]; then
        #     continue
        # fi
        if [[ "$(basename "$dir")" != "2026-01-24-14-57-11_" ]]; then
            continue
        fi
        if [[ "$(basename "$dir")" == "2026-02-02-11-56-36_" ]]; then
            ckpt=26100
        elif [[ "$(basename "$dir")" == "2026-02-02-17-28-57_" ]]; then
            ckpt=22800
        elif [[ "$(basename "$dir")" == "2026-02-18-13-20-34_" ]]; then
            ckpt=25200
        elif [[ "$(basename "$dir")" == "2026-02-18-13-25-03_" ]]; then
            ckpt=24600
        elif [[ "$(basename "$dir")" == "2026-02-15-20-53-00_" ]]; then
            ckpt=26400
        elif [[ "$(basename "$dir")" == "2025-11-08-10-42-55_" ]] || [[ "$(basename "$dir")" == "2025-11-08-10-45-22_" ]]; then
            ckpt=19700
        elif [[ "$(basename "$dir")" == "2025-11-08-10-47-40_" ]]; then
            ckpt=19500
        elif [[ "$(basename "$dir")" == "2025-09-24-13-08-31_" ]]; then
            ckpt=18700
        else
            ckpt=20000
        fi

        if [ -d "$dir" ] && [ "$dir" != *"cyber"* ]; then
            echo "Found directory: $dir"
            python legged_gym/scripts/play_and_save.py \
                --task="cyber2_${task}${suffix}" \
                --headless \
                --right \
                --load_run "$(basename "$dir")" \
                --checkpoint "$ckpt"
            if [[ "$(basename "$subdir")" == *"push_door"* ]]; then
                python legged_gym/scripts/play_and_save.py \
                    --task="cyber2_${task}${suffix}" \
                    --headless \
                    --left \
                    --load_run "$(basename "$dir")" \
                    --checkpoint "$ckpt"
            fi
        fi
    done
done

# Processing directory: isaacgym_recordings/push_door_cyber/ with suffix:
# Found directory: isaacgym_recordings/push_door_cyber/2025-06-24-11-36-27_/
# Found directory: isaacgym_recordings/push_door_cyber/2025-07-19-10-55-36_/
# Found directory: isaacgym_recordings/push_door_cyber/2025-07-19-10-55-49_/
# Processing directory: isaacgym_recordings/push_door_cyber_cdae/ with suffix: _cdae
# Found directory: isaacgym_recordings/push_door_cyber_cdae/2025-06-26-10-39-40_/
# Found directory: isaacgym_recordings/push_door_cyber_cdae/2025-07-21-10-06-58_/
# Found directory: isaacgym_recordings/push_door_cyber_cdae/2025-07-21-10-38-56_/
# Processing directory: isaacgym_recordings/push_door_cyber_cdae_online/ with suffix: _cdae_online
# Found directory: isaacgym_recordings/push_door_cyber_cdae_online/2025-08-12-16-04-52_/
# Found directory: isaacgym_recordings/push_door_cyber_cdae_online/2025-08-17-04-10-30_/
# Found directory: isaacgym_recordings/push_door_cyber_cdae_online/2025-08-17-06-05-39_/
# Processing directory: isaacgym_recordings/push_door_cyber_emlp/ with suffix: _emlp
# Found directory: isaacgym_recordings/push_door_cyber_emlp/2025-07-17-15-17-32_/
# Found directory: isaacgym_recordings/push_door_cyber_emlp/2025-07-21-11-33-44_/
# Found directory: isaacgym_recordings/push_door_cyber_emlp/2025-07-21-11-33-54_/
# Processing directory: isaacgym_recordings/push_door_cyber_emlp_ecdae/ with suffix: _emlp_ecdae
# Found directory: isaacgym_recordings/push_door_cyber_emlp_ecdae/2025-06-26-10-45-16_/
# Found directory: isaacgym_recordings/push_door_cyber_emlp_ecdae/2025-07-22-10-27-03_/
# Found directory: isaacgym_recordings/push_door_cyber_emlp_ecdae/2025-07-22-10-27-24_/
# Found directory: isaacgym_recordings/push_door_cyber_emlp_ecdae/2025-07-28-12-53-37_/
# Processing directory: isaacgym_recordings/push_door_cyber_emlp_ecdae_online/ with suffix: _emlp_ecdae_online
# Found directory: isaacgym_recordings/push_door_cyber_emlp_ecdae_online/2025-08-28-13-20-14_/
# Found directory: isaacgym_recordings/push_door_cyber_emlp_ecdae_online/2025-08-28-13-20-47_/
# Found directory: isaacgym_recordings/push_door_cyber_emlp_ecdae_online/2025-08-28-13-22-11_/