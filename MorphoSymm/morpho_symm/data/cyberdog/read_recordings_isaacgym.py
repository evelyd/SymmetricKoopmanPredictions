from pathlib import Path

import numpy as np
import pybullet
from escnn.group import Group, Representation
from morpho_symm.data.DynamicsRecording import DynamicsRecording, split_train_val_test
from morpho_symm.utils.algebra_utils import permutation_matrix
from morpho_symm.utils.rep_theory_utils import escnn_representation_form_mapping, group_rep_from_gens
from morpho_symm.utils.robot_utils import load_symmetric_system
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation
from hydra import compose, initialize

def get_kinematic_three_rep(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(4, dtype=int)}
    gens = [permutation_matrix([1, 0, 3, 2]), permutation_matrix([2, 3, 0, 1]), permutation_matrix([0, 1, 2, 3])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three

def get_ground_reaction_forces_rep(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations['Rd']
    rep_F = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    rep_F.name = "R3_on_legs"
    return rep_F

def get_kinematic_three_rep_two(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(2, dtype=int)}
    gens = [permutation_matrix([1, 0])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three

def get_ground_reaction_forces_rep_two(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations['Rd']
    rep_F = {G.identity: np.eye(6, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    rep_F.name = "R3_on_front_legs"
    return rep_F

def get_friction_rep(G: Group, rep_kin_three: Representation):
    rep_friction = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(np.kron(np.eye(2,dtype=int), rep_kin_three(g)), np.eye(3,dtype=int))
             for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_friction[h] = rep_h

    rep_friction = group_rep_from_gens(G, rep_friction)
    rep_friction.name = "friction_on_legs"
    return rep_friction



def convert_cyberdog2_isaacgym_recordings(data_paths: list):
    """Convertion script for the recordings of observations from the Mini-Cheetah Robot.

    This function takes recordings stored into a single numpy array of shape (time, state_dim) where the state is
    defined as [state]:
        base_velocity,                           (3,)
        base_angular_velocity,                   (3,)
        projected_gravity,                       (3,)
        velocity_commands,                       (3,)
        joint position,                          (12,)
        joint velocities,                        (12,)
        actions,                                 (12,)
        base z,                                  (1,)
        base_quat,                               (4,)
        _________________________________________________________
        TOTAL:                                    53. Dimensions
    The conversion process takes these measurements and does the following:
        1. Stores them into a DynamicsRecording format, for easy loading
        2. Defines the group representation for each observation.
        3. Changes joint position to Pinocchio convention, used by MorphoSymm.
    """
    """Convertion script for the recordings of observations from the Mini-Cheetah Robot.

    This function takes recordings stored into a single numpy array of shape (time, state_dim) where the state is
    defined as [state]:
        projected_gravity,                       (3,)
        projected_forward_vec,                   (3,)
        velocity commands,                       (3,)
        dof_pos,                                 (12,)
        dof_vel,                                 (12,)
        action (past),                           (12,)
        clock_inputs,                            (2,)

        _________________________________________________________
        TOTAL:                                    47. Dimensions
    The conversion process takes these measurements and does the following:
        1. Stores them into a DynamicsRecording format, for easy loading
        2. Defines the group representation for each observation.
        3. Changes joint position to Pinocchio convention, used by MorphoSymm.
    """
    all_obs = []
    for data_path in data_paths:
        data = np.load(data_path, allow_pickle=True)
        all_obs.append(np.array([traj['obs'] for traj in data]))
    print(f"Loaded {len(all_obs)} files. Shape: {all_obs[0].shape}")

    # 拼接所有 traj 的 obs
    # obs_batched = np.concatenate(all_obs, axis=1)  # (num_traj, traj_len, obs_dim)
    # obs_flat = obs_batched.transpose((1, 0, 2)).reshape(obs_batched.shape[0]*obs_batched.shape[1], -1)
    # all_obs 是一个 list，每个元素是 (N, 141) 的 array
    obs_flat = np.concatenate(all_obs, axis=0)  # shape: (total_frames, 141)
    assert obs_flat.shape[-1] == 141, f"Expect 141D obs (3 x 47), got {obs_flat.shape[-1]}"

    # 每帧中提取当前帧（最后 47 维）
    state = obs_flat[:, -47:]
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path="../../cfg/robot", version_base='1.3')
    robot_name = 'a1'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
    robot_cfg = compose(config_name=f"{robot_name}.yaml")
    robot, G = load_symmetric_system(robot_cfg=robot_cfg)

    rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
    rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
    rep_O3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
    rep_O3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3
    trivial_rep = G.trivial_representation
    rep_kin_three = get_kinematic_three_rep_two(G)
    rep_friction = get_friction_rep(G, rep_kin_three)

    # 获取关节顺序
    joint_order = [
        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    print(f"robot type = {type(robot)}, G type = {type(G)}")
    default_joint_angles = {
        'FL_hip_joint': 0.0,
        'FR_hip_joint': 0.0,
        'RL_hip_joint': 0.0,
        'RR_hip_joint': 0.0,
        'FL_thigh_joint': -1.396,  # -80 deg
        'FR_thigh_joint': -1.396,
        'RL_thigh_joint': -1.396,
        'RR_thigh_joint': -1.396,
        'FL_calf_joint': 2.356,    # 135 deg
        'FR_calf_joint': 2.356,
        'RL_calf_joint': 2.356,
        'RR_calf_joint': 2.356,
    }
    default_dof_pos = np.array([default_joint_angles[j] for j in joint_order])  # (12,)

    # 恢复 absolute joint_pos
    joint_pos_rel = state[:, 9:21]  # 12D
    joint_pos = joint_pos_rel + default_dof_pos[np.newaxis, :]

    # joint_vel, actions
    joint_vel = state[:, 21:33]  # 12D
    actions = state[:, 33:45]    # 12D

    # 其他观测量（含 projected_gravity, projected_forward_vec, command, etc.）
    projected_gravity = state[:, 0:3]
    projected_forward_vec = state[:, 3:6]
    commands = state[:, 6:9]
    clock_inputs = state[:, 45:47]

    # 建立 DynamicsRecording
    dt = 0.02
    data_recording = DynamicsRecording(
        description="CyberDog2 Observation Only",
        info=dict(num_traj=len(data_paths), trajectory_length=state.shape[0]),
        dynamics_parameters=dict(dt=dt, group=dict(group_name=G.name, group_order=G.order())),
        recordings=dict(
            joint_pos=joint_pos[None, ...].astype(np.float32),
            joint_vel=joint_vel[None, ...].astype(np.float32),
            actions=actions[None, ...].astype(np.float32),
            projected_gravity=projected_gravity[None, ...].astype(np.float32),
            projected_forward_vec=projected_forward_vec[None, ...].astype(np.float32),
            commands=commands[None, ...].astype(np.float32),
            clock_inputs=clock_inputs[None, ...].astype(np.float32),
        ),
        state_obs=("projected_gravity", "projected_forward_vec", "commands","joint_pos", "joint_vel", "actions","clock_inputs"),
        action_obs=("actions",),
        obs_representations=dict(
            projected_gravity=rep_O3,
            projected_forward_vec=rep_O3,
            commands=rep_O3,
            joint_pos=rep_TqQJ,
            joint_vel=rep_TqQJ,
            actions=rep_TqQJ,
            clock_inputs=rep_kin_three,
        )
    )

    # Compute the mean and variance of all observations considering symmetry constraints.
    for obs_name in data_recording.recordings.keys():
        if obs_name in data_recording.obs_moments:
            continue
        data_recording.compute_obs_moments(obs_name=obs_name)

    train_dr, val_dr, test_dr = split_train_val_test(
        data_recording, partition_sizes=(0.7, 0.15, 0.15), split_dimension="time" # needed for Orange data
        # data_recording, partition_sizes=(1/3, 1/3, 1/3), split_dimension="time" # needed for Purple data
    )

    for dr, p_name in zip([train_dr, val_dr, test_dr], ["train", "val", "test"]):
        file_name = f"n_trajs={dr.info['num_traj']}-frames={dr.info['trajectory_length']}-{p_name}.pkl"
        dr.save_to_file(data_path.parent.parent / file_name)
        print(f"{p_name} Dynamics Recording saved to {data_path.parent.parent / file_name}")


if __name__ == "__main__":
    task = "stand_dance_cyber_aug" #, "uneven_easy", "uneven_medium", "uneven_hard_squares"]
    # modes = ["2025-05-16_16-16-41"]
    modes = "20250521_203857"
    # for terrain in task:
    #     for mode in modes:
    data_paths = list(Path(f"legged_gym/isaacgym_recordings/{task}/{modes}").glob("*.npy"))
    print(f"[INFO] Searching for .npy files in: {Path(f'legged_gym/isaacgym_recordings/{task}/{modes}').resolve()}")
    print(f"[INFO] Found {len(data_paths)} .npy files.")
    convert_cyberdog2_isaacgym_recordings(data_paths)
