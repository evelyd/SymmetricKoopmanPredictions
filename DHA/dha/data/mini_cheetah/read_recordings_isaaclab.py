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


def get_Rd_signals_on_kin_subchains(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations["R3"]
    rep_F = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    rep_F.name = "R3_on_legs"
    return rep_F

def compute_joint_pos_obs(q_js_ms_rel: np.ndarray, q0_isaaclab: np.ndarray, q0: np.ndarray, joint_order_indices: list):
    q_js_ms = q_js_ms_rel[:, joint_order_indices] + q0_isaaclab[joint_order_indices] + q0[7:]  # Add offset to the measurements from UMich
    cos_q_js, sin_q_js = np.cos(q_js_ms), np.sin(q_js_ms)  # convert from angle to unit circle parametrization
    # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
    q_js_unit_circle_t = np.stack([cos_q_js, sin_q_js], axis=2)
    q_js_unit_circle_t = q_js_unit_circle_t.reshape(q_js_unit_circle_t.shape[0], -1)
    joint_pos_S1 = q_js_unit_circle_t  # Joints in angle not unit circle representation
    joint_pos = q_js_ms  # Joints in angle representation
    return joint_pos_S1, joint_pos

def convert_mini_cheetah_isaaclab_recordings(data_paths: list):
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
        base z,                                  (1,) [Optional]
        base_quat,                               (4,) [Optional]
        _________________________________________________________
        TOTAL:                                    53. Dimensions
    The conversion process takes these measurements and does the following:
        1. Stores them into a DynamicsRecording format, for easy loading
        2. Defines the group representation for each observation.
        3. Changes joint position to Pinocchio convention, used by MorphoSymm.
    """
    all_data = []
    all_action_data = []
    for data_path in data_paths:
        assert data_path.exists(), f"Path {data_path.absolute()} does not exist"
        data = np.load(data_path, allow_pickle=True)
        all_data.append(np.array([traj['obs'] for traj in data]))
        all_action_data.append(np.array([traj['actions'] for traj in data]))
    print(f"Shape of all data: {all_data[0].shape}")
    state_batched = np.concatenate(all_data, axis=1)
    action_batched = np.concatenate(all_action_data, axis=1)
    # Reshape the data so that the first dimension is end to end
    state = state_batched.transpose((1,0,2)).reshape(state_batched.shape[0] * state_batched.shape[1], -1)
    action = action_batched.transpose((1,0,2)).reshape(action_batched.shape[0] * action_batched.shape[1], -1)
    num_states = state.shape[-1]
    assert num_states == 48 or num_states == 53, f"Expected 48 or 53 dimensions in the state, got {state.shape[-1]}"

    dt = 0.02  # Time step of the simulation

    # Load the Mini-Cheetah robot
    robot, G = load_symmetric_system(robot_name="mini_cheetah")
    rep_Q_js = G.representations["Q_js"]  # Representation on joint space position coordinates
    rep_TqQ_js = G.representations["TqQ_js"]  # Representation on joint space velocity coordinates
    rep_Rd = G.representations["R3"]  # Representation on vectors in R^d
    rep_Rd_pseudo = G.representations["R3_pseudo"]  # Representation on pseudo vectors in R^d
    rep_euler_xyz = G.representations["euler_xyz"]  # Representation on Euler angles
    rep_kin_three = get_kinematic_three_rep(G)  # Permutation of legs
    rep_Rd_on_limbs = get_Rd_signals_on_kin_subchains(G, rep_kin_three)  # Representation on R^3 on legs

    rep_z = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_z.name = "base_z"
    rep_xy = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[:2, :2].reshape((2, 2)) for h in G.elements if h != G.identity})
    rep_xy.name = "base_xy"
    rep_euler_z = group_rep_from_gens(G, rep_H={h: rep_euler_xyz(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_euler_z.name = "euler_z"

    # Create a mapping from current joint order to morphosymm order
    usd_joint_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    joint_order_for_morphosymm = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    joint_order_indices = [usd_joint_order.index(joint) for joint in joint_order_for_morphosymm]

    # Define the default joint positions in Isaaclab
    q0_isaaclab = np.array([0.10000000149011612, -0.10000000149011612, 0.10000000149011612, -0.10000000149011612, -0.800000011920929, -0.800000011920929, -0.800000011920929, -0.800000011920929, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716]) #TODO this is hardcoded, if the defaults change then I have to change this too

    # Define observation variables and their group representations

    # Base body observations ___________________________________________________________________________________________
    base_vel = state[:, :3]  # Rep: rep_Rd
    base_ang_vel = state[:, 3:6]  # Rep: rep_euler_xyz
    projected_gravity = state[:, 6:9] # Rep: red Rd
    ref_projected_gravity = np.array([0, 0, -1.0])  # Rep: rep_Rd
    projected_gravity_error = projected_gravity - ref_projected_gravity
    velocity_commands_xy = state[:, 9:11] # Rep: Rd for xy, euler xyz for heading? idk
    ref_base_lin_vel = np.hstack([velocity_commands_xy, np.zeros((velocity_commands_xy.shape[0], 1))]) # set ref lin z vel to 0
    base_vel_error = base_vel - ref_base_lin_vel
    velocity_commands_z = state[:, 11][:, np.newaxis]
    ref_base_ang_vel = np.hstack([np.zeros((base_ang_vel.shape[0], 2)), velocity_commands_z])
    base_ang_vel_error = base_ang_vel - ref_base_ang_vel

    if num_states == 53:
        initial_base_z = state[0, 48]  # Use the first value in a traj as the reference
        ref_base_z = initial_base_z #TODO this value is hardcoded for now to avoid needing to collect the data yet again
        base_z = state[:, 48][:, np.newaxis]  # Rep: rep_z
        base_z_error = base_z - ref_base_z
        base_ori = Rotation.from_quat(state[:, 49:53], scalar_first=True).as_euler('xyz')  # Rep: rep_euler_xyz
        # Define the representation of the rotation matrix R that transforms the base orientation.
        rep_rot_flat = {}
        # R = Rotation.from_euler("xyz", base_ori[2]).as_matrix()
        for h in G.elements:
            rep_rot_flat[h] = np.kron(rep_Rd(h), rep_Rd(~h).T)
        rep_rot_flat = escnn_representation_form_mapping(G, rep_rot_flat)
        rep_rot_flat.name = "SO(3)_flat"
        base_ori_R = np.asarray([Rotation.from_euler("xyz", ori).as_matrix() for ori in base_ori])
        base_ori_R_flat = base_ori_R.reshape(base_ori.shape[0], -1)

    # Joint-Space observations _________________________________________________________________________________________
    joint_vel = state[:, 24:36]
    # Reorder the joint velocities to match the morphosymm order
    joint_vel = joint_vel[:, joint_order_indices]
    # Joint positions need to be converted to the unit circle parametrization [cos(q), sin(q)].
    # For God’s sake, we need to avoid using PyBullet.
    bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
    robot.configure_bullet_simulation(bullet_client=bullet_client)
    # Get zero reference position.
    q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
    q_js_ms_rel = state[:, 12:24] # the raw joitn position that comes from isaaclab is the relative one (relative to isaaclab default joint pos) where qrrel = qabs - q0isaaclab
    joint_pos_S1, joint_pos = compute_joint_pos_obs(q_js_ms_rel, q0_isaaclab, q0, joint_order_indices)
    action_joint_pos = state[:, 36:48]  # Rep: rep_TqQ_js
    a_joint_pos_S1, a_joint_pos = compute_joint_pos_obs(action_joint_pos, q0_isaaclab, q0, joint_order_indices)

    # Joint-Space actions ============================================================
    current_action_S1, current_action = compute_joint_pos_obs(action, q0_isaaclab, q0, joint_order_indices)

    # Subsample the data by skippig by ignoring odd frames. ============================================================
    dt_subsample = 1
    velocity_commands_xy = velocity_commands_xy[::dt_subsample]
    velocity_commands_z = velocity_commands_z[::dt_subsample]
    if num_states == 53:
        base_z = base_z[::dt_subsample]
        base_z_error = base_z_error[::dt_subsample]
    base_vel = base_vel[::dt_subsample]
    projected_gravity = projected_gravity[::dt_subsample]
    projected_gravity_error = projected_gravity_error[::dt_subsample]
    base_vel_error = base_vel_error[::dt_subsample]
    if num_states == 53:
        base_ori = base_ori[::dt_subsample]
        base_ori_R_flat = base_ori_R_flat[::dt_subsample]
    base_ang_vel = base_ang_vel[::dt_subsample]
    base_ang_vel_error = base_ang_vel_error[::dt_subsample]
    joint_pos = joint_pos[::dt_subsample]
    joint_pos_S1 = joint_pos_S1[::dt_subsample]
    joint_vel = joint_vel[::dt_subsample]
    action_joint_pos = action_joint_pos[::dt_subsample]
    a_joint_pos_S1 = a_joint_pos_S1[::dt_subsample]
    current_action_S1 = current_action_S1[::dt_subsample]
    # Define the dataset.
    data_recording = DynamicsRecording(
        description=f"Mini Cheetah {data_path.parent.parent.stem}",
        info=dict(num_traj=1, trajectory_length=state.shape[0]),
        dynamics_parameters=dict(dt=dt * dt_subsample, group=dict(group_name=G.name, group_order=G.order())),
        recordings=dict(
            velocity_commands_xy=velocity_commands_xy[None, ...].astype(np.float32),
            velocity_commands_z=velocity_commands_z[None, ...].astype(np.float32),
            # base_z=base_z[None, ...].astype(np.float32),
            # base_z_error=base_z_error[None, ...].astype(np.float32),
            base_vel=base_vel[None, ...].astype(np.float32),
            base_vel_error=base_vel_error[None, ...].astype(np.float32),
            projected_gravity=projected_gravity[None, ...].astype(np.float32),
            projected_gravity_error=projected_gravity_error[None, ...].astype(np.float32),
            # base_ori=base_ori[None, ...].astype(np.float32),
            # base_ori_R_flat=base_ori_R_flat[None, ...].astype(np.float32),
            base_ang_vel=base_ang_vel[None, ...].astype(np.float32),
            base_ang_vel_error=base_ang_vel_error[None, ...].astype(np.float32),
            joint_pos=joint_pos[None, ...].astype(np.float32),
            joint_pos_S1=joint_pos_S1[None, ...].astype(np.float32),
            joint_vel=joint_vel[None, ...].astype(np.float32),
            action_joint_pos=action_joint_pos[None, ...].astype(np.float32),
            a_joint_pos_S1=a_joint_pos_S1[None, ...].astype(np.float32),
            current_action_S1=current_action_S1[None, ...].astype(np.float32),
        ),
        state_obs=(
            "joint_pos",
            "joint_vel",
            # "base_z",
            # "base_ori",
        ),
        action_obs=("current_action_S1",),
        obs_representations=dict(
            joint_pos=rep_TqQ_js,  # Joint-Space observations
            joint_pos_S1=rep_Q_js,  # Joint-Space position unit circle parametrization.
            joint_vel=rep_TqQ_js,
            action_joint_pos=rep_TqQ_js,
            a_joint_pos_S1=rep_Q_js,
            # Base body observations
            velocity_commands_xy=rep_xy,
            velocity_commands_z=rep_euler_z,
            base_pos=rep_Rd,
            # base_z=rep_z,
            # base_z_error=rep_z,
            base_vel=rep_Rd,
            base_vel_error=rep_Rd,
            projected_gravity=rep_Rd,
            projected_gravity_error=rep_Rd,
            # base_ori=rep_euler_xyz,
            # base_ori_R_flat=rep_rot_flat,
            base_ang_vel=rep_euler_xyz,
            base_ang_vel_error=rep_euler_xyz,
            current_action_S1=rep_Q_js,
        ),
        # Ensure the angles in the unit circle are not disturbed by the normalization.
        obs_moments=dict(
            joint_pos_S1=(
                np.zeros(joint_pos_S1.shape[-1]),
                np.ones(joint_pos_S1.shape[-1]),
            ),
            a_joint_pos_S1=(
                np.zeros(a_joint_pos_S1.shape[-1]),
                np.ones(a_joint_pos_S1.shape[-1]),
            ),
            current_action_S1=(
                np.zeros(current_action_S1.shape[-1]),
                np.ones(current_action_S1.shape[-1]),
            ),
            # base_ori_R_flat=(
                # np.zeros(base_ori_R_flat.shape[-1]),
                # np.ones(base_ori_R_flat.shape[-1]),
            # ),
        ),
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
    terrains = ["curriculum"] #, "uneven_easy", "uneven_medium", "uneven_hard_squares"]
    modes = ["2025-05-16_16-16-41"]
    # modes = ["2025-05-16_16-22-18"]
    for terrain in terrains:
        for mode in modes:
            data_paths = list(Path(f"data/mini_cheetah/isaaclab_recordings/{terrain}/{mode}/raw_recording").glob("*.npy"))
            convert_mini_cheetah_isaaclab_recordings(data_paths)
