import os
import argparse
import sys
import xml.etree.ElementTree as ET
import math

# Parse args early to catch headless mode before loading MuJoCo
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--headless', action='store_true')
early_args, _ = parser.parse_known_args()

if early_args.headless:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import mujoco
import imageio
from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR
from matplotlib import pyplot as plt

plt.rcParams.update({'font.family':'serif'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def get_heading(q):
    forward_vec = np.array([1., 0., 0.])
    xyzw_q = np.array([q[1], q[2], q[3], q[0]])
    xyzw_q[:2] = 0.0
    xyzw_q = xyzw_q / np.linalg.norm(xyzw_q)

    xyz = xyzw_q[:3]
    t = np.cross(xyz, forward_vec) * 2.0
    heading_vec = forward_vec + xyzw_q[3] * t + np.cross(xyz, t)
    heading = np.arctan2(heading_vec[1], heading_vec[0])
    return wrap_to_pi(heading)

# --- Configuration ---
class CyberSimConfig:
    dt = 0.005
    decimation = 4
    num_dofs = 12
    num_history = 3

    init_pos = [0.0, 0.0, 0.11]
    init_quat = [1.0, 0.0, 0.0, 0.0] # WXYZ

    init_joints = {
        'FL_hip_joint': 0.0, 'FR_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'RR_hip_joint': 0.0,
        'FL_thigh_joint': -80 / 57.3, 'FR_thigh_joint': -80 / 57.3, 'RL_thigh_joint': -80 / 57.3, 'RR_thigh_joint': -80 / 57.3,
        'FL_calf_joint': 135 / 57.3, 'FR_calf_joint': 135 / 57.3, 'RL_calf_joint': 135 / 57.3, 'RR_calf_joint': 135 / 57.3
    }

    default_joints = {
        'FL_hip_joint': 0.0, 'FR_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'RR_hip_joint': 0.0,
        'FL_thigh_joint': -45 / 57.3, 'FR_thigh_joint': -45 / 57.3, 'RL_thigh_joint': -45 / 57.3, 'RR_thigh_joint': -45 / 57.3,
        'FL_calf_joint': 70 / 57.3, 'FR_calf_joint': 70 / 57.3, 'RL_calf_joint': 70 / 57.3, 'RR_calf_joint': 70 / 57.3
    }

    joint_names_ordered = [
        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
    ]

    stiffness = 30.0
    damping = 3.0
    action_scale = 0.25
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.0
    cmd_scale = [2.0, 2.0, 0.25]
    gait_freq = 2.5
    torque_limit = 12.0001

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="cyber2_stand_dance", help='Task name')
    parser.add_argument('--load_run', type=str, default=None, help='Path to policy run dir (Required if not using --eval_success)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint number (e.g., 20000). If omitted, finds the highest one.')
    parser.add_argument('--xml_path', type=str, default=f"{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2.xml", help='Path to converted MuJoCo XML')
    parser.add_argument('--num_steps', type=int, default=2000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--friction', type=float, default=1.0)
    parser.add_argument('--mass_offset', type=float, default=0.0)
    parser.add_argument('--push_vel', type=float, default=0.0)
    parser.add_argument('--armature', type=float, default=0.005)
    parser.add_argument('--left', action='store_true', help='Mirror the door to the left side')
    parser.add_argument('--eval_success', action='store_true', help='Evaluate success rate over all runs in the task directory')
    parser.add_argument('--num_eval_runs', type=int, default=100, help='Number of runs for evaluation')
    args, _ = parser.parse_known_args()
    return args

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def rotate_vec_by_inverse_quat(vec, q):
    R = quaternion_to_rotation_matrix(q)
    return R.T @ vec

class GaitGenerator:
    def __init__(self, freq=2.5, dt=0.005):
        self.dt = dt
        self.freq = freq
        self.gait_indices = 0.0
        self.phase = 0.5
        self.offset = 0.0
        self.bound = 0.0

    def step(self):
        self.gait_indices = (self.gait_indices + self.dt * self.freq) % 1.0
        foot_indices = np.array([
            self.gait_indices + self.phase + self.offset + self.bound,
            self.gait_indices + self.offset,
            self.gait_indices + self.bound,
            self.gait_indices + self.phase
        ])
        return np.sin(2 * np.pi * foot_indices)

class CyberEnvShim:
    def __init__(self, cfg, model, task):
        self.cfg = cfg
        self.history = deque(maxlen=cfg.num_history)

        self.qpos_indices = []
        self.dof_indices = []

        print("Mapping joints...")
        for name in cfg.joint_names_ordered:
            try:
                j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if j_id == -1:
                    fallback = name.replace("_joint", "")
                    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fallback)
                    if j_id == -1: raise ValueError(f"Joint {name} not found")

                self.qpos_indices.append(model.jnt_qposadr[j_id])
                self.dof_indices.append(model.jnt_dofadr[j_id])
            except Exception as e:
                print(f"Error mapping joint {name}: {e}")
                sys.exit(1)

        self.qpos_indices = np.array(self.qpos_indices, dtype=int)
        self.dof_indices = np.array(self.dof_indices, dtype=int)

        if "push_door" in task:
            self.noise_scale_vec = np.zeros(43)
            self.noise_scale_vec[0:3] = 0.05
            self.noise_scale_vec[3:6] = 0.05
            self.noise_scale_vec[6:18] = 0.01
            self.noise_scale_vec[32:] = 0.5 #1.0
        else:
            self.noise_scale_vec = np.zeros(47)
            self.noise_scale_vec[0:3] = 0.05
            self.noise_scale_vec[3:6] = 0.05
            self.noise_scale_vec[9:21] = 0.01
            self.noise_scale_vec[21:33] = 0.0

        self.default_dof_pos = np.array([cfg.default_joints[name] for name in cfg.joint_names_ordered])
        self.gait_gen = GaitGenerator(freq=cfg.gait_freq, dt=cfg.dt * cfg.decimation)

    def update_command_heading(self, data, command_base, target_heading):
        current_quat = data.qpos[3:7]
        current_heading = get_heading(current_quat)

        heading_error = wrap_to_pi(target_heading - current_heading)
        clip_ang_vel = 0.25 * np.pi
        scale = 0.5 * np.pi / clip_ang_vel
        yaw_vel = np.clip(0.5 * heading_error, -clip_ang_vel, clip_ang_vel) * scale

        new_cmd = command_base.copy()
        new_cmd[2] = yaw_vel
        return new_cmd

    def get_obs(self, data, command, last_actions, task="stand_dance", is_left=False, add_noise=True):
        base_quat = data.qpos[3:7]
        proj_gravity = rotate_vec_by_inverse_quat(np.array([0., 0., -1.]), base_quat)
        proj_forward = rotate_vec_by_inverse_quat(np.array([1., 0., 0.]), base_quat)

        dof_pos = data.qpos[self.qpos_indices]
        dof_vel = data.qvel[self.dof_indices]
        dof_pos_err = (dof_pos - self.default_dof_pos) * self.cfg.dof_pos_scale

        if "push_door" in task:
            phase = self.gait_gen.gait_indices
            phase_val = phase if phase <= 0.5 else 0.5 - phase

            base_pos = data.qpos[0:3]
            door_normal = np.array([-1.0, 0.0, 0.0])

            if is_left:
                door_corner = np.array([0.48, -0.425, 0.0])
                lr_vec = np.array([0.0, 1.0])
            else:
                door_corner = np.array([0.48, 0.425, 0.0])
                lr_vec = np.array([1.0, 0.0])

            current_obs = np.concatenate([
                proj_gravity, proj_forward, dof_pos_err, last_actions,
                np.array([phase_val * 2, -phase_val * 2]),
                base_pos, door_corner, door_normal, lr_vec
            ])
        else:
            obs_cmd = command.copy()
            obs_cmd[0], obs_cmd[1] = obs_cmd[1], obs_cmd[0]
            obs_cmd = obs_cmd * np.array(self.cfg.cmd_scale)
            dof_vel_scaled = dof_vel * self.cfg.dof_vel_scale
            clock_full = self.gait_gen.step()
            obs_clock = clock_full[2:]

            current_obs = np.concatenate([
                proj_gravity, proj_forward, obs_cmd, dof_pos_err, dof_vel_scaled, last_actions, obs_clock
            ])

        if add_noise:
            noise = (2 * np.random.rand(len(current_obs)) - 1) * self.noise_scale_vec
            current_obs += noise

        hist_len = 5 if "push_door" in task else 3
        if len(self.history) == 0:
            for _ in range(hist_len):
                self.history.append(np.zeros_like(current_obs))
        self.history.append(current_obs)

        while len(self.history) > hist_len:
            self.history.popleft()

        return torch.tensor(np.concatenate(list(self.history)), dtype=torch.float32)

def play_mujoco():
    args = get_args()
    cfg = CyberSimConfig()

    if not args.eval_success and args.load_run is None:
        print("Error: --load_run is required when --eval_success is not provided.")
        sys.exit(1)

    if "push_door" in args.task:
        cfg.num_history = 5

    print(f"Loading model: {args.xml_path}")
    if not os.path.exists(args.xml_path):
        print(f"Error: XML file not found at {args.xml_path}")
        return

    tree = ET.parse(args.xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    if "walk_slope" in args.task:
        cfg.init_pos = [0.0, 0.0, 0.18]
        slope_start_x = 1.0
        incline_rad = 0.1
        L = 20.0
        H = 0.1

        pos_x = slope_start_x + L * math.cos(incline_rad) + H * math.sin(incline_rad)
        pos_z = L * math.sin(incline_rad) - H * math.cos(incline_rad)

        ramp = ET.Element("geom", {
            "name": "ramp",
            "type": "box",
            "size": f"{L} 5 {H}",
            "pos": f"{pos_x:.4f} 0 {pos_z:.4f}",
            "euler": f"0 {-incline_rad} 0",
            "rgba": "0.6 0.4 0.3 1",
            "friction": f"{args.friction} {args.friction}"
        })
        if worldbody is not None:
            worldbody.append(ramp)

        for cam in root.iter('camera'):
            if cam.get('name') == 'static_isaac_view':
                cam.set('pos', '-2.0 0.0 1.0')

    elif "push_door" in args.task:
        if args.left:
            door_pos = "0.5 -0.45 1.42"
            door_quat = "0 0.7071 0.7071 0"
        else:
            door_pos = "0.5 0.45 1.42"
            door_quat = "0.7071 0 0 -0.7071"

        door_mjcf = ET.fromstring(f'''
        <body name="door_assembly" pos="{door_pos}" quat="{door_quat}">
            <geom name="left_frame" type="box" size="0.025 0.03 1.42" pos="0 0 0" rgba="0.2 0.2 0.2 1"/>
            <geom name="right_frame" type="box" size="0.025 0.03 1.42" pos="0.97 0 0" rgba="0.2 0.2 0.2 1"/>
            <body name="door_panel" pos="0.025 0.03 0">
                <joint name="door_hinge" type="hinge" axis="0 0 1" range="0 1.5708" damping="0.05" frictionloss="0.02"/>
                <geom name="panel" type="box" size="0.45 0.025 1.4" pos="0.425 -0.025 0" rgba="0.6 0.8 1.0 0.5" mass="20" friction="2.0 0.005 0.0001"/>
            </body>
        </body>
        ''')
        if worldbody is not None:
            worldbody.append(door_mjcf)

        contact_block = root.find('contact')
        if contact_block is None:
            contact_block = ET.SubElement(root, 'contact')
        ET.SubElement(contact_block, 'exclude', {'body1': 'door_assembly', 'body2': 'door_panel'})

        for cam in root.iter('camera'):
            if cam.get('name') == 'static_isaac_view':
                cam.set('pos', '-1.0 1.0 1.0')

    xml_string = ET.tostring(root, encoding='unicode')

    original_cwd = os.getcwd()
    xml_dir = os.path.dirname(os.path.abspath(args.xml_path))
    os.chdir(xml_dir)

    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
    finally:
        os.chdir(original_cwd)

    model.opt.timestep = CyberSimConfig.dt
    data = mujoco.MjData(model)

    model.geom_friction[:, 0] = args.friction
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    model.body_mass[body_id] += args.mass_offset
    model.body_inertia[body_id] += args.mass_offset
    model.dof_armature[:] = args.armature
    mujoco.mj_setConst(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    sim_env = CyberEnvShim(cfg, model, args.task)

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

    exp_dir = f"{LEGGED_GYM_ROOT_DIR}/logs/{exp_name}"

    if args.eval_success and args.load_run is None:
        if not os.path.exists(exp_dir):
            print(f"Error: Experiment directory {exp_dir} not found.")
            return
        # Get all subdirectories (runs)
        run_dirs = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))])
    else:
        # Respect the specific load_run if one is provided
        run_dirs = [args.load_run]

    door_hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
    door_qpos_idx = model.jnt_qposadr[door_hinge_id] if door_hinge_id != -1 else -1

    # To track overall results if evaluating multiple runs
    eval_results = {}
    current_checkpoint = args.checkpoint

    for current_run in run_dirs:
        run_dir_path = os.path.join(exp_dir, current_run)

        # Dynamically find the highest checkpoint if not explicitly provided
        if args.checkpoint is None:
            highest_ckpt = -1
            if os.path.exists(run_dir_path):
                for f in os.listdir(run_dir_path):
                    if f.startswith("model_") and f.endswith(".pt"):
                        try:
                            num = int(f.replace("model_", "").replace(".pt", ""))
                            if num > highest_ckpt:
                                highest_ckpt = num
                        except ValueError:
                            pass

            if highest_ckpt == -1:
                print(f"Skipping {current_run}: No model_*.pt files found.")
                continue

            current_checkpoint = str(highest_ckpt)
        else:
            current_checkpoint = args.checkpoint

        policy_path = os.path.join(run_dir_path, f"model_{current_checkpoint}.pt")

        if not os.path.exists(policy_path):
            if args.eval_success:
                print(f"Skipping {current_run}: Policy not found at {policy_path}")
                continue
            else:
                print(f"Error: Policy Load Error: {policy_path} does not exist.")
                return

        print(f"\n=======================================")
        print(f"Loading Policy Run: {current_run} | Checkpoint: {current_checkpoint}")
        print(f"=======================================")

        try:
            loaded_dict = torch.load(policy_path, map_location='cpu')
            weights = loaded_dict['model_state_dict'] if 'model_state_dict' in loaded_dict else loaded_dict

            if 'actor.0.weight' in weights:
                layer_dims = [
                    weights['actor.0.weight'].shape[1],
                    weights['actor.0.weight'].shape[0],
                    weights['actor.2.weight'].shape[0],
                    weights['actor.4.weight'].shape[0],
                    weights['actor.6.weight'].shape[0]
                ]
                is_escnn = False
            else:
                l0_key = next(k for k in weights.keys() if k.startswith('actor.net.linear_0') and k.endswith('.matrix'))
                l1_key = next(k for k in weights.keys() if k.startswith('actor.net.linear_1') and k.endswith('.matrix'))
                l2_key = next(k for k in weights.keys() if k.startswith('actor.net.linear_2') and k.endswith('.matrix'))
                l3_key = next(k for k in weights.keys() if k.startswith('actor.net.linear_3') and k.endswith('.matrix'))

                layer_dims = [
                    weights[l0_key].shape[1],
                    weights[l0_key].shape[0],
                    weights[l1_key].shape[0],
                    weights[l2_key].shape[0],
                    weights[l3_key].shape[0]
                ]
                is_escnn = True

            class SimpleActor(torch.nn.Module):
                def __init__(self, dims):
                    super().__init__()
                    self.actor = torch.nn.Sequential(
                        torch.nn.Linear(dims[0], dims[1]), torch.nn.ELU(),
                        torch.nn.Linear(dims[1], dims[2]), torch.nn.ELU(),
                        torch.nn.Linear(dims[2], dims[3]), torch.nn.ELU(),
                        torch.nn.Linear(dims[3], dims[4])
                    )
                def forward(self, x): return self.actor(x)

            policy = SimpleActor(layer_dims)

            with torch.no_grad():
                if not is_escnn:
                    policy.actor[0].weight.copy_(weights['actor.0.weight'])
                    policy.actor[0].bias.copy_(weights['actor.0.bias'])
                    policy.actor[2].weight.copy_(weights['actor.2.weight'])
                    policy.actor[2].bias.copy_(weights['actor.2.bias'])
                    policy.actor[4].weight.copy_(weights['actor.4.weight'])
                    policy.actor[4].bias.copy_(weights['actor.4.bias'])
                    policy.actor[6].weight.copy_(weights['actor.6.weight'])
                    policy.actor[6].bias.copy_(weights['actor.6.bias'])
                else:
                    policy.actor[0].weight.copy_(weights[l0_key])
                    policy.actor[0].bias.copy_(weights[l0_key.replace('.matrix', '.expanded_bias')])
                    policy.actor[2].weight.copy_(weights[l1_key])
                    policy.actor[2].bias.copy_(weights[l1_key.replace('.matrix', '.expanded_bias')])
                    policy.actor[4].weight.copy_(weights[l2_key])
                    policy.actor[4].bias.copy_(weights[l2_key.replace('.matrix', '.expanded_bias')])
                    policy.actor[6].weight.copy_(weights[l3_key])
                    policy.actor[6].bias.copy_(weights[l3_key.replace('.matrix', '.expanded_bias')])

            print(f"[✓] Policy {current_run} loaded successfully!")

        except Exception as e:
            print(f"Policy Load Error for {current_run}: {e}")
            continue

        num_runs = args.num_eval_runs if args.eval_success else 1
        success_count = 0

        for run in range(num_runs):
            mujoco.mj_resetData(model, data)
            data.qpos[:3] = cfg.init_pos
            data.qpos[3:7] = cfg.init_quat
            init_vec = np.array([cfg.init_joints[name] for name in cfg.joint_names_ordered])
            data.qpos[sim_env.qpos_indices] = init_vec

            sim_env.history.clear()
            sim_env.gait_gen = GaitGenerator(freq=cfg.gait_freq, dt=cfg.dt * cfg.decimation)

            if "walk_slope" in args.task:
                target_heading = 0.0
                command = np.array([0.3, 0.0, 0.0])
            else:
                target_heading = 0.5 * np.pi
                command = np.array([0.0, 0.0, 0.0])

            last_actions = np.zeros(12)
            run_success = False

            cmd_heading, cmd_yaw_vels, actual_yaw_vels, current_headings = [], [], [], []
            base_x_positions, base_y_positions, data_log = [], [], []
            cmd_x_vels, cmd_y_vels, actual_x_vels, actual_y_vels, frames = [], [], [], [], []

            for i in range(args.num_steps):
                if "stand_dance" in args.task:
                    if i % 500 == 0 and i > 0:
                        target_heading += 0.5 * np.pi

                if args.push_vel > 0.0 and i % 1000 == 0 and i > 0:
                    data.qvel[0] += np.random.uniform(-args.push_vel, args.push_vel)
                    data.qvel[1] += np.random.uniform(-args.push_vel, args.push_vel)
                    mujoco.mj_forward(model, data)

                if not args.eval_success:
                    cmd_x_vels.append(command[0])
                    cmd_y_vels.append(command[1])

                base_quat = data.qpos[3:7]
                dynamic_command = sim_env.update_command_heading(data, command, target_heading)

                # The add_noise=True injects variance natively when evaluating
                obs = sim_env.get_obs(data, dynamic_command, last_actions, task=args.task, is_left=args.left, add_noise=True)

                if not args.eval_success:
                    cmd_yaw_vels.append(dynamic_command[2])
                    base_ang_vel = rotate_vec_by_inverse_quat(data.qvel[3:6], base_quat)
                    base_lin_x_vel = rotate_vec_by_inverse_quat(data.qvel[0:3], base_quat)[0]
                    base_lin_y_vel = rotate_vec_by_inverse_quat(data.qvel[0:3], base_quat)[1]
                    actual_yaw_vels.append(base_ang_vel[2])
                    actual_x_vels.append(base_lin_x_vel)
                    actual_y_vels.append(base_lin_y_vel)
                    cur_heading = get_heading(base_quat)
                    current_headings.append(cur_heading)
                    cmd_heading.append(target_heading)
                    base_x_positions.append(data.qpos[0])
                    base_y_positions.append(data.qpos[1])
                    data_log.append({
                        "obs": obs.cpu().numpy(),
                        "desired_heading": target_heading,
                        "actual_heading": cur_heading
                    })

                with torch.no_grad():
                    actions = policy(obs).numpy()

                for _ in range(cfg.decimation):
                    target_pos = sim_env.default_dof_pos + cfg.action_scale * actions
                    q_pos = data.qpos[sim_env.qpos_indices]
                    q_vel = data.qvel[sim_env.dof_indices]

                    torques = cfg.stiffness * (target_pos - q_pos) - cfg.damping * q_vel
                    torques = np.clip(torques, -cfg.torque_limit, cfg.torque_limit)

                    data.qfrc_applied[sim_env.dof_indices] = torques
                    data.ctrl[:] = 0.0
                    mujoco.mj_step(model, data)

                last_actions = actions

                # --- Hand Down Termination (Matches Isaac Gym hand_down_condition) ---
                if i > 80:
                    fl_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FL_foot")
                    fr_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_foot")

                    # Fallback to calf bodies just in case the URDF doesn't have explicit foot links
                    if fl_foot_id == -1: fl_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FL_calf")
                    if fr_foot_id == -1: fr_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")

                    fl_z = data.xpos[fl_foot_id][2]
                    fr_z = data.xpos[fr_foot_id][2]

                    if fl_z < 0.02 or fr_z < 0.02:
                        run_success = False
                        if not args.eval_success:
                            print(f"Failed: Front feet touched ground (FL_z={fl_z:.3f}, FR_z={fr_z:.3f}) at step {i}")
                        break # Terminate episode early
                # ---------------------------------------------------------------------

                if "push_door" in args.task and door_qpos_idx != -1:
                    door_angle = data.qpos[door_qpos_idx]
                    base_x = data.qpos[0]
                    start_x = cfg.init_pos[0]

                    cond1 = (base_x - start_x > 2.0) and (door_angle > np.pi / 3)
                    cond2 = door_angle > (np.pi / 2 - 0.1)

                    if (cond1 or cond2) and (i > 150) and (data.qpos[2] > 0.2):
                        run_success = True
                        if not args.eval_success:
                            print(f"Success condition met: door_angle={door_angle:.2f}, base_x={base_x:.2f}")
                        break # Terminate on success to save computation time

                # Skip rendering to accelerate success rate calculation
                if not args.eval_success and i % 2 == 0:
                    id_isaac = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "static_isaac_view")
                    id_track = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")

                    if id_isaac != -1:
                        cam_name = "static_isaac_view"
                    elif id_track != -1:
                        cam_name = "track"
                    else:
                        cam_name = None

                    renderer.update_scene(data, camera=cam_name)
                    frames.append(renderer.render())

                if not args.eval_success and i % 50 == 0:
                    print(f"Step {i} | Base Height: {data.qpos[2]:.3f}")

            if run_success:
                success_count += 1

            if not args.eval_success:
                print(f"\n=======================================")
                print(f"Single Run Finished!")
                print(f"Task Success: {run_success}")
                print(f"Final Base Height: {data.qpos[2]:.3f}")
                print(f"=======================================\n")

        if args.eval_success:
            sr_percent = success_count / num_runs * 100
            print(f"-> Success Rate for {current_run} (Checkpoint {current_checkpoint}): {sr_percent:.2f}% ({success_count}/{num_runs})")
            eval_results[current_run] = sr_percent

    # Exit script cleanly after returning success rates for all policies
    if args.eval_success:
        print(f"\n====== FINAL EVALUATION RESULTS ======")
        for run_name, sr in eval_results.items():
            print(f"{run_name}: {sr:.2f}%")
        print(f"======================================")

        # Save results to file
        suffix = "_left" if args.left else ""
        results_file = os.path.join(exp_dir, f"eval_results_noise_1{suffix}.txt")
        with open(results_file, 'w') as f:
            f.write("====== FINAL EVALUATION RESULTS ======\n")
            for run_name, sr in eval_results.items():
                f.write(f"{run_name}: {sr:.2f}%\n")
            f.write("======================================\n")
        print(f"[✓] Results saved to {results_file}")
        return

    # Visualizations (Only runs if --eval_success is NOT passed, implying a single visual test run)
    plt.figure(figsize=(8, 8))
    plt.plot(base_x_positions, base_y_positions, linestyle='--', color='blue')
    plt.legend()
    plt.title('Base Position Trajectory')
    plt.ylabel('Y Position (m)')
    plt.xlabel('X Position (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, current_run, f"{current_checkpoint}_base_pos_plot_mujoco.pdf")
    plt.savefig(plot_path)
    print(f"[✓] Saved base pos plot to {plot_path}")
    plt.close()

    plot_headings = np.array(current_headings)
    plot_headings[500:][plot_headings[500:] < np.pi/8] += 2 * np.pi

    plt.figure(figsize=(12, 5))
    plt.plot(cmd_yaw_vels, label='Commanded Yaw Velocity', linestyle='--')
    plt.plot(actual_yaw_vels, label='Actual Yaw Velocity', alpha=0.8)
    plt.legend()
    plt.title('Yaw Velocity Tracking')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Velocity (rad/s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, current_run, f"{current_checkpoint}_yaw_vel_plot_mujoco.pdf")
    plt.savefig(plot_path)
    print(f"[✓] Saved yaw velocity plot to {plot_path}")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(cmd_heading, label='Commanded Heading', linestyle='--')
    plt.plot(plot_headings, label='Actual Heading', alpha=0.8)
    plt.legend()
    plt.title('Heading Tracking')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Angle (rad)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, current_run, f"{current_checkpoint}_heading_plot_mujoco.pdf")
    plt.savefig(plot_path)
    print(f"[✓] Saved heading plot to {plot_path}")
    plt.close()

    npy_path = os.path.join(exp_dir, current_run, f"{current_checkpoint}_yaw_heading_data_mujoco.npy")
    np.save(npy_path, {
        "cmd_yaw_vels": np.array(cmd_yaw_vels), "actual_yaw_vels": np.array(actual_yaw_vels),
        "cmd_heading": np.array(cmd_heading), "actual_heading": np.array(plot_headings),
        "cmd_vx": np.array(cmd_x_vels), "cmd_vy": np.array(cmd_y_vels),
        "actual_vx": np.array(actual_x_vels), "actual_vy": np.array(actual_y_vels),
        "base_x_positions": np.array(base_x_positions), "base_y_positions": np.array(base_y_positions)
    })

    print(f"Saving video ...")
    suffix = "_left" if args.left else ""
    imageio.mimsave(os.path.join(exp_dir, current_run, f"{current_checkpoint}{suffix}.mp4"), frames, fps=25)

if __name__ == "__main__":
    play_mujoco()

