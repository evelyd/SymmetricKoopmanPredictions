from isaacgym import gymapi, gymutil, gymtorch
import math
import torch
import numpy as np

# 初始化 gym
gym = gymapi.acquire_gym()

# 创建模拟环境
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# 创建 viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# 加载 robot asset
asset_root = "/home/weishu/SymmLoco/legged_gym/resources/robots/DOG/urdf"
asset_file = "DOG.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.disable_gravity = False
asset_options.collapse_fixed_joints = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # 注意启用 position control
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 创建地形
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# 创建 env
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0.5)
pose.r = gymapi.Quat(0, 0, 0, 1)
actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

# 设置初始 joint angles（单位：弧度）
stand_angles = {
    "FBL_ABAD_JOINT": 0.0,
    "RBL_ABAD_JOINT": 0.0,
    "FAR_ABAD_JOINT": 0.0,
    "RAR_ABAD_JOINT": 0.0,
    "FBL_HIP_JOINT": -0.6,
    "RBL_HIP_JOINT": -0.6,
    "FAR_HIP_JOINT": -0.6,
    "RAR_HIP_JOINT": -0.6,
    "FBL_KNEE_JOINT": 1.2,
    "RBL_KNEE_JOINT": 1.2,
    "FAR_KNEE_JOINT": 1.2,
    "RAR_KNEE_JOINT": 1.2,
}

# 获取 DOF 名称
dof_names = gym.get_asset_dof_names(asset)
num_dofs = len(dof_names)
dof_dict = {name: i for i, name in enumerate(dof_names)}

# 设置目标位置张量
dof_targets = torch.zeros(num_dofs, dtype=torch.float32)
for name, angle in stand_angles.items():
    if name in dof_dict:
        dof_targets[dof_dict[name]] = angle

# 设置机器人 DOF 状态（角度 + 速度）
dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
for i in range(num_dofs):
    dof_states[i]['pos'] = dof_targets[i].item()
    dof_states[i]['vel'] = 0.0
gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

# 设置 Position 控制目标角度
dof_targets_array = dof_targets.numpy().astype(np.float32)
gym.set_actor_dof_position_targets(env, actor_handle, dof_targets_array)

# 主循环
print("✅ Robot loaded in stand pose. Close the viewer window to exit.")
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


# from isaacgym import gymapi, gymutil
# import math

# # 初始化 gym
# gym = gymapi.acquire_gym()

# # 创建模拟环境
# sim_params = gymapi.SimParams()
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
# sim_params.physx.use_gpu = True
# sim_params.physx.num_position_iterations = 4
# sim_params.physx.num_velocity_iterations = 1
# sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# if sim is None:
#     raise Exception("Failed to create sim")

# # 创建 viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     raise Exception("Failed to create viewer")

# # 加载 robot asset
# asset_root = "/home/weishu/SymmLoco/legged_gym/resources/robots/DOG/urdf"
# asset_file = "DOG.urdf"
# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = False
# asset_options.disable_gravity = False
# asset_options.collapse_fixed_joints = False
# asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
# asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# # 创建地形
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
# gym.add_ground(sim, plane_params)

# # 创建 env
# env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
# pose = gymapi.Transform()
# pose.p = gymapi.Vec3(0, 0, 1.0)
# pose.r = gymapi.Quat(0, 0, 0, 1)  # 单位四元数，表示无旋转，机器人正着落地
# actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

# # 添加灯光（默认 viewer 已有）
# print("✅ Robot loaded. Close the viewer window to exit.")
# while not gym.query_viewer_has_closed(viewer):
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
