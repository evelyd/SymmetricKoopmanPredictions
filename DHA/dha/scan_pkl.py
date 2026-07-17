# from pathlib import Path
# from morpho_symm.data.DynamicsRecording import DynamicsRecording

# # 将路径变为 Path 对象
# pkl_path = Path("/home/weishu/SymmLoco/legged_gym/isaacgym_recordings/stand_dance_cyber_aug/n_trajs=3-frames=14400-test.pkl")

# # 正确加载
# dr = DynamicsRecording.load_from_file(pkl_path)

# print("State observation keys:", dr.state_obs)
# print("State dim:", dr.state_dim)

import torch

ckpt_path = "/home/weishu/SymmLoco/experiments/test/S:20250521203857-OS:3-G:C2-H:5-EH:5_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=278/best.ckpt"
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

print("keys in checkpoint:", checkpoint.keys())