import os
import torch

def find_all_ckpts(root_dir):
    """递归查找所有 best.ckpt 文件"""
    ckpt_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file == "best.ckpt":
                ckpt_paths.append(os.path.join(root, file))
    return ckpt_paths

def get_input_dim_from_ckpt(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        for k, v in state_dict.items():
            if "obs_fn.net.block_0.linear_0.weight" in k:
                return v.shape[1]
    except Exception as e:
        print(f"⚠️ Error loading {ckpt_path}: {e}")
    return None

def main():
    root_dir = "/home/weishu/SymmLoco/experiments/test/S:20250521203857-OS:3-G:C2-H:5-EH:5_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=278"  # 替换为你的模型根目录
    ckpt_paths = find_all_ckpts(root_dir)
    print(f"\n🔍 Found {len(ckpt_paths)} checkpoint(s)\n")
    

    for path in ckpt_paths:
        input_dim = get_input_dim_from_ckpt(path)
        print(f"{path} -> input_dim = {input_dim}")

if __name__ == "__main__":
    main()