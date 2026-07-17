import sys
import functools

# 1. Patch functools.cache (New in 3.9)
if not hasattr(functools, "cache"):
    functools.cache = functools.lru_cache(maxsize=None)

# 2. Patch importlib.resources.files (New in 3.9)
# We use the 'importlib_resources' backport installed via pip
if sys.version_info < (3, 9):
    try:
        import importlib.resources
        import importlib_resources
        # Inject the backport into the stdlib module so other libs find it
        importlib.resources.files = importlib_resources.files
    except ImportError:
        print("CRITICAL ERROR: Python 3.8 detected but 'importlib_resources' not found.")
        print("Please run: pip install importlib_resources")
        sys.exit(1)

import os
import glob
import trimesh
from legged_gym import LEGGED_GYM_ROOT_DIR

def convert_all_dae_to_obj(root_dir=f"{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/"):
    print(f"Searching for .dae files in '{root_dir}'...")
    files = glob.glob(f"{root_dir}/**/*.dae", recursive=True)

    if not files:
        print("No .dae files found! Check your folder structure.")
        return

    print(f"Found {len(files)} files. Converting to OBJ...")
    for dae_file in files:
        obj_file = dae_file.replace(".dae", ".obj")

        # Skip if already exists to save time
        if os.path.exists(obj_file):
            print(f"  [Skip] {obj_file} already exists.")
            continue

        try:
            # Load and export
            mesh = trimesh.load(dae_file, force='mesh')
            mesh.export(obj_file)
            print(f"  [OK] Converted {dae_file} -> {obj_file}")
        except Exception as e:
            print(f"  [FAIL] Could not convert {dae_file}: {e}")

if __name__ == "__main__":
    convert_all_dae_to_obj()