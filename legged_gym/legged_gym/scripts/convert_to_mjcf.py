import os
import xml.etree.ElementTree as ET
import mujoco
from legged_gym import LEGGED_GYM_ROOT_DIR

def convert_urdf_to_mjcf(urdf_path, xml_output_path):
    print(f"--- Processing {urdf_path} ---")

    # 1. Setup Paths
    if not os.path.exists(urdf_path):
        print(f"ERROR: File not found at {urdf_path}")
        return

    # /.../cyberdog2/urdf
    abs_urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    
    # 2. Parse URDF
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR Parsing URDF: {e}")
        return

    print("--- 1. Patching URDF Paths & Joints ---")

    # Smart Mesh Directory Detection
    abs_mesh_dir = os.path.abspath(os.path.join(abs_urdf_dir, "../dae"))
    rel_mesh_dir = "../dae"
    if not os.path.exists(abs_mesh_dir):
        alt_mesh_dir = os.path.abspath(os.path.join(abs_urdf_dir, "../meshes"))
        if os.path.exists(alt_mesh_dir):
            abs_mesh_dir = alt_mesh_dir
            rel_mesh_dir = "../meshes"
        else:
            print(f"WARNING: Neither ../dae nor ../meshes found relative to URDF.")

    # A. Fix Paths & Names
    for link in root.findall('link'):
        link_name = link.get('name')

        # Rename visual/collision elements to be unique
        for tag_type in ['visual', 'collision']:
            for i, elem in enumerate(link.findall(tag_type)):
                elem.set('name', f"{link_name}_{tag_type}_{i}")

                # FIX MESH PATHS
                geom = elem.find('geometry')
                if geom is not None:
                    mesh = geom.find('mesh')
                    if mesh is not None:
                        filename = mesh.get('filename', '')

                        # Look for .dae to swap to .obj (Also handle .stl if present)
                        if filename.endswith('.dae') or filename.endswith('.stl'):
                            base_name = os.path.basename(filename)
                            
                            # Swap extension to .obj if it was .dae
                            if filename.endswith('.dae'):
                                new_name = base_name.replace('.dae', '.obj')
                            else:
                                new_name = base_name 

                            mesh.set('filename', new_name)

    # B. Fix Foot Joints (Prevent collapsing)
    foot_joints = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
    for joint in root.findall('joint'):
        if joint.get('name') in foot_joints:
            joint.set('type', 'revolute')
            axis = joint.find('axis')
            if axis is None: axis = ET.SubElement(joint, 'axis')
            axis.set('xyz', '0 0 1')
            limit = joint.find('limit')
            if limit is None: limit = ET.SubElement(joint, 'limit')
            limit.set('lower', '0')
            limit.set('upper', '0')
            limit.set('effort', '0')
            limit.set('velocity', '0')

    # C. Inject MuJoCo Compiler Settings
    mj_tag = root.find('mujoco')
    if mj_tag is None: mj_tag = ET.SubElement(root, 'mujoco')
    compiler_tag = mj_tag.find('compiler')
    if compiler_tag is None: compiler_tag = ET.SubElement(mj_tag, 'compiler')

    compiler_tag.set('fusestatic', 'false')
    compiler_tag.set('angle', 'radian')
    compiler_tag.set('discardvisual', 'false')

    # Explicitly tell MuJoCo where the meshes are
    compiler_tag.set('meshdir', abs_mesh_dir)

    # Save Temporary URDF (with absolute paths)
    temp_urdf_path = os.path.join(abs_urdf_dir, "temp_fixed.urdf")
    tree.write(temp_urdf_path)

    # 3. Compile with MuJoCo
    print("--- 2. Compiling to MJCF ---")
    try:
        model = mujoco.MjModel.from_xml_path(temp_urdf_path)
        mujoco.mj_saveLastXML(xml_output_path, model)
    except Exception as e:
        print(f"FATAL: MuJoCo compilation failed.\n{e}")
        print(f"Debug Hint: Check {temp_urdf_path} to see what paths were generated.")
        return

    # 4. Post-Process MJCF
    print("--- 3. Post-Processing MJCF ---")
    try:
        mjcf_tree = ET.parse(xml_output_path)
        mjcf_root = mjcf_tree.getroot()

        # A. Make Paths Relative Again
        if mjcf_root.find('asset'):
            for mesh in mjcf_root.find('asset').findall('mesh'):
                abs_file = mesh.get('file')
                if abs_file:
                    mesh.set('file', os.path.basename(abs_file))

        # B. Set Compiler meshdir (Using detected relative path)
        compiler = mjcf_root.find('compiler')
        if compiler is None: compiler = ET.SubElement(mjcf_root, 'compiler')
        compiler.set('meshdir', rel_mesh_dir)
        compiler.set('texturedir', rel_mesh_dir)

        # C. Add Aesthetics (Sky, Floor, Lights)
        asset = mjcf_root.find('asset')
        if asset is None:
            asset = ET.Element('asset')
            mjcf_root.insert(0, asset)

        ET.SubElement(asset, 'texture', {'type': 'skybox', 'builtin': 'gradient', 'rgb1': '.3 .5 .7', 'rgb2': '0 0 0', 'width': '32', 'height': '32'})
        ET.SubElement(asset, 'texture', {'name': 'grid', 'type': '2d', 'builtin': 'checker', 'width': '512', 'height': '512', 'rgb1': '.1 .2 .3', 'rgb2': '.2 .3 .4'})
        ET.SubElement(asset, 'material', {'name': 'grid_mat', 'texture': 'grid', 'texrepeat': '1 1', 'texuniform': 'true', 'reflectance': '0.2'})

        visual = mjcf_root.find('visual')
        if visual is None:
            visual = ET.SubElement(mjcf_root, 'visual')
            ET.SubElement(visual, 'headlight', {'diffuse': '0.6 0.6 0.6', 'ambient': '0.3 0.3 0.3'})
            ET.SubElement(visual, 'rgba', {'haze': '0.15 0.25 0.35 1'})

        worldbody = mjcf_root.find('worldbody')
        if worldbody is not None:
            # Floor
            floor = worldbody.find("geom")
            if floor is None:
                floor = ET.Element('geom', {'name': 'floor', 'type': 'plane', 'size': '50 50 0.1'})
                worldbody.insert(0, floor)
            floor.set('material', 'grid_mat')
            if 'rgba' in floor.attrib: del floor.attrib['rgba']

            # Add Cameras
            for body in worldbody.findall('body'):
                # Note: Assuming the root link is named 'base' or 'trunk'. 
                # Adjust if CyberDog2 uses a different root link name.
                if body.get('name') in ['base', 'trunk', 'base_link']:
                    if body.find('freejoint') is None:
                        body.insert(0, ET.Element('freejoint', {'name': 'root_joint'}))

                    if body.find('camera') is None:
                        ET.SubElement(body, 'camera', {'name': 'track', 'mode': 'trackcom', 'pos': '-0.5 0 0.5', 'xyaxes': '0 -1 0 0 0 1'})
                    break

            # Global view camera
            ET.SubElement(worldbody, 'camera', {'name': 'isaac_view', 'pos': '1.5 1.5 0.5', 'mode': 'targetbody', 'target': 'base'})

        # D. Add Motors
        actuator_tag = mjcf_root.find('actuator')
        if actuator_tag is None: actuator_tag = ET.SubElement(mjcf_root, 'actuator')
        actuated_joints = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        for joint_name in actuated_joints:
            if not any(child.get('joint') == joint_name for child in actuator_tag):
                ET.SubElement(actuator_tag, 'motor', {'name': f"{joint_name}_motor", 'joint': joint_name, 'gear': '1', 'ctrllimited': 'false'})

        mjcf_tree.write(xml_output_path)
        print(f"--- Finished! Saved to {xml_output_path} ---")

    except Exception as e:
        print(f"Error during XML patching: {e}")

    # Cleanup temp file
    if os.path.exists(temp_urdf_path): os.remove(temp_urdf_path)

if __name__ == "__main__":
    # CyberDog 2 paths
    candidate_path = "legged_gym/resources/robots/cyberdog2/urdf/cyberdog2.urdf"
    
    if not os.path.exists(candidate_path):
        candidate_path = "resources/robots/cyberdog2/urdf/cyberdog2.urdf"

    if os.path.exists(candidate_path):
        urdf_file = candidate_path
        output_file = candidate_path.replace(".urdf", ".xml")
        convert_urdf_to_mjcf(urdf_file, output_file)
    else:
        print("Could not auto-detect URDF path. Using absolute defaults.")
        urdf_file = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2.urdf"
        output_file = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2.xml"
        convert_urdf_to_mjcf(urdf_file, output_file)