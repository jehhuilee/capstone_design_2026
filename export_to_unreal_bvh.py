import numpy as np
import smplx
import torch
import sys
import os
from scipy.spatial.transform import Rotation

# SMPL-X 표준 55개 관절 이름
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2',
    'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1',
    'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
    'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2',
    'right_index3', 'right_middle1', 'right_middle2', 'right_middle3',
    'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2',
    'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3'
]

def export_smplx_to_bvh(bvh_filename, smplx_model_path, pose_data_npz, fps=30.0):
    print(f"Loading SMPL-X model from {smplx_model_path}...")
    model = smplx.create(smplx_model_path, model_type='smplx',
                         gender='neutral', use_pca=False, batch_size=1)
    
    # 뼈대 계층 구조 (Kinematic Tree) 추출
    parents = model.parents.numpy()
    num_joints = len(parents)
    
    print("Calculating rest pose offsets...")
    # T-pose 상태에서의 관절 3D 좌표 추출
    output = model(return_verts=False)
    joints_3d = output.joints[0].detach().numpy() # [55, 3] or more
    
    # BVH용 로컬 오프셋 계산
    offsets = np.zeros((num_joints, 3))
    for i in range(num_joints):
        if parents[i] == -1: # Root
            offsets[i] = joints_3d[i]
        else:
            offsets[i] = joints_3d[i] - joints_3d[parents[i]]
            
    print(f"Loading motion data from {pose_data_npz}...")
    data = np.load(pose_data_npz)
    
    # 루트 이동값
    transl = data['transl'] 
    
    # 개별 파라미터들을 하나로 병합하여 (N, 55, 3) 형태로 만들기
    try:
        global_orient = data['global_orient']
        body_pose = data['body_pose']
        jaw_pose = data['jaw_pose']
        leye_pose = data['leye_pose']
        reye_pose = data['reye_pose']
        left_hand_pose = data['left_hand_pose']
        right_hand_pose = data['right_hand_pose']
        
        # 순서: root(1), body(21), jaw(1), leye(1), reye(1), lhand(15), rhand(15) -> 총 55개 관절
        poses_flat = np.concatenate([
            global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose
        ], axis=1)
        
        poses = poses_flat.reshape(-1, num_joints, 3)
    except KeyError as e:
        print(f"Error: 필수 파라미터가 npz 파일에 없습니다 - {e}")
        return

    num_frames = poses.shape[0]
    
    print("Converting Axis-Angle to Euler angles (ZXY)...")
    # 언리얼 엔진 및 마야 호환성을 위해 ZXY 오더 사용
    euler_poses = np.zeros((num_frames, num_joints, 3))
    for f in range(num_frames):
        for j in range(num_joints):
            rot_vec = poses[f, j]
            r = Rotation.from_rotvec(rot_vec)
            euler_poses[f, j] = r.as_euler('ZXY', degrees=True)
            
    print(f"Writing BVH to {bvh_filename}...")
    with open(bvh_filename, 'w') as f:
        # --- HIERARCHY SECTION ---
        f.write("HIERARCHY\n")
        
        children = {i: [] for i in range(num_joints)}
        for i in range(1, num_joints):
            children[parents[i]].append(i)
            
        def write_joint(joint_idx, indent_level):
            indent = "  " * indent_level
            j_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"Joint_{joint_idx}"
            
            if joint_idx == 0:
                f.write(f"{indent}ROOT {j_name}\n")
            else:
                f.write(f"{indent}JOINT {j_name}\n")
            
            f.write(f"{indent}{{\n")
            
            indent2 = "  " * (indent_level + 1)
            off = offsets[joint_idx]
            f.write(f"{indent2}OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
            
            if joint_idx == 0:
                f.write(f"{indent2}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            else:
                f.write(f"{indent2}CHANNELS 3 Zrotation Xrotation Yrotation\n")
                
            if len(children[joint_idx]) > 0:
                for child_idx in children[joint_idx]:
                    write_joint(child_idx, indent_level + 1)
            else:
                # End Site
                f.write(f"{indent2}End Site\n")
                f.write(f"{indent2}{{\n")
                f.write(f"{indent2}  OFFSET 0.0 0.0 0.0\n")
                f.write(f"{indent2}}}\n")
                
            f.write(f"{indent}}}\n")

        write_joint(0, 0)
        
        # --- MOTION SECTION ---
        f.write("MOTION\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {1.0 / fps:.7f}\n")
        
        for frame in range(num_frames):
            line_data = []
            for j in range(num_joints):
                if j == 0:
                    t = transl[frame]
                    line_data.extend([f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}"])
                
                e = euler_poses[frame, j]
                line_data.extend([f"{e[0]:.6f}", f"{e[1]:.6f}", f"{e[2]:.6f}"])
                
            f.write(" ".join(line_data) + "\n")

    print(f"Success: Exported {num_frames} frames to {bvh_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_to_unreal_bvh.py <input.npz> <output.bvh>")
        sys.exit(1)
        
    input_npz = os.path.abspath(sys.argv[1])
    output_bvh = os.path.abspath(sys.argv[2])
    smplx_model_file = os.path.abspath("model/SMPLX_NEUTRAL.npz")
    
    export_smplx_to_bvh(output_bvh, smplx_model_file, input_npz)
