import bpy
import sys
import numpy as np
import os

def convert_hmr_to_amass(input_npz, output_npz):
    data = np.load(input_npz, allow_pickle=True)
    N = data['global_orient'].shape[0]
    poses = np.zeros((N, 165), dtype=np.float32)
    poses[:, 0:3] = data['global_orient']
    if 'body_pose' in data: poses[:, 3:66] = data['body_pose']
    if 'left_hand_pose' in data: poses[:, 75:120] = data['left_hand_pose']
    if 'right_hand_pose' in data: poses[:, 120:165] = data['right_hand_pose']
    trans = data['transl'] if 'transl' in data else np.zeros((N, 3), dtype=np.float32)
    betas = data['betas'] if 'betas' in data else np.zeros(10, dtype=np.float32)
    if len(betas.shape) == 2: betas = betas[0]
    gender = str(data['gender']) if 'gender' in data else 'neutral'
    np.savez(output_npz, poses=poses, trans=trans, betas=betas, gender=gender, mocap_frame_rate=30.0)

def main():
    animation_file = r"C:\Users\user\Documents\GitHub\capstone_design_2026\gt\06_06_13_stageii.npz"
    temp_npz = "temp.npz"
    convert_hmr_to_amass(animation_file, temp_npz)
    
    bpy.ops.object.smplx_add_animation(filepath=temp_npz, anim_format='SMPL-X')
    
    final_armature = bpy.context.view_layer.objects.active
    print("Armature:", final_armature.name)
    print("Children:", [c.name for c in final_armature.children])
    print("All Objects:", [o.name for o in bpy.data.objects])

if __name__ == "__main__":
    main()
