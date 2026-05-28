import bpy
import sys
import os
import numpy as np

def _rotvec_to_mat3(rv):
    angle = float(np.linalg.norm(rv))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = rv / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def _mat3_to_rotvec(mat):
    theta = np.arccos(np.clip((np.trace(mat) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    if abs(theta - np.pi) < 1e-6:
        RpI = mat + np.eye(3)
        col = int(np.argmax(np.sum(RpI ** 2, axis=0)))
        axis = RpI[:, col].copy()
        axis /= np.linalg.norm(axis)
        return (axis * theta).astype(np.float32)
    denom = 2.0 * np.sin(theta)
    axis = np.array([mat[2, 1] - mat[1, 2],
                     mat[0, 2] - mat[2, 0],
                     mat[1, 0] - mat[0, 1]], dtype=np.float64) / denom
    return (axis * theta).astype(np.float32)

# Y-up → Z-up: R_x(+90°), (x,y,z) → (x,-z,y)
_R_Y2Z = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]], dtype=np.float64)

def convert_hmr_to_amass(input_npz, output_npz):
    data = np.load(input_npz, allow_pickle=True)
    
    N = data['global_orient'].shape[0]
    
    poses = np.zeros((N, 165), dtype=np.float32)
    poses[:, 0:3] = data['global_orient']
    if 'body_pose' in data:
        poses[:, 3:66] = data['body_pose']
    
    if 'left_hand_pose' in data:
        poses[:, 75:120] = data['left_hand_pose']
    if 'right_hand_pose' in data:
        poses[:, 120:165] = data['right_hand_pose']
        
    trans = data['transl'] if 'transl' in data else np.zeros((N, 3), dtype=np.float32)
    
    betas = data['betas'] if 'betas' in data else np.zeros(10, dtype=np.float32)
    if len(betas.shape) == 2:
        betas = betas[0]
        
    gender = str(data['gender']) if 'gender' in data else 'neutral'

    # ── Y-up → Z-up 좌표 변환 ──
    trans = ((_R_Y2Z @ trans.astype(np.float64).T).T).astype(np.float32)
    for i in range(N):
        rv = poses[i, :3].astype(np.float64)
        R_orig = _rotvec_to_mat3(rv)
        R_new = _R_Y2Z @ R_orig
        poses[i, :3] = _mat3_to_rotvec(R_new)
    print(f"  [Y->Z] Converted {N} frames from Y-up to Z-up")
    
    np.savez(output_npz,
             poses=poses,
             trans=trans,
             betas=betas,
             gender=gender,
             mocap_frame_rate=30.0)

def main():
    argv = sys.argv
    if "--" not in argv:
        print("Error: Not enough arguments.")
        sys.exit(1)
        
    args = argv[argv.index("--") + 1:]
    if len(args) < 1:
        print("Usage: blender -b smplx_template.blend -P export_to_unreal_fbx.py -- <input.npz> [output_dir]")
        sys.exit(1)
        
    animation_file = os.path.abspath(args[0])
    
    if len(args) >= 2:
        output_dir = os.path.abspath(args[1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = os.path.dirname(animation_file)
        
    base_name = os.path.splitext(os.path.basename(animation_file))[0]
    output_fbx = os.path.join(output_dir, f"{base_name}_unreal.fbx")

    print("--------------------------------------------------")
    print(f"Input NPZ : {animation_file}")
    print(f"Output FBX: {output_fbx}")
    print("--------------------------------------------------")

    # Create temporary AMASS npz
    temp_npz = os.path.join(output_dir, f"{base_name}_amass_temp.npz")
    print("Converting HMR NPZ to AMASS NPZ format...")
    convert_hmr_to_amass(animation_file, temp_npz)

    # Find Armature
    armature_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
            break
            
    if armature_obj is None:
        print("Error: No armature found.")
        sys.exit(1)
        
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # Load SMPL-X Animation using converted npz
    try:
        if hasattr(bpy.ops.object, 'smplx_add_animation'):
            # anim_format='AMASS' applies a -90 degree rotation on the root bone to correctly orient it from OpenGL Y-up to Z-up, matching Unreal!
            bpy.ops.object.smplx_add_animation(filepath=temp_npz, anim_format='AMASS')
            print("Animation loaded successfully.")
        else:
            print("Error: smplx_add_animation operator not found.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during animation load: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_npz):
            os.remove(temp_npz)

    # After animation load, find the newly created armature
    # The addon creates a new armature and mesh. The active object might be the mesh.
    # We will look for the armature that was just created (its name will contain "temp" because of temp_npz)
    final_armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and 'amass_temp' in obj.name:
            final_armature = obj
            break
            
    # Fallback: just find any armature that is not the original one
    if not final_armature:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and obj != armature_obj:
                final_armature = obj
                break
                
    if not final_armature:
        final_armature = armature_obj  # fallback to whatever we had
        
    if final_armature and final_armature.type == 'ARMATURE':
        # Rename armature to "root" so Unreal Engine creates a proper root bone
        final_armature.name = "root"
        
        # Ensure only this armature and its child meshes are selected for export
        bpy.ops.object.select_all(action='DESELECT')
        final_armature.select_set(True)
        # Also select the mesh that the addon created. Usually it's parented to the armature, or it has an armature modifier.
        # Let's just find meshes that have an armature modifier pointing to this armature.
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object == final_armature:
                        obj.select_set(True)
                        # Remove shape keys if they cause issues in Unreal (Optional, but let's keep them and rely on bake_anim)
        
        bpy.context.view_layer.objects.active = final_armature
    
    # Export FBX
    print(f"Starting FBX Export: {output_fbx}")
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=True,
        global_scale=1.0,
        bake_anim=True,
        add_leaf_bones=False,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        axis_forward='X',
        axis_up='Z',
        bake_space_transform=True,
        path_mode='COPY',
        embed_textures=False
    )
    
    print("FBX Export success!")

if __name__ == "__main__":
    main()
