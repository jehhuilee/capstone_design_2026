"""Diagnose hand pose data and bone mapping."""
import numpy as np
import torch
import sys

print("=" * 60)
print("HAND POSE DIAGNOSTIC")
print("=" * 60)

# 1. Check original .pt file
pt_path = r"c:\Users\user\Desktop\CG\캡스톤\cap_pipeline\front_4\smplx_merged_hamer_post.pt"
print(f"\n[1] Original .pt file: {pt_path}")
data = torch.load(pt_path, map_location='cpu')
params = data['smpl_params_global']

print(f"  Keys in smpl_params_global: {list(params.keys())}")

for key in ['left_hand_pose', 'right_hand_pose']:
    if key in params:
        v = params[key]
        arr = v.numpy() if hasattr(v, 'numpy') else np.array(v)
        print(f"\n  {key}:")
        print(f"    shape: {arr.shape}")
        print(f"    dtype: {arr.dtype}")
        print(f"    min: {arr.min():.6f}, max: {arr.max():.6f}")
        print(f"    mean abs: {np.abs(arr).mean():.6f}")
        print(f"    all zeros?: {np.allclose(arr, 0.0)}")
        # Check per-joint stats (15 joints x 3 axis-angle)
        T = arr.shape[0]
        if arr.shape[1] == 45:
            joints = arr.reshape(T, 15, 3)
            norms = np.linalg.norm(joints, axis=2)  # (T, 15)
            print(f"    Per-joint max rotation (degrees):")
            joint_names = ["index1", "index2", "index3",
                          "middle1", "middle2", "middle3",
                          "pinky1", "pinky2", "pinky3",
                          "ring1", "ring2", "ring3",
                          "thumb1", "thumb2", "thumb3"]
            for j in range(15):
                max_deg = np.degrees(norms[:, j].max())
                mean_deg = np.degrees(norms[:, j].mean())
                nonzero = np.count_nonzero(norms[:, j] > 0.01)
                print(f"      {joint_names[j]:>10s}: max={max_deg:6.1f}°  mean={mean_deg:5.1f}°  nonzero_frames={nonzero}/{T}")
    else:
        print(f"  {key}: NOT FOUND")

# 2. Check postprocessed NPZ
npz_path = r"c:\Users\user\Documents\GitHub\capstone_design_2026\Result\smplx_merged_hamer_post_postprocessed.npz"
print(f"\n[2] Postprocessed NPZ: {npz_path}")
npz = np.load(npz_path, allow_pickle=True)
print(f"  Keys: {list(npz.keys())}")

for key in ['left_hand_pose', 'right_hand_pose']:
    if key in npz:
        arr = npz[key]
        print(f"\n  {key}:")
        print(f"    shape: {arr.shape}")
        print(f"    min: {arr.min():.6f}, max: {arr.max():.6f}")
        print(f"    all zeros?: {np.allclose(arr, 0.0)}")
    else:
        print(f"  {key}: NOT FOUND in postprocessed NPZ!")

# 3. Check AMASS tmp format
print(f"\n[3] Simulating AMASS conversion to check poses[:, 75:165]")
npz2 = np.load(npz_path, allow_pickle=True)
if 'left_hand_pose' in npz2:
    lh = npz2['left_hand_pose']
    rh = npz2['right_hand_pose']
    N = lh.shape[0]
    poses = np.zeros((N, 165), dtype=np.float32)
    poses[:, 75:75+min(lh.shape[1], 45)] = lh[:, :45]
    poses[:, 120:120+min(rh.shape[1], 45)] = rh[:, :45]
    print(f"  poses[:, 75:120] (left hand) - all zeros?: {np.allclose(poses[:, 75:120], 0.0)}")
    print(f"  poses[:, 120:165] (right hand) - all zeros?: {np.allclose(poses[:, 120:165], 0.0)}")
    print(f"  Left hand max abs value: {np.abs(poses[:, 75:120]).max():.6f}")
    print(f"  Right hand max abs value: {np.abs(poses[:, 120:165]).max():.6f}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
