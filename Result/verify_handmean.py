"""RELAXED 수정 검증: FBX 손가락 회전이 (npz_hand) 인지 (npz_hand + hands_mean) 인지 판별.
실행: blender -b -P Result/verify_handmean.py -- <npz> <fbx> <model_npz>
"""
import bpy, sys, os, math
import numpy as np
import mathutils

a = sys.argv[sys.argv.index("--") + 1:]
npz_path, fbx_path, model_path = os.path.abspath(a[0]), os.path.abspath(a[1]), os.path.abspath(a[2])

d = np.load(npz_path, allow_pickle=True)
N = d['global_orient'].shape[0]
lh = np.asarray(d['left_hand_pose']).reshape(N, 15, 3)
rh = np.asarray(d['right_hand_pose']).reshape(N, 15, 3)

m = np.load(model_path, allow_pickle=True)
meanl = np.asarray(m['hands_meanl']).reshape(15, 3)
meanr = np.asarray(m['hands_meanr']).reshape(15, 3)

JOINTS = ["index1","index2","index3","middle1","middle2","middle3",
          "pinky1","pinky2","pinky3","ring1","ring2","ring3",
          "thumb1","thumb2","thumb3"]

def q(rv):
    n = float(np.linalg.norm(rv))
    return mathutils.Quaternion((1,0,0,0)) if n < 1e-8 else mathutils.Quaternion(mathutils.Vector(rv/n), n)

bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=fbx_path)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

f = int(np.argmax(np.linalg.norm(lh.reshape(N, -1), axis=1)))
bpy.context.scene.frame_set(f + 1); bpy.context.view_layer.update()
print(f"[chk] frame npz={f} / FBX={f+1}")

diff_raw, diff_mean = [], []
for side, hand, mean in [("left", lh, meanl), ("right", rh, meanr)]:
    for j, jn in enumerate(JOINTS):
        pb = arm.pose.bones.get(f"{side}_{jn}")
        if pb is None:
            continue
        fbx_q = pb.matrix_basis.to_quaternion()
        diff_raw.append(math.degrees(q(hand[f, j]).rotation_difference(fbx_q).angle))
        diff_mean.append(math.degrees(q(hand[f, j] + mean[j]).rotation_difference(fbx_q).angle))

diff_raw, diff_mean = np.array(diff_raw), np.array(diff_mean)
print(f"\nFBX vs (npz_hand)        : 평균 Δ={diff_raw.mean():.4f}°  최대={diff_raw.max():.4f}°")
print(f"FBX vs (npz_hand + 평균) : 평균 Δ={diff_mean.mean():.4f}°  최대={diff_mean.max():.4f}°")
print("\n→ '+평균' 쪽 Δ≈0 이면 RELAXED 수정이 평균을 올바르게 더한 것 (시각화와 일치).")
