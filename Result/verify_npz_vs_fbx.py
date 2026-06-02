"""변환 전 npz(손 axis-angle)와 변환 후 FBX(본 쿼터니언)의 손가락 값 일치 검증.
실행: blender -b -P Result/verify_npz_vs_fbx.py -- <npz> <fbx>
"""
import bpy, sys, os, math
import numpy as np
import mathutils

args = sys.argv[sys.argv.index("--") + 1:]
npz_path, fbx_path = os.path.abspath(args[0]), os.path.abspath(args[1])

# ── npz 손 포즈 로드 (axis-angle) ────────────────────────────────
d = np.load(npz_path, allow_pickle=True)
N = d['global_orient'].shape[0]
lh = np.asarray(d['left_hand_pose']).reshape(N, 15, 3)
rh = np.asarray(d['right_hand_pose']).reshape(N, 15, 3)
JOINTS = ["index1","index2","index3","middle1","middle2","middle3",
          "pinky1","pinky2","pinky3","ring1","ring2","ring3",
          "thumb1","thumb2","thumb3"]

def rotvec_to_quat(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-8:
        return mathutils.Quaternion((1, 0, 0, 0))
    return mathutils.Quaternion(mathutils.Vector(rv / a), a)

# ── FBX 임포트 ───────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=fbx_path)
arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
print(f"[chk] armature={arm.name}, npz N={N}")

# 비교할 프레임: 0, 손가락 모션 최대, 중간
f_max = int(np.argmax(np.linalg.norm(lh.reshape(N, -1), axis=1)))
SAMPLE_FRAMES = sorted(set([0, f_max, N // 2, N - 1]))

all_diffs = []
for npz_f in SAMPLE_FRAMES:
    bpy.context.scene.frame_set(npz_f + 1)        # addon: FBX frame = npz frame + 1
    bpy.context.view_layer.update()
    print(f"\n=== npz frame {npz_f}  (FBX frame {npz_f+1}) ===")
    print(f"{'bone':<16s} {'npz rotvec (axis-angle)':<34s} {'Δ(npz→quat, FBX)°':>18s}")
    for side, hand in [("left", lh), ("right", rh)]:
        for j, jn in enumerate(JOINTS):
            bname = f"{side}_{jn}"
            pb = arm.pose.bones.get(bname)
            if pb is None:
                continue
            rv = hand[npz_f, j].astype(float)
            q_npz = rotvec_to_quat(rv)
            q_fbx = pb.matrix_basis.to_quaternion()
            diff = math.degrees(q_npz.rotation_difference(q_fbx).angle)
            all_diffs.append(diff)
            # 대표 본만 자세히 출력(너무 길지 않게)
            if jn in ("index1", "thumb1", "middle2"):
                rvs = f"({rv[0]:+.3f}, {rv[1]:+.3f}, {rv[2]:+.3f})"
                print(f"{bname:<16s} {rvs:<34s} {diff:18.4f}")

all_diffs = np.array(all_diffs)
print("\n──────────────────────────────────────────")
print(f"전체 손가락 본 × 샘플프레임 비교: n={len(all_diffs)}")
print(f"  평균 Δ = {all_diffs.mean():.4f}°")
print(f"  최대 Δ = {all_diffs.max():.4f}°")
print(f"  중앙값 Δ = {np.median(all_diffs):.4f}°")
print("Δ≈0° 이면 npz 손가락 값이 FBX에 그대로(직접대입) 보존됨을 의미.")
