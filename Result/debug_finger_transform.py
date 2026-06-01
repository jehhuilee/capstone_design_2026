"""애드온이 적용한 손가락 회전(정답) vs 우리 conjugation 공식 vs 직접대입 비교.
실행: blender -b smplx_template.blend -P Result/debug_finger_transform.py -- <hmr_npz>
"""
import bpy, sys, os, math
import numpy as np
import mathutils

argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []
hmr_npz = os.path.abspath(args[0])

# ── HMR npz → AMASS poses(165) 임시 변환 ──────────────────────────
d = np.load(hmr_npz, allow_pickle=True)
N = d['global_orient'].shape[0]
poses = np.zeros((N, 165), dtype=np.float32)
poses[:, 0:3] = np.asarray(d['global_orient']).reshape(N, -1)[:, :3]
if 'body_pose' in d:
    bp = np.asarray(d['body_pose']).reshape(N, -1)
    poses[:, 3:3 + min(bp.shape[1], 63)] = bp[:, :63]
if 'left_hand_pose' in d:
    poses[:, 75:120] = np.asarray(d['left_hand_pose']).reshape(N, -1)[:, :45]
if 'right_hand_pose' in d:
    poses[:, 120:165] = np.asarray(d['right_hand_pose']).reshape(N, -1)[:, :45]
trans = np.asarray(d['transl']).reshape(N, -1)[:, :3] if 'transl' in d else np.zeros((N, 3), np.float32)
betas = np.asarray(d['betas'], np.float32)
betas = betas[0] if betas.ndim == 2 else betas

tmp = os.path.join(os.path.dirname(hmr_npz), "_dbg_amass.npz")
np.savez(tmp, poses=poses.astype(np.float32), trans=trans.astype(np.float32),
         betas=betas, gender="neutral", mocap_frame_rate=30.0)

# ── 손가락 모션이 가장 큰 프레임 선택 ────────────────────────────
lh = poses[:, 75:120].reshape(N, 15, 3)
frame_pick = int(np.argmax(np.linalg.norm(lh.reshape(N, -1), axis=1)))
print(f"[dbg] N={N}, 손가락 모션 최대 프레임 = {frame_pick}")

# ── 애드온으로 로드 ──────────────────────────────────────────────
arm0 = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'), None)
bpy.ops.object.select_all(action='DESELECT')
arm0.select_set(True)
bpy.context.view_layer.objects.active = arm0
before = {o.name for o in bpy.data.objects}
bpy.ops.object.smplx_add_animation(filepath=tmp, anim_format='AMASS')
arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE' and o.name not in before), arm0)
print(f"[dbg] armature = {arm.name}, object world rot = {tuple(round(x,3) for x in arm.matrix_world.to_quaternion())}")

# addon은 frame = index+1 로 키프레임을 넣음
bpy.context.scene.frame_set(frame_pick + 1)
bpy.context.view_layer.update()

S_inv = mathutils.Matrix(((1,0,0),(0,0,1),(0,-1,0))).to_quaternion()
obj_rot = arm.matrix_world.to_quaternion()

TEST_BONES = ["left_index1", "left_index2", "left_thumb1", "left_middle1", "left_pinky1"]
print(f"\n{'bone':<14s} {'Δ(addon,direct)°':>16s} {'Δ(addon,conj)°':>16s}")
for j, bname in enumerate([
        "index1","index2","index3","middle1","middle2","middle3",
        "pinky1","pinky2","pinky3","ring1","ring2","ring3",
        "thumb1","thumb2","thumb3"]):
    key = f"left_{bname}"
    pb = arm.pose.bones.get(key)
    if pb is None:
        continue
    addon_q = pb.matrix_basis.to_quaternion()          # 애드온이 넣은 값 (정답)

    rv = lh[frame_pick, j].astype(float)
    ang = float(np.linalg.norm(rv))
    direct_q = (mathutils.Quaternion((1,0,0,0)) if ang < 1e-8
                else mathutils.Quaternion(mathutils.Vector(rv/ang), ang))

    rest_native = S_inv @ (obj_rot @ pb.bone.matrix_local.to_quaternion())
    conj_q = rest_native.inverted() @ direct_q @ rest_native

    d_direct = math.degrees(addon_q.rotation_difference(direct_q).angle)
    d_conj   = math.degrees(addon_q.rotation_difference(conj_q).angle)
    print(f"{key:<14s} {d_direct:16.3f} {d_conj:16.3f}")

if os.path.exists(tmp):
    os.remove(tmp)
print("\n[dbg] Δ(addon,direct)≈0 이면 '직접대입'이 정답이고 conjugation이 버그임이 증명됨.")
