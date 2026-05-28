from __future__ import annotations

import argparse
import os
import sys
import math
import re
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
import mathutils


# ─────────────────────────────────────────────────────────────
#  좌표계 변환 (Y-up SMPL-X → Z-up Blender/Unreal)
# ─────────────────────────────────────────────────────────────
_R_Y2Z = np.array(
    [[1.0,  0.0,  0.0],
     [0.0,  0.0, -1.0],
     [0.0,  1.0,  0.0]],
    dtype=np.float64,
)

EXPORT_AXIS: Dict[str, Dict] = {
    "unreal": {"axis_forward": "X",  "axis_up": "Z"},
    "unity":  {"axis_forward": "-Z", "axis_up": "Y"},
}

# SMPL-X 손가락 관절 순서 (각 손 15개, AMASS poses 인덱스 순서)
HAND_JOINT_NAMES: List[str] = [
    "index1",  "index2",  "index3",
    "middle1", "middle2", "middle3",
    "pinky1",  "pinky2",  "pinky3",
    "ring1",   "ring2",   "ring3",
    "thumb1",  "thumb2",  "thumb3",
]

# SMPL-X Blender 애드온 버전별 손가락 뼈 이름 패턴
# (버전마다 다르므로 여러 후보를 순서대로 시도)
def _finger_name_candidates(side: str, joint: str) -> List[str]:
    """
    joint 예시: "index1", "middle2", "pinky3", "thumb1"
    """
    finger = re.sub(r"\d+$", "", joint)   # "index", "middle", ...
    num    = re.sub(r"^\D+", "", joint)   # "1", "2", "3"
    cap    = side.capitalize()            # "Left" / "Right"

    # pinky = little 로도 표현됨
    little = "little" if finger == "pinky" else finger

    return [
        # SMPL-X 공식 애드온 (대부분 이 패턴)
        f"{side}_{finger}{num}",             # left_index1
        f"{side}_{little}{num}",             # left_little1  (pinky 별칭)
        # 일부 버전: _finger 접미사 포함
        f"{side}_{finger}_finger{num}",      # left_index_finger1
        f"{side}_{little}_finger{num}",      # left_little_finger1
        # Mixamo 스타일로 미리 변환된 경우
        f"{cap}Hand{finger.capitalize()}{num}",   # LeftHandIndex1
        f"{cap}HandPinky{num}" if finger == "pinky" else "",  # LeftHandPinky1
        # 짧은 접두사 스타일
        f"L_{finger.capitalize()}{num}" if side == "left"  else "",
        f"R_{finger.capitalize()}{num}" if side == "right" else "",
    ]


# ─────────────────────────────────────────────────────────────
#  유틸리티
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender -b template.blend -P script.py -- input.npz [output_dir] [options]")
        sys.exit(1)
    raw = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("input_npz")
    p.add_argument("output_dir", nargs="?", default=None)
    p.add_argument("--engine",        choices=["unreal", "unity"], default="unreal")
    p.add_argument("--fps",           type=float, default=30.0)
    p.add_argument("--max-frames",    type=int,   default=0)
    p.add_argument("--naming",        default="smplx")  # 호환성을 위해 추가
    p.add_argument("--remove-shape-keys", action="store_true")
    p.add_argument("--keep-temp",     action="store_true")
    p.add_argument("--print-bones",   action="store_true")
    return p.parse_args(raw)


def safe_makedirs(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def obj_names() -> set:
    return {o.name for o in bpy.data.objects}


def _rotvec_to_mat3(rv: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rv))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = rv / angle
    K = np.array(
        [[0, -axis[2], axis[1]],
         [axis[2], 0, -axis[0]],
         [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _mat3_to_rotvec(mat: np.ndarray) -> np.ndarray:
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
    axis = np.array(
        [mat[2, 1] - mat[1, 2],
         mat[0, 2] - mat[2, 0],
         mat[1, 0] - mat[0, 1]],
        dtype=np.float64,
    ) / denom
    return (axis * theta).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  NPZ → AMASS 포맷 변환 (Y-up → Z-up 좌표 변환 포함)
# ─────────────────────────────────────────────────────────────
def convert_to_amass(src: str, dst: str, fps: float, max_frames: int) -> int:
    data = np.load(src, allow_pickle=True)

    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        N = poses.shape[0]
        if poses.shape[1] < 165:
            padded = np.zeros((N, 165), dtype=np.float32)
            padded[:, :poses.shape[1]] = poses
            poses = padded
        else:
            poses = poses[:, :165]
    else:
        if "global_orient" not in data:
            raise KeyError("NPZ must contain 'poses' or 'global_orient'.")
        N = np.asarray(data["global_orient"]).shape[0]
        poses = np.zeros((N, 165), dtype=np.float32)
        poses[:, :3] = np.asarray(data["global_orient"], dtype=np.float32)
        if "body_pose" in data:
            bp = np.asarray(data["body_pose"], dtype=np.float32)
            poses[:, 3:3 + min(bp.shape[1], 63)] = bp[:, :63]
        for key, sl in [
            ("jaw_pose",  slice(66, 69)),
            ("leye_pose", slice(69, 72)),
            ("reye_pose", slice(72, 75)),
        ]:
            if key in data:
                poses[:, sl] = np.asarray(data[key], dtype=np.float32)[:, :3]
        if "left_hand_pose" in data:
            lh = np.asarray(data["left_hand_pose"], dtype=np.float32)
            poses[:, 75:75 + min(lh.shape[1], 45)] = lh[:, :45]
        if "right_hand_pose" in data:
            rh = np.asarray(data["right_hand_pose"], dtype=np.float32)
            poses[:, 120:120 + min(rh.shape[1], 45)] = rh[:, :45]

    if max_frames > 0:
        N = min(max_frames, poses.shape[0])
        poses = poses[:N]
    else:
        N = poses.shape[0]

    # Translation
    trans = np.zeros((N, 3), dtype=np.float32)
    for key in ("trans", "transl"):
        if key in data:
            raw = np.asarray(data[key], dtype=np.float32)
            trans[: min(N, len(raw))] = raw[: min(N, len(raw))]
            break

    # Y-up → Z-up 변환
    # Translation: (x, y, z) → (x, -z, y)
    trans = ((_R_Y2Z @ trans.astype(np.float64).T).T).astype(np.float32)

    # Global orient: R_new = R_y2z @ R_orig
    for i in range(N):
        rv = poses[i, :3].astype(np.float64)
        poses[i, :3] = _mat3_to_rotvec(_R_Y2Z @ _rotvec_to_mat3(rv))

    print(f"  [Y→Z] {N} frames converted (Y-up → Z-up)")

    betas = (
        np.asarray(data["betas"], dtype=np.float32)
        if "betas" in data
        else np.zeros(10, dtype=np.float32)
    )
    if betas.ndim == 2:
        betas = betas[0]

    gender_raw = data.get("gender", "neutral")
    if hasattr(gender_raw, "item"):
        gender_raw = gender_raw.item()

    np.savez(
        dst,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=str(gender_raw),
        mocap_frame_rate=float(fps),
    )
    return int(N)


# ─────────────────────────────────────────────────────────────
#  Blender 헬퍼
# ─────────────────────────────────────────────────────────────
def first_armature() -> Optional[bpy.types.Object]:
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def find_anim_armature(before: set, fallback) -> Optional[bpy.types.Object]:
    new = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.name not in before]
    if new:
        return new[-1]
    animated = [
        o for o in bpy.data.objects
        if o.type == "ARMATURE" and o.animation_data and o.animation_data.action
    ]
    return animated[-1] if animated else fallback


def meshes_of(arm: bpy.types.Object) -> List[bpy.types.Object]:
    seen, result = set(), []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == arm:
            if obj.name not in seen:
                result.append(obj)
                seen.add(obj.name)
            continue
        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and mod.object == arm:
                if obj.name not in seen:
                    result.append(obj)
                    seen.add(obj.name)
                break
    return result


def set_frame_range(
    arm: bpy.types.Object, fps: float, fallback: int
) -> Tuple[int, int]:
    bpy.context.scene.render.fps = int(round(fps))
    start, end = 1, max(1, fallback)
    if arm.animation_data and arm.animation_data.action:
        a, b = arm.animation_data.action.frame_range
        start, end = int(math.floor(a)), int(math.ceil(b))
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end   = end
    bpy.context.scene.frame_set(start)
    print(f"Frame range: {start}-{end}  fps={fps}")
    return start, end


def remove_shape_keys(mesh: bpy.types.Object) -> None:
    if mesh.type != "MESH" or not mesh.data.shape_keys:
        return
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    try:
        bpy.ops.object.shape_key_remove(all=True)
    except Exception as e:
        print(f"shape_key_remove failed on {mesh.name}: {e}")


def get_all_fcurves(action) -> List:
    """Blender 4.x / 5.x 모두 대응하는 fcurve 수집."""
    try:
        result = list(action.fcurves)
        if result:
            return result
    except (AttributeError, TypeError):
        pass

    fcurves = []

    if hasattr(action, "slots"):
        try:
            from bpy_extras import anim_utils
            for slot in action.slots:
                try:
                    cb = anim_utils.action_get_channelbag_for_slot(action, slot)
                    if cb and hasattr(cb, "fcurves"):
                        for fc in cb.fcurves:
                            if fc not in fcurves:
                                fcurves.append(fc)
                except Exception:
                    pass
        except Exception:
            pass

    if hasattr(action, "layers"):
        try:
            for layer in action.layers:
                for strip in getattr(layer, "strips", []):
                    for slot in getattr(action, "slots", []):
                        try:
                            cb = strip.channelbag(slot, ensure=False)
                            if cb and hasattr(cb, "fcurves"):
                                for fc in cb.fcurves:
                                    if fc not in fcurves:
                                        fcurves.append(fc)
                        except Exception:
                            pass
        except Exception:
            pass

    return fcurves


# ─────────────────────────────────────────────────────────────
#  손가락 키프레임 삽입
#  ※ bake_to_keyframes() 실행 후 적용해야 덮어써지지 않음!
# ─────────────────────────────────────────────────────────────
def apply_hand_keyframes(
    arm: bpy.types.Object,
    amass_npz_path: str,
    frame_start: int,
    frame_end: int,
) -> None:
    if not os.path.exists(amass_npz_path):
        print(f"[Hand] NPZ not found: {amass_npz_path}")
        return

    data = np.load(amass_npz_path, allow_pickle=True)
    if "poses" not in data:
        print("[Hand] 'poses' key not found, skipping hand keyframes")
        return

    poses = np.asarray(data["poses"], dtype=np.float64)
    N = poses.shape[0]

    left_hand  = poses[:, 75:120].reshape(N, 15, 3)
    right_hand = poses[:, 120:165].reshape(N, 15, 3)

    if np.allclose(left_hand, 0.0) and np.allclose(right_hand, 0.0):
        print("[Hand] All hand pose data is zero, skipping")
        return

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")

    finger_bones: Dict[str, bpy.types.PoseBone] = {}
    found_patterns: Dict[str, str] = {}

    for side in ("left", "right"):
        for j, joint in enumerate(HAND_JOINT_NAMES):
            key = f"{side}_{joint}"
            for candidate in _finger_name_candidates(side, joint):
                if not candidate:
                    continue
                pbone = arm.pose.bones.get(candidate)
                if pbone is not None:
                    pbone.rotation_mode = "QUATERNION"
                    finger_bones[key] = pbone
                    found_patterns[key] = candidate
                    break

    if not finger_bones:
        print("[Hand] ❌ No finger bones found. Bone names in this armature:")
        for b in arm.pose.bones:
            print(f"    {b.name}")
        bpy.ops.object.mode_set(mode="OBJECT")
        return

    print(
        f"[Hand] Found {len(finger_bones)}/30 finger bones "
        f"(sample: {next(iter(found_patterns.values()))})"
    )

    num_frames = min(N, frame_end - frame_start + 1)

    for frame_idx in range(num_frames):
        frame = frame_start + frame_idx

        for j, joint in enumerate(HAND_JOINT_NAMES):
            for side, hand_data in [("left", left_hand), ("right", right_hand)]:
                key = f"{side}_{joint}"
                pbone = finger_bones.get(key)
                if pbone is None:
                    continue

                rv    = hand_data[frame_idx, j]
                angle = float(np.linalg.norm(rv))

                if angle < 1e-8:
                    quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
                else:
                    ax   = rv / angle
                    quat = mathutils.Quaternion(
                        mathutils.Vector((ax[0], ax[1], ax[2])), angle
                    )

                pbone.rotation_quaternion = quat
                pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        if frame_idx % 200 == 0:
            print(f"  [Hand] {frame_idx}/{num_frames}")

    print(f"  [Hand] Done: {num_frames} frames")

    if arm.animation_data and arm.animation_data.action:
        finger_fc = sum(
            1 for fc in get_all_fcurves(arm.animation_data.action)
            if any(
                kw in fc.data_path
                for kw in ("index", "middle", "ring", "pinky", "thumb", "little")
            )
        )
        print(f"  [Hand] Finger fcurves in action: {finger_fc}")

    bpy.ops.object.mode_set(mode="OBJECT")


# ─────────────────────────────────────────────────────────────
#  NLA Bake (Blender 5.x 슬롯 기반 Action → 전통 키프레임)
# ─────────────────────────────────────────────────────────────
def bake_to_keyframes(
    arm: bpy.types.Object,
    frame_start: int,
    frame_end: int,
) -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")

    bake_kwargs = dict(
        frame_start=frame_start,
        frame_end=frame_end,
        step=1,
        only_selected=False,
        visual_keying=True,
        clear_constraints=False,
        clear_parents=False,
        use_current_action=True,
        bake_types={"POSE"},
    )

    try:
        bpy.ops.nla.bake(**bake_kwargs, channel_types={"LOCATION", "ROTATION", "SCALE"})
    except TypeError:
        bpy.ops.nla.bake(**bake_kwargs)

    bpy.ops.object.mode_set(mode="OBJECT")

    if arm.animation_data and arm.animation_data.action:
        act = arm.animation_data.action
        try:
            fc_count = len(list(act.fcurves))
        except (AttributeError, TypeError):
            fc_count = len(get_all_fcurves(act))
        print(f"[Bake] action={act.name}, fcurves={fc_count}, range={frame_start}-{frame_end}")
    else:
        print("[Bake] WARNING: no action after bake")


# ─────────────────────────────────────────────────────────────
#  FBX 내보내기
# ─────────────────────────────────────────────────────────────
def export_fbx(path: str, engine: str, anim_only: bool = False) -> None:
    preset = EXPORT_AXIS[engine]
    obj_types = {"ARMATURE"} if anim_only else {"ARMATURE", "MESH"}
    tag = "Anim-Only" if anim_only else "Full"
    print(f"\n[FBX Export ({tag})] {path}  engine={engine}")
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        object_types=obj_types,
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        use_subsurf=False,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        axis_forward=preset["axis_forward"],
        axis_up=preset["axis_up"],
        bake_space_transform=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="COPY",
        embed_textures=False,
    )


def select_for_export(
    arm: bpy.types.Object, meshes: List[bpy.types.Object]
) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = arm


def rename_objects(
    arm: bpy.types.Object, meshes: List[bpy.types.Object]
) -> None:
    arm.name      = "SMPLXRig"
    arm.data.name = "SMPLXArmature"
    for i, mesh in enumerate(meshes):
        mesh.name      = "SMPLXBody" if i == 0 else f"SMPLXMesh_{i:02d}"
        mesh.data.name = mesh.name + "Mesh"


# ─────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    input_npz  = os.path.abspath(args.input_npz)
    output_dir = (
        os.path.abspath(args.output_dir) if args.output_dir
        else os.path.dirname(input_npz)
    )
    safe_makedirs(output_dir)

    base    = os.path.splitext(os.path.basename(input_npz))[0]
    suffix  = EXPORT_AXIS[args.engine]["axis_up"].lower()
    
    # 🌟 수정: FBX 출력 이름을 원본 스크립트가 기대하는 형태(smplx_unreal)로 맞춤
    out_fbx = os.path.join(output_dir, f"{base}_smplx_unreal.fbx")
    tmp_npz = os.path.join(output_dir, f"{base}_amass_tmp.npz")

    print("=" * 60)
    print("SMPL-X → FBX Exporter (bone names preserved)")
    print("=" * 60)
    print(f"Input   : {input_npz}")
    print(f"Output  : {out_fbx}")
    print(f"Engine  : {args.engine}")
    print("=" * 60)

    if not os.path.exists(input_npz):
        print(f"Error: {input_npz} not found")
        sys.exit(1)

    # ── 1. NPZ → AMASS 변환 ─────────────────────────────────────────────
    try:
        num_frames = convert_to_amass(input_npz, tmp_npz, args.fps, args.max_frames)
        print(f"Frames: {num_frames}")
    except Exception as e:
        print(f"NPZ conversion error: {e}")
        sys.exit(1)

    # ── 2. SMPL-X 애드온으로 애니메이션 로드 ────────────────────────────
    template_arm = first_armature()
    if template_arm is None:
        print("Error: no armature in .blend file")
        sys.exit(1)

    before = obj_names()
    bpy.ops.object.select_all(action="DESELECT")
    template_arm.select_set(True)
    bpy.context.view_layer.objects.active = template_arm

    try:
        if not hasattr(bpy.ops.object, "smplx_add_animation"):
            print("Error: SMPL-X Blender add-on not installed")
            sys.exit(1)
        bpy.ops.object.smplx_add_animation(filepath=tmp_npz, anim_format="AMASS")
    except Exception as e:
        print(f"smplx_add_animation error: {e}")
        sys.exit(1)

    src_arm = find_anim_armature(before, fallback=template_arm)
    if src_arm is None:
        print("Error: animated armature not found")
        sys.exit(1)

    meshes = meshes_of(src_arm)
    if not meshes:
        print("Warning: no mesh found — skeleton-only FBX will be generated.")

    frame_start, frame_end = set_frame_range(src_arm, args.fps, num_frames)

    if args.print_bones:
        print("[Bones in armature]")
        for b in src_arm.data.bones:
            print(f"  {b.name}")

    if args.remove_shape_keys:
        for m in meshes:
            remove_shape_keys(m)

    rename_objects(src_arm, meshes)

    # ── 3. NLA Bake (Blender 5.x 슬롯 기반 Action 대응) ─────────────────
    # 🌟 수정: 몸체 뼈를 먼저 전통적 프레임으로 굽습니다.
    print("\n[Step 3] Baking animation to traditional keyframes...")
    bake_to_keyframes(src_arm, frame_start, frame_end)

    # ── 4. 손가락 키프레임 삽입 (bake 이후!) ───────────────────────
    # 🌟 수정: 굽기가 끝난 Action 위에 손가락을 삽입해야 덮어써지지 않고 유지됩니다.
    print("\n[Step 4] Applying hand pose keyframes from NPZ...")
    apply_hand_keyframes(src_arm, tmp_npz, frame_start, frame_end)

    # ── 5. tmp NPZ 정리 ──────────────────────────────────────────────────
    if os.path.exists(tmp_npz) and not args.keep_temp:
        try:
            os.remove(tmp_npz)
        except OSError:
            pass

    # ── 6. FBX 내보내기 ──────────────────────────────────────────────────
    select_for_export(src_arm, meshes)

    try:
        export_fbx(out_fbx, args.engine, anim_only=False)
    except Exception as e:
        print(f"FBX export error: {e}")
        sys.exit(1)

    # 애니메이션 전용 FBX (언리얼에서 Animation Sequence로 임포트할 때 사용)
    anim_fbx = out_fbx.replace(".fbx", "_anim.fbx")
    try:
        bpy.ops.object.select_all(action="DESELECT")
        src_arm.select_set(True)
        bpy.context.view_layer.objects.active = src_arm
        export_fbx(anim_fbx, args.engine, anim_only=True)
    except Exception as e:
        print(f"Animation-only FBX export error: {e}")

    print("=" * 60)
    print("Done:")
    print(f"  Full FBX : {out_fbx}")
    print(f"  Anim FBX : {anim_fbx}")
    print("=" * 60)


if __name__ == "__main__":
    main()
