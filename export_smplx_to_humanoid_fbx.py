from __future__ import annotations

import argparse
import os
import sys
import re
import math
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
import mathutils


SMPLX_TO_MIXAMO: Dict[str, str] = {
    "pelvis":        "Hips",
    "hips":          "Hips",
    "spine1":        "Spine",
    "spine_1":       "Spine",
    "spine2":        "Spine1",
    "spine_2":       "Spine1",
    "spine3":        "Spine2",
    "spine_3":       "Spine2",
    "neck":          "Neck",
    "head":          "Head",
    "left_hip":      "LeftUpLeg",
    "leftupleg":     "LeftUpLeg",
    "left_knee":     "LeftLeg",
    "leftleg":       "LeftLeg",
    "left_ankle":    "LeftFoot",
    "leftfoot":      "LeftFoot",
    "left_foot":     "LeftToeBase",
    "lefttoebase":   "LeftToeBase",
    "right_hip":     "RightUpLeg",
    "rightupleg":    "RightUpLeg",
    "right_knee":    "RightLeg",
    "rightleg":      "RightLeg",
    "right_ankle":   "RightFoot",
    "rightfoot":     "RightFoot",
    "right_foot":    "RightToeBase",
    "righttoebase":  "RightToeBase",
    "left_collar":   "LeftShoulder",
    "left_shoulder": "LeftArm",
    "left_elbow":    "LeftForeArm",
    "left_wrist":    "LeftHand",
    "left_hand":     "LeftHand",
    "right_collar":  "RightShoulder",
    "right_shoulder":"RightArm",
    "right_elbow":   "RightForeArm",
    "right_wrist":   "RightHand",
    "right_hand":    "RightHand",
}

SMPLX_TO_UNITY: Dict[str, str] = {
    "pelvis":        "Hips",
    "hips":          "Hips",
    "spine1":        "Spine",
    "spine_1":       "Spine",
    "spine2":        "Chest",
    "spine_2":       "Chest",
    "spine3":        "UpperChest",
    "spine_3":       "UpperChest",
    "neck":          "Neck",
    "head":          "Head",
    "left_hip":      "LeftUpperLeg",
    "left_knee":     "LeftLowerLeg",
    "left_ankle":    "LeftFoot",
    "left_foot":     "LeftToes",
    "right_hip":     "RightUpperLeg",
    "right_knee":    "RightLowerLeg",
    "right_ankle":   "RightFoot",
    "right_foot":    "RightToes",
    "left_collar":   "LeftShoulder",
    "left_shoulder": "LeftUpperArm",
    "left_elbow":    "LeftLowerArm",
    "left_wrist":    "LeftHand",
    "right_collar":  "RightShoulder",
    "right_shoulder":"RightUpperArm",
    "right_elbow":   "RightLowerArm",
    "right_wrist":   "RightHand",
}

FINGER_MAP: Dict[str, str] = {
    f"left_{finger}{i}":  f"LeftHand{finger.capitalize()}{i}"
    for finger in ("thumb", "index", "middle", "ring", "pinky", "little")
    for i in (1, 2, 3)
} | {
    f"right_{finger}{i}": f"RightHand{finger.capitalize()}{i}"
    for finger in ("thumb", "index", "middle", "ring", "pinky", "little")
    for i in (1, 2, 3)
}
for side in ("left", "right"):
    cap = side.capitalize()
    for i in (1, 2, 3):
        FINGER_MAP[f"{side}_little{i}"] = f"{cap}HandPinky{i}"

EXPORT_AXIS: Dict[str, Dict] = {
    "unreal": {"axis_forward": "X",  "axis_up": "Z", "suffix": "unreal"},
    "unity":  {"axis_forward": "-Z", "axis_up": "Y", "suffix": "unity"},
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender -b template.blend -P script.py -- input.npz [output_dir] [options]")
        sys.exit(1)
    raw = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("input_npz")
    p.add_argument("output_dir", nargs="?", default=None)
    p.add_argument("--target-fbx", default=None)
    p.add_argument("--engine", choices=["unreal", "unity"], default="unreal")
    p.add_argument("--naming", choices=["mixamo", "unity"], default="mixamo")
    p.add_argument("--mixamo-prefix", default="")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--remove-shape-keys", action="store_true")
    p.add_argument("--print-bones", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    return p.parse_args(raw)


def norm(name: str) -> str:
    s = name.strip().split(":")[-1]
    s = re.sub(r"\.\d+$", "", s)
    s = s.replace(" ", "_").replace("-", "_").replace(".", "_")
    s = re.sub(r"__+", "_", s)
    return s.lower()


def safe_makedirs(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def obj_names() -> set:
    return {o.name for o in bpy.data.objects}


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
        for key, sl in [("jaw_pose", slice(66, 69)), ("leye_pose", slice(69, 72)),
                        ("reye_pose", slice(72, 75))]:
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

    trans = np.zeros((N, 3), dtype=np.float32)
    for key in ("trans", "transl"):
        if key in data:
            raw = np.asarray(data[key], dtype=np.float32)
            trans[:min(N, len(raw))] = raw[:min(N, len(raw))]
            break

    betas = np.asarray(data["betas"], dtype=np.float32) if "betas" in data else np.zeros(10, dtype=np.float32)
    if betas.ndim == 2:
        betas = betas[0]

    gender_raw = data["gender"] if "gender" in data else "neutral"
    if hasattr(gender_raw, "item"):
        gender_raw = gender_raw.item()
    gender = str(gender_raw)

    np.savez(dst, poses=poses, trans=trans, betas=betas,
             gender=gender, mocap_frame_rate=float(fps))
    return int(N)


def first_armature() -> Optional[bpy.types.Object]:
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def find_anim_armature(before: set, fallback) -> Optional[bpy.types.Object]:
    new = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.name not in before]
    if new:
        return new[-1]
    animated = [o for o in bpy.data.objects
                if o.type == "ARMATURE" and o.animation_data and o.animation_data.action]
    return animated[-1] if animated else fallback


def meshes_of(arm: bpy.types.Object) -> List[bpy.types.Object]:
    seen, result = set(), []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == arm:
            if obj.name not in seen:
                result.append(obj); seen.add(obj.name)
            continue
        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and mod.object == arm:
                if obj.name not in seen:
                    result.append(obj); seen.add(obj.name)
                break
    return result


def set_frame_range(arm: bpy.types.Object, fps: float, fallback: int) -> Tuple[int, int]:
    bpy.context.scene.render.fps = int(round(fps))
    start, end = 1, max(1, fallback)
    if arm.animation_data and arm.animation_data.action:
        a, b = arm.animation_data.action.frame_range
        start, end = int(math.floor(a)), int(math.ceil(b))
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
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


def build_alias_map(naming: str, prefix: str) -> Dict[str, str]:
    base = dict(SMPLX_TO_MIXAMO if naming == "mixamo" else SMPLX_TO_UNITY)
    if naming == "mixamo":
        base.update(FINGER_MAP)
    if prefix:
        return {k: f"{prefix}{v}" for k, v in base.items()}
    return base


def detect_renames(arm: bpy.types.Object, alias_map: Dict[str, str]) -> Dict[str, str]:
    norm_to_actual = {norm(b.name): b.name for b in arm.data.bones}
    return {
        norm_to_actual[norm(alias)]: target
        for alias, target in alias_map.items()
        if norm(alias) in norm_to_actual
    }


def get_all_fcurves(action) -> List:
    fcurves = []
    if hasattr(action, "fcurves") and action.fcurves is not None:
        try:
            return list(action.fcurves)
        except AttributeError:
            pass

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


def patch_fcurves(arm: bpy.types.Object, mapping: Dict[str, str]) -> None:
    actions = []
    if arm.animation_data:
        if arm.animation_data.action:
            actions.append(arm.animation_data.action)
        for track in arm.animation_data.nla_tracks:
            for strip in track.strips:
                if strip.action:
                    actions.append(strip.action)
    seen, unique = set(), []
    for a in actions:
        if a.name not in seen:
            unique.append(a); seen.add(a.name)
    for action in unique:
        for fc in get_all_fcurves(action):
            dp = fc.data_path
            for old, new in mapping.items():
                dp = dp.replace(f'pose.bones["{old}"]', f'pose.bones["{new}"]')
            fc.data_path = dp


def rename_vgroups(meshes: List[bpy.types.Object], mapping: Dict[str, str]) -> None:
    for mesh in meshes:
        for old, new in mapping.items():
            vg = mesh.vertex_groups.get(old)
            if vg:
                vg.name = new


def rename_bones_fallback(arm: bpy.types.Object,
                           meshes: List[bpy.types.Object],
                           naming: str, prefix: str,
                           print_bones: bool) -> Dict[str, str]:
    if print_bones:
        print("[Bones BEFORE rename]")
        for b in arm.data.bones:
            print(" ", b.name)

    alias_map = build_alias_map(naming, prefix)
    mapping = detect_renames(arm, alias_map)
    if not mapping:
        print("Warning: no matching bones found, skipping rename.")
        return {}

    existing = {b.name for b in arm.data.bones}
    final = {old: new for old, new in mapping.items()
             if old != new and not (new in existing and new not in mapping)}

    rename_vgroups(meshes, final)
    for old, new in final.items():
        b = arm.data.bones.get(old)
        if b:
            b.name = new
    patch_fcurves(arm, final)

    print(f"[Name remap] {len(final)} bones renamed ({naming} preset)")
    for old, new in sorted(final.items()):
        print(f"  {old} -> {new}")
    if print_bones:
        print("[Bones AFTER rename]")
        for b in arm.data.bones:
            print(" ", b.name)
    return final


def import_target_fbx(target_fbx_path: str) -> Optional[bpy.types.Object]:
    before = obj_names()
    try:
        bpy.ops.import_scene.fbx(
            filepath=target_fbx_path,
            use_anim=False,
            ignore_leaf_bones=True,
            force_connect_children=False,
            automatic_bone_orientation=False,
            primary_bone_axis="Y",
            secondary_bone_axis="X",
        )
    except Exception as e:
        print(f"Target FBX import failed: {e}")
        return None

    new_arms = [o for o in bpy.data.objects
                if o.type == "ARMATURE" and o.name not in before]
    if not new_arms:
        print("Warning: no armature found in target FBX.")
        return None

    target_arm = new_arms[-1]
    print(f"Target armature loaded: {target_arm.name}")
    return target_arm


def build_source_target_bone_map(
    src_arm: bpy.types.Object,
    tgt_arm: bpy.types.Object,
    naming: str,
    prefix: str,
) -> Dict[str, str]:
    alias_map = build_alias_map(naming, prefix)

    src_norm_to_actual = {norm(b.name): b.name for b in src_arm.data.bones}
    src_to_humanoid: Dict[str, str] = {}
    for alias, humanoid in alias_map.items():
        a_norm = norm(alias)
        if a_norm in src_norm_to_actual:
            src_to_humanoid[src_norm_to_actual[a_norm]] = humanoid

    tgt_bones = {b.name: b.name for b in tgt_arm.data.bones}
    tgt_norm_to_actual = {norm(b.name): b.name for b in tgt_arm.data.bones}

    result: Dict[str, str] = {}
    unmatched = []

    for src_bone, humanoid in src_to_humanoid.items():
        if humanoid in tgt_bones:
            result[src_bone] = humanoid
            continue
        h_norm = norm(humanoid)
        if h_norm in tgt_norm_to_actual:
            result[src_bone] = tgt_norm_to_actual[h_norm]
            continue
        if prefix:
            candidate = f"{prefix}{humanoid}"
            if candidate in tgt_bones:
                result[src_bone] = candidate
                continue
            if norm(candidate) in tgt_norm_to_actual:
                result[src_bone] = tgt_norm_to_actual[norm(candidate)]
                continue
        unmatched.append(f"{src_bone} -> {humanoid}")

    print(f"[Retarget] {len(result)} bone pairs matched")
    if unmatched:
        print(f"  unmatched {len(unmatched)}: {unmatched[:10]}")
    return result


ROOT_BONE_NORMS = frozenset({"hips", "pelvis", "mixamorighips"})


def apply_retarget_constraints(
    src_arm: bpy.types.Object,
    tgt_arm: bpy.types.Object,
    bone_map: Dict[str, str],
) -> List[str]:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode="POSE")

    constrained: List[str] = []
    tgt_to_src = {v: k for k, v in bone_map.items()}

    for tgt_bone_name, src_bone_name in tgt_to_src.items():
        pose_bone = tgt_arm.pose.bones.get(tgt_bone_name)
        if pose_bone is None:
            continue

        if norm(tgt_bone_name) in ROOT_BONE_NORMS:
            c_loc = pose_bone.constraints.new("COPY_LOCATION")
            c_loc.name = "SMPLX_CopyLoc"
            c_loc.target = src_arm
            c_loc.subtarget = src_bone_name
            c_loc.use_offset = False

        c_rot = pose_bone.constraints.new("COPY_ROTATION")
        c_rot.name = "SMPLX_CopyRot"
        c_rot.target = src_arm
        c_rot.subtarget = src_bone_name
        c_rot.mix_mode = "REPLACE"
        c_rot.target_space = "POSE"
        c_rot.owner_space  = "POSE"

        constrained.append(tgt_bone_name)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[Retarget] constraints applied: {len(constrained)} target bones")
    return constrained


def bake_animation_to_target(
    tgt_arm: bpy.types.Object,
    frame_start: int,
    frame_end: int,
) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")

    bake_kwargs = dict(
        frame_start=frame_start,
        frame_end=frame_end,
        step=1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=False,
        bake_types={"POSE"},
    )

    try:
        bpy.ops.nla.bake(**bake_kwargs, channel_types={"LOCATION", "ROTATION", "SCALE"})
    except TypeError:
        bpy.ops.nla.bake(**bake_kwargs)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[Retarget] animation bake complete ({frame_start}-{frame_end})")


def remove_constraints_manually(tgt_arm: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode="POSE")
    for pbone in tgt_arm.pose.bones:
        for c in list(pbone.constraints):
            if c.name.startswith("SMPLX_"):
                pbone.constraints.remove(c)
    bpy.ops.object.mode_set(mode="OBJECT")


def delete_armature_and_children(arm: bpy.types.Object) -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for child in list(arm.children):
        if child.type == "MESH" and not child.select_get():
            child.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.delete(use_global=False)


def retarget_to_target_armature(
    src_arm: bpy.types.Object,
    target_fbx_path: str,
    meshes: List[bpy.types.Object],
    frame_start: int,
    frame_end: int,
    naming: str,
    prefix: str,
) -> Optional[bpy.types.Object]:
    print("\n[Retarget pipeline start]")

    tgt_arm = import_target_fbx(target_fbx_path)
    if tgt_arm is None:
        return None

    bone_map = build_source_target_bone_map(src_arm, tgt_arm, naming, prefix)
    if not bone_map:
        print("Error: bone map is empty, aborting retarget.")
        return None

    apply_retarget_constraints(src_arm, tgt_arm, bone_map)

    src_arm.hide_set(False)
    src_arm.hide_viewport = False

    bake_animation_to_target(tgt_arm, frame_start, frame_end)
    remove_constraints_manually(tgt_arm)

    vgroup_mapping = {src_bone: tgt_bone for src_bone, tgt_bone in bone_map.items()}
    rename_vgroups(meshes, vgroup_mapping)

    for mesh in meshes:
        mesh.parent = tgt_arm
        for mod in mesh.modifiers:
            if mod.type == "ARMATURE":
                mod.object = tgt_arm

    src_meshes_to_delete = []
    for child in list(src_arm.children):
        if child.type == "MESH" and child not in meshes:
            src_meshes_to_delete.append(child)

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    src_arm.select_set(True)
    for m in src_meshes_to_delete:
        m.select_set(True)
    bpy.context.view_layer.objects.active = src_arm
    bpy.ops.object.delete(use_global=False)

    tgt_mesh_children = [o for o in bpy.data.objects
                         if o.type == "MESH" and o.parent == tgt_arm
                         and o not in meshes]
    if tgt_mesh_children:
        bpy.ops.object.select_all(action="DESELECT")
        for m in tgt_mesh_children:
            m.select_set(True)
        bpy.ops.object.delete(use_global=False)

    print("[Retarget pipeline complete]")
    return tgt_arm


def rename_objects(arm: bpy.types.Object, meshes: List[bpy.types.Object]) -> None:
    arm.name = "HumanoidRig"
    arm.data.name = "HumanoidArmature"
    for i, mesh in enumerate(meshes):
        mesh.name = "HumanoidBody" if i == 0 else f"HumanoidMesh_{i:02d}"
        mesh.data.name = mesh.name + "Mesh"


def bake_to_keyframes(
    arm: bpy.types.Object,
    frame_start: int,
    frame_end: int,
) -> None:
    """Blender 5.x 슬롯 기반 Action을 전통적 키프레임으로 변환.

    smplx_add_animation이 생성한 슬롯 기반 fcurves는
    FBX exporter가 제대로 처리하지 못할 수 있으므로,
    NLA bake로 전통적 pose bone 키프레임을 강제로 작성한다.
    """
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

    # 결과 확인
    if arm.animation_data and arm.animation_data.action:
        act = arm.animation_data.action
        fc_count = 0
        try:
            fc_count = len(list(act.fcurves))
        except (AttributeError, TypeError):
            pass
        print(f"[Bake] action={act.name}, fcurves={fc_count}, "
              f"range={frame_start}-{frame_end}")
    else:
        print("[Bake] WARNING: no action after bake")


def export_fbx(path: str, engine: str, anim_only: bool = False) -> None:
    preset = EXPORT_AXIS[engine]
    obj_types = {"ARMATURE"} if anim_only else {"ARMATURE", "MESH"}
    tag = "Animation-Only" if anim_only else "Full"
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
        bake_space_transform=False,
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


def select_for_export(arm: bpy.types.Object, meshes: List[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = arm


def main() -> None:
    args = parse_args()

    input_npz  = os.path.abspath(args.input_npz)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir \
                 else os.path.dirname(input_npz)
    safe_makedirs(output_dir)

    base    = os.path.splitext(os.path.basename(input_npz))[0]
    mode    = "retarget" if args.target_fbx else args.naming
    suffix  = EXPORT_AXIS[args.engine]["suffix"]
    out_fbx = os.path.join(output_dir, f"{base}_{mode}_{suffix}.fbx")
    tmp_npz = os.path.join(output_dir, f"{base}_amass_tmp.npz")

    print("=" * 60)
    print("SMPL-X to Humanoid FBX Exporter v2")
    print("=" * 60)
    print(f"Input     : {input_npz}")
    print(f"Output    : {out_fbx}")
    print(f"Engine    : {args.engine}")
    print(f"Naming    : {args.naming}")
    print(f"Target FBX: {args.target_fbx or 'None (name mapping fallback)'}")
    print("=" * 60)

    if not os.path.exists(input_npz):
        print(f"Error: {input_npz} not found"); sys.exit(1)
    if args.target_fbx and not os.path.exists(args.target_fbx):
        print(f"Error: target FBX {args.target_fbx} not found"); sys.exit(1)

    try:
        num_frames = convert_to_amass(input_npz, tmp_npz, args.fps, args.max_frames)
        print(f"Frames: {num_frames}")
    except Exception as e:
        print(f"NPZ conversion error: {e}"); sys.exit(1)

    template_arm = first_armature()
    if template_arm is None:
        print("Error: no armature in .blend file"); sys.exit(1)

    before = obj_names()
    bpy.ops.object.select_all(action="DESELECT")
    template_arm.select_set(True)
    bpy.context.view_layer.objects.active = template_arm

    try:
        if not hasattr(bpy.ops.object, "smplx_add_animation"):
            print("Error: SMPL-X Blender add-on not installed"); sys.exit(1)
        bpy.ops.object.smplx_add_animation(filepath=tmp_npz, anim_format="AMASS")
    except Exception as e:
        print(f"smplx_add_animation error: {e}"); sys.exit(1)
    finally:
        if os.path.exists(tmp_npz) and not args.keep_temp:
            try:
                os.remove(tmp_npz)
            except OSError:
                pass

    src_arm = find_anim_armature(before, fallback=template_arm)
    if src_arm is None:
        print("Error: animated armature not found"); sys.exit(1)

    meshes = meshes_of(src_arm)
    if not meshes:
        print("Warning: no mesh found - skeleton only FBX will be generated.")

    frame_start, frame_end = set_frame_range(src_arm, args.fps, num_frames)

    if args.remove_shape_keys:
        for m in meshes:
            remove_shape_keys(m)

    export_arm = src_arm

    if args.target_fbx:
        result = retarget_to_target_armature(
            src_arm=src_arm,
            target_fbx_path=os.path.abspath(args.target_fbx),
            meshes=meshes,
            frame_start=frame_start,
            frame_end=frame_end,
            naming=args.naming,
            prefix=args.mixamo_prefix,
        )
        if result is None:
            print("Retarget failed. Falling back to name mapping.")
            rename_bones_fallback(src_arm, meshes, args.naming,
                                  args.mixamo_prefix, args.print_bones)
        else:
            export_arm = result
            meshes = meshes_of(export_arm)
    else:
        rename_bones_fallback(src_arm, meshes, args.naming,
                              args.mixamo_prefix, args.print_bones)

    rename_objects(export_arm, meshes)

    # ── Blender 5.x: 슬롯 기반 Action → 전통적 키프레임으로 bake ──
    print("\n[Pre-export] Baking animation to traditional keyframes...")
    bake_to_keyframes(export_arm, frame_start, frame_end)

    select_for_export(export_arm, meshes)

    # ── 전체 FBX (Skeletal Mesh + Animation) ──
    try:
        export_fbx(out_fbx, args.engine, anim_only=False)
    except Exception as e:
        print(f"FBX export error: {e}"); sys.exit(1)

    # ── 애니메이션 전용 FBX (Skeleton + Animation만, 메시 없음) ──
    anim_fbx = out_fbx.replace(".fbx", "_anim.fbx")
    try:
        # armature만 선택
        bpy.ops.object.select_all(action="DESELECT")
        export_arm.select_set(True)
        bpy.context.view_layer.objects.active = export_arm
        export_fbx(anim_fbx, args.engine, anim_only=True)
    except Exception as e:
        print(f"Animation-only FBX export error: {e}")

    print("=" * 60)
    print(f"Done:")
    print(f"  Full FBX : {out_fbx}")
    print(f"  Anim FBX : {anim_fbx}")
    if not args.target_fbx:
        print("Note: name mapping fallback mode - use --target-fbx for full Mixamo compatibility.")
    print("Tip: 언리얼에서 Anim FBX를 임포트하면 Animation Sequence로 인식됩니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
