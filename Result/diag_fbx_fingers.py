"""Check actual finger animation in the exported FBX file using Blender."""
import bpy
import sys
import math

argv = sys.argv
if "--" not in argv:
    sys.exit(1)
args = argv[argv.index("--") + 1:]
fbx_path = args[0]

# Clean scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX
bpy.ops.import_scene.fbx(filepath=fbx_path)

arm = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        arm = obj
        break

if not arm:
    print("ERROR: No armature found")
    sys.exit(1)

print(f"Armature: {arm.name}")
print(f"Bones: {len(arm.data.bones)}")

# List all finger bones
finger_keywords = ['index', 'middle', 'ring', 'pinky', 'thumb', 'little']
finger_bones = []
for bone in arm.data.bones:
    if any(kw in bone.name.lower() for kw in finger_keywords):
        finger_bones.append(bone.name)

print(f"\nFinger bones found ({len(finger_bones)}):")
for b in sorted(finger_bones):
    print(f"  {b}")

# Check animation data
if not arm.animation_data:
    print("\nERROR: No animation_data on armature")
    sys.exit(1)

action = arm.animation_data.action
if not action:
    print("\nERROR: No action on armature")
    sys.exit(1)

print(f"\nAction: {action.name}")

# Try to get fcurves (Blender 5.x compatible)
fcurves = []
try:
    fcurves = list(action.fcurves)
except (AttributeError, TypeError):
    pass

if not fcurves:
    try:
        from bpy_extras import anim_utils
        for slot in action.slots:
            cb = anim_utils.action_get_channelbag_for_slot(action, slot)
            if cb and hasattr(cb, 'fcurves'):
                fcurves.extend(cb.fcurves)
    except:
        pass

if not fcurves:
    try:
        for layer in action.layers:
            for strip in getattr(layer, 'strips', []):
                for slot in getattr(action, 'slots', []):
                    try:
                        cb = strip.channelbag(slot, ensure=False)
                        if cb:
                            fcurves.extend(cb.fcurves)
                    except:
                        pass
    except:
        pass

print(f"Total fcurves: {len(fcurves)}")

# Analyze finger fcurves
finger_fcurves = [fc for fc in fcurves if any(kw in fc.data_path for kw in finger_keywords)]
print(f"Finger fcurves: {len(finger_fcurves)}")

if finger_fcurves:
    print("\nFinger fcurve details:")
    for fc in sorted(finger_fcurves, key=lambda x: x.data_path)[:30]:
        kps = fc.keyframe_points if hasattr(fc, 'keyframe_points') else []
        vals = [kp.co[1] for kp in kps]
        if vals:
            print(f"  {fc.data_path}[{fc.array_index}]: "
                  f"keyframes={len(vals)}, min={min(vals):.4f}, max={max(vals):.4f}, "
                  f"range={max(vals)-min(vals):.4f}")
        else:
            print(f"  {fc.data_path}[{fc.array_index}]: no keyframes")

# Also check by directly sampling the pose bones
print("\n\nDirect pose bone sampling (frame 1 vs frame 500):")
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()

frame1_rots = {}
for pb in arm.pose.bones:
    if any(kw in pb.name.lower() for kw in finger_keywords):
        pb.rotation_mode = 'QUATERNION'
        frame1_rots[pb.name] = pb.matrix_basis.to_quaternion().copy()

bpy.context.scene.frame_set(500)
bpy.context.view_layer.update()

print(f"{'Bone':<25s} {'Frame1 Quat':<45s} {'Frame500 Quat':<45s} {'Diff(deg)'}")
for pb in arm.pose.bones:
    if pb.name in frame1_rots:
        q1 = frame1_rots[pb.name]
        q2 = pb.matrix_basis.to_quaternion()
        diff = q1.rotation_difference(q2).angle
        diff_deg = math.degrees(diff)
        if diff_deg > 0.01:
            print(f"  {pb.name:<23s} {str(tuple(round(x,3) for x in q1)):<45s} "
                  f"{str(tuple(round(x,3) for x in q2)):<45s} {diff_deg:.2f}")
