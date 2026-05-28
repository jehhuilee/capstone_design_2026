import bpy
import sys
import os

argv = sys.argv
if "--" not in argv:
    sys.exit(1)

args = argv[argv.index("--") + 1:]
fbx_path = os.path.abspath(args[0])

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.import_scene.fbx(filepath=fbx_path)

arm = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        arm = obj
        break

if not arm:
    print("No armature found")
    sys.exit(1)

print("Bone Names:")
for b in arm.data.bones:
    if "Hand" in b.name or "Index" in b.name or "Thumb" in b.name:
        print("  " + b.name)

action = None
if arm.animation_data and arm.animation_data.action:
    action = arm.animation_data.action

print(f"\nAction: {action.name if action else 'None'}")
if action:
    count = 0
    finger_fc = 0
    for fc in action.fcurves:
        count += 1
        if "Hand" in fc.data_path or "Index" in fc.data_path or "Thumb" in fc.data_path:
            finger_fc += 1
    print(f"Total F-Curves: {count}")
    print(f"Finger F-Curves: {finger_fc}")
