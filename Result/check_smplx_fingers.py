import bpy
import sys
import math

argv = sys.argv
if '--' not in argv: sys.exit(1)
args = argv[argv.index('--') + 1:]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=args[0])

arm = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE': arm = obj; break

if not arm or not arm.animation_data or not arm.animation_data.action:
    print('No animation')
    sys.exit()

action = arm.animation_data.action
rotations = {}
for fc in action.fcurves:
    if 'pose.bones' in fc.data_path and 'rotation_quaternion' in fc.data_path:
        bone_name = fc.data_path.split('"')[1]
        if 'index' in bone_name.lower() or 'thumb' in bone_name.lower():
            if bone_name not in rotations:
                rotations[bone_name] = set()
            for kp in fc.keyframe_points:
                rotations[bone_name].add(kp.co[0]) # store frame numbers

print('Max Finger Movements (smplx naming):')
for b in sorted(rotations.keys()):
    print(f'  {b}: found {len(rotations[b])} keyframes')
