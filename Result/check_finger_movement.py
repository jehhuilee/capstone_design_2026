import bpy
import sys
import os
import math

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
    sys.exit(1)

bpy.context.scene.frame_set(1)
initial_rots = {}
for pb in arm.pose.bones:
    if "Hand" in pb.name or "Index" in pb.name or "Thumb" in pb.name:
        initial_rots[pb.name] = pb.matrix_basis.to_quaternion().copy()

max_diffs = {}

for f in range(2, 100, 2):
    bpy.context.scene.frame_set(f)
    for pb in arm.pose.bones:
        if pb.name in initial_rots:
            q1 = initial_rots[pb.name]
            q2 = pb.matrix_basis.to_quaternion()
            diff = q1.rotation_difference(q2).angle
            if pb.name not in max_diffs or diff > max_diffs[pb.name]:
                max_diffs[pb.name] = diff

print("Max Finger Movements (Degrees):")
for b in sorted(max_diffs.keys()):
    if max_diffs[b] > 0.001:
        print(f"  {b}: {math.degrees(max_diffs[b]):.2f} deg")
