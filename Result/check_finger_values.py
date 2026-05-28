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

bpy.context.scene.frame_set(50)
print("Rotations at frame 50:")
for pb in arm.pose.bones:
    if pb.name in ["LeftHandIndex1", "RightHandIndex1"]:
        print(f"{pb.name} matrix_basis:")
        print(pb.matrix_basis.to_euler('XYZ'))
        print(f"{pb.name} matrix (world):")
        print(pb.matrix.to_euler('XYZ'))
