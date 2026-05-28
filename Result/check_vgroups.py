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

mesh = None
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mesh = obj
        break

if not mesh:
    sys.exit(1)

vgs = {vg.name for vg in mesh.vertex_groups}
print("Vertex Groups in FBX mesh:")
for name in sorted(vgs):
    if "Hand" in name or "hand" in name or "index" in name or "Index" in name:
        print("  " + name)
