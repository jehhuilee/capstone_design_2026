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

vg_weights = {}
for vg in mesh.vertex_groups:
    vg_weights[vg.name] = 0

for v in mesh.data.vertices:
    for g in v.groups:
        if g.weight > 0.01:
            vg_name = mesh.vertex_groups[g.group].name
            vg_weights[vg_name] += 1

print("Vertex Groups with >0 vertices:")
for name in sorted(vg_weights.keys()):
    if "Index" in name or "Thumb" in name:
        print(f"  {name}: {vg_weights[name]} vertices")
