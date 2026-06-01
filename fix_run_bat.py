import codecs

BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

content = """@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
echo Running post-processing pipeline...
call venv\\Scripts\\activate
python animate_postprocess.py "smplx_hmp_injected_full.pt" --naming smplx --visualize

echo Exporting to Unreal FBX (with finger retarget fix)...
"{blender}" -b smplx_template.blend -P export_to_unreal_fbx.py -- "Result\\smplx_hmp_injected_full_postprocessed_smplx.npz" "Result"

echo Copying output files to front_4...
copy /y "Result\\smplx_hmp_injected_full_postprocessed_smplx_unreal.fbx" "c:\\Users\\user\\Desktop\\CG\\캡스톤\\cap_pipeline\\front_4\\V2.fbx"
copy /y "Result\\smplx_hmp_injected_full_postprocessed_smplx_unreal.fbx" "c:\\Users\\user\\Desktop\\CG\\캡스톤\\cap_pipeline\\front_4\\front_v2.fbx"

echo Done.
pause
""".format(blender=BLENDER)

with open('run.bat', 'w', encoding='utf-8-sig') as f:
    f.write(content)

