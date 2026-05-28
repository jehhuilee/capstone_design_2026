import codecs

content = """@echo off
chcp 65001 > nul
echo Running post-processing pipeline...
call venv\\Scripts\\activate
python animate_postprocess.py "c:\\Users\\user\\Desktop\\CG\\캡스톤\\cap_pipeline\\front_4\\smplx_merged_hamer_post.pt" --naming smplx --visualize

echo Renaming output files...
cd /d "c:\\Users\\user\\Desktop\\CG\\캡스톤\\cap_pipeline\\front_4"
if exist "smplx_merged_hamer_post_postprocessed_smplx_unreal.fbx" move /y "smplx_merged_hamer_post_postprocessed_smplx_unreal.fbx" "V2.fbx"
if exist "smplx_merged_hamer_post_postprocessed_smplx_unreal_anim.fbx" move /y "smplx_merged_hamer_post_postprocessed_smplx_unreal_anim.fbx" "front_v2.fbx"

echo Done.
pause
"""

with open('run.bat', 'w', encoding='utf-8-sig') as f:
    f.write(content)
