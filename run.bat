@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
echo Running post-processing pipeline...
call venv\Scripts\activate
python animate_postprocess.py "c:\Users\user\Desktop\CG\캡스톤\cap_pipeline\front_4\smplx_merged_hamer_post.pt" --naming smplx --visualize

echo Copying output files to front_4...
copy /y "Result\smplx_merged_hamer_post_postprocessed_smplx_unreal.fbx" "c:\Users\user\Desktop\CG\캡스톤\cap_pipeline\front_4\V2.fbx"
copy /y "Result\smplx_merged_hamer_post_postprocessed_smplx_unreal_anim.fbx" "c:\Users\user\Desktop\CG\캡스톤\cap_pipeline\front_4\front_v2.fbx"

echo Done.
pause
