@echo off
echo Running post-processing pipeline...
call venv\Scripts\activate
python animate_postprocess.py "c:\Users\user\Desktop\CG\Ä¸½ºÅæ\4k_tennis\smplx_merged_hamer.pt" --visualize
pause
