import bpy
import sys
import os

# ==============================================================================
# SMPL-X to FBX Exporter for Unreal Engine
# 이 스크립트는 백그라운드 모드(headless)로 실행되어야 합니다.
# 
# 실행 방법:
# blender --background --python export_to_unreal_fbx.py -- [npz_파일_경로] [저장할_fbx_경로]
#
# 주의: Blender 내부에 SMPL-X for Blender Add-on이 설치되어 있어야 합니다.
# ==============================================================================

def main():
    argv = sys.argv
    if "--" not in argv:
        print("Error: 인자가 부족합니다. '--' 뒤에 npz 경로와 fbx 출력 경로를 입력하세요.")
        sys.exit(1)
        
    args = argv[argv.index("--") + 1:]
    if len(args) < 2:
        print("Usage: blender --background --python export_to_unreal_fbx.py -- <input.npz> <output.fbx>")
        sys.exit(1)
        
    animation_file = os.path.abspath(args[0])
    output_fbx = os.path.abspath(args[1])
    
    # SMPL-X 모델 디렉토리 경로 (본인 환경에 맞게 수정 필요)
    smplx_model_dir = os.path.abspath("model") 
    
    print(f"Input NPZ : {animation_file}")
    print(f"Output FBX: {output_fbx}")
    print("--------------------------------------------------")

    # 1. 씬 초기화
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 2. SMPL-X 메쉬 추가 및 애니메이션 로드
    try:
        # 중립(NEUTRAL) 젠더 모델 추가
        # Add-on 버전에 따라 파라미터가 다를 수 있습니다.
        bpy.ops.smplx.add_gender(gender='NEUTRAL', model_dir=smplx_model_dir)
        
        # 추가된 객체가 활성화되어 있는지 확인
        if bpy.context.active_object is None or "SMPLX" not in bpy.context.active_object.name:
            print("Error: SMPL-X 객체가 정상적으로 생성되지 않았습니다.")
            sys.exit(1)
            
        # 애니메이션 파일(.npz) 적용
        bpy.ops.smplx.load_animation(filepath=animation_file)
        print("애니메이션 로드 완료.")
        
    except AttributeError:
        print("Error: SMPL-X Blender Add-on이 설치되어 있지 않거나 활성화되지 않았습니다.")
        print("블렌더 설정(Preferences) -> Add-ons에서 SMPL-X 애드온을 확인해주세요.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during SMPL-X generation: {e}")
        sys.exit(1)

    # 3. 언리얼 엔진 호환성을 위한 FBX 추출
    # 언리얼 기준 Forward: X (또는 -Y), Up: Z 이지만 기본 축으로 맞추고 엔진에서 자동 변환하는 것이 일반적입니다.
    print(f"FBX 추출 시작: {output_fbx}")
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=False,         # 씬 전체 익스포트
        global_scale=1.0,
        bake_anim=True,              # 애니메이션 베이킹 필수
        add_leaf_bones=False,        # 언리얼 스켈레탈 메쉬 뼈대 생성 방지를 위해 False
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        axis_forward='-Z',
        axis_up='Y',
        path_mode='COPY',
        embed_textures=False
    )
    
    print("FBX 추출 성공!")

if __name__ == "__main__":
    main()
