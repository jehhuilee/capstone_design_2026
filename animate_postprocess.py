"""
animate_postprocess.py
======================
Post-processing pipeline for SMPL-X motion data.

Functions:
    postprocess   - One-Euro filter + XPBD + Grip IK  (no GUI, server-safe)
    export_fbx    - Blender subprocess to convert NPZ → FBX
    visualize     - PyVista interactive viewer  (local only, optional)
    run_pipeline  - postprocess + export_fbx in one call

CLI usage:
    python animate_postprocess.py <input.pt> [options]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

PROJ_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# One-Euro Filter
# ---------------------------------------------------------------------------

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = x0.copy()
        self.dx_prev    = np.zeros_like(x0)
        self.t_prev     = t0

    def _sf(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x
        dx      = (x - self.x_prev) / t_e
        ad      = self._sf(t_e, self.d_cutoff)
        dx_hat  = ad * dx + (1 - ad) * self.dx_prev
        cutoff  = self.min_cutoff + self.beta * np.abs(dx_hat)
        a       = self._sf(t_e, cutoff)
        x_hat   = a * x + (1 - a) * self.x_prev
        self.x_prev  = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev  = t
        return x_hat


def filter_rotations(rot_seq: np.ndarray, fps: int = 30) -> np.ndarray:
    T, N, _ = rot_seq.shape
    dt = 1.0 / fps
    euler   = R.from_rotvec(rot_seq.reshape(-1, 3)).as_euler('XYZ').reshape(T, N, 3)
    euler_u = np.unwrap(euler, axis=0)
    oef = OneEuroFilter(0.0, euler_u[0].flatten(), min_cutoff=0.8, beta=0.03)
    out = np.zeros_like(rot_seq)
    for i in range(T):
        smoothed = oef(i * dt, euler_u[i].flatten()).reshape(N, 3)
        out[i]   = R.from_euler('XYZ', smoothed).as_rotvec().reshape(N, 3)
    return out


def filter_translations(trans_seq: np.ndarray, fps: int = 30) -> np.ndarray:
    T = trans_seq.shape[0]
    dt  = 1.0 / fps
    oef = OneEuroFilter(0.0, trans_seq[0].copy(), min_cutoff=0.8, beta=0.03)
    out = np.zeros_like(trans_seq)
    for i in range(T):
        out[i] = oef(i * dt, trans_seq[i])
    return out


# ---------------------------------------------------------------------------
# SMPL-X helpers
# ---------------------------------------------------------------------------

def smplx_forward_chunked(model, params_np: dict, T: int, chunk: int = 64):
    verts_list: list[np.ndarray]  = []
    joints_list: list[np.ndarray] = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        feed = {}
        for k, v in params_np.items():
            feed[k] = torch.tensor(v[start:end]).float()
        if 'expression' not in feed:
            bs = end - start
            feed['expression'] = torch.zeros(bs, 10, dtype=torch.float32)
        with torch.no_grad():
            out = model(**feed)
        verts_list.append(out.vertices.numpy())
        joints_list.append(out.joints.numpy())
    return np.concatenate(verts_list, 0), np.concatenate(joints_list, 0)


def fix_two_handed_grip_ik(model, params_np: dict, stroke_types, phases) -> dict:
    from swing_classifier import StrokeType, SwingPhase
    T = params_np['body_pose'].shape[0]

    target_frames = []
    for t in range(T):
        if stroke_types[t] == StrokeType.BACKHAND and phases[t] not in [SwingPhase.RECOVERY, SwingPhase.READY]:
            target_frames.append(t)

    if not target_frames:
        return params_np

    print(f"  Grip IK correction: {len(target_frames)} target frames")

    model.eval()

    fixed_params = {}
    for k, v in params_np.items():
        fixed_params[k] = torch.tensor(v).float()
    if 'expression' not in fixed_params:
        fixed_params['expression'] = torch.zeros(T, 10, dtype=torch.float32)

    body_pose = fixed_params['body_pose'].clone()

    ls_idx, le_idx, lw_idx = 16 * 3, 18 * 3, 20 * 3

    left_arm_params = torch.zeros(len(target_frames), 9, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for i, t in enumerate(target_frames):
            left_arm_params[i, 0:3] = body_pose[t, ls_idx:ls_idx + 3]
            left_arm_params[i, 3:6] = body_pose[t, le_idx:le_idx + 3]
            left_arm_params[i, 6:9] = body_pose[t, lw_idx:lw_idx + 3]

    optimizer = torch.optim.Adam([left_arm_params], lr=0.05)

    for step in range(30):
        optimizer.zero_grad()

        curr_body_pose = body_pose[target_frames].clone()
        curr_body_pose[:, ls_idx:ls_idx + 3] = left_arm_params[:, 0:3]
        curr_body_pose[:, le_idx:le_idx + 3] = left_arm_params[:, 3:6]
        curr_body_pose[:, lw_idx:lw_idx + 3] = left_arm_params[:, 6:9]

        feed = {k: v[target_frames] for k, v in fixed_params.items()}
        feed['body_pose'] = curr_body_pose

        out = model(**feed)
        joints = out.joints

        left_wrist = joints[:, 20]
        right_wrist = joints[:, 21]

        target_dist = 0.10
        dists = torch.norm(left_wrist - right_wrist, dim=1)
        loss = torch.mean((dists - target_dist) ** 2)

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for i, t in enumerate(target_frames):
            body_pose[t, ls_idx:ls_idx + 3] = left_arm_params[i, 0:3]
            body_pose[t, le_idx:le_idx + 3] = left_arm_params[i, 3:6]
            body_pose[t, lw_idx:lw_idx + 3] = left_arm_params[i, 6:9]

    params_np['body_pose'] = body_pose.numpy()
    return params_np


# ---------------------------------------------------------------------------
# 1) postprocess  —  서버에서 안전하게 호출 가능 (GUI 없음)
# ---------------------------------------------------------------------------

def postprocess(
    input_pt: str | Path,
    output_npz: str | Path | None = None,
    model_path: str | Path = "model/SMPLX_NEUTRAL.npz",
    fps: int = 30,
) -> Path:
    """
    GVHMR+HaMeR 결과(.pt)를 받아 후처리 후 NPZ로 저장한다.

    Parameters
    ----------
    input_pt   : smplx_merged_hamer.pt 경로
    output_npz : 출력 NPZ 경로 (None이면 input 옆에 _postprocessed.npz)
    model_path : SMPL-X 모델 경로 (기본값: 프로젝트 내 model/SMPLX_NEUTRAL.npz)
    fps        : 프레임 레이트

    Returns
    -------
    Path : 저장된 NPZ 경로
    """
    import smplx as smplx_lib
    from xpbd_constraints import apply_xpbd_constraints
    from swing_classifier import classify_swing_phases

    input_pt = Path(input_pt)
    if output_npz is None:
        output_npz = input_pt.with_name(input_pt.stem + "_postprocessed.npz")
    output_npz = Path(output_npz)

    # model_path를 절대경로로 해석
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJ_ROOT / model_path

    print(f"[postprocess] Loading: {input_pt}")
    data = torch.load(str(input_pt), map_location='cpu')
    params = data['smpl_params_global']

    T = params['body_pose'].shape[0]
    print(f"[postprocess] {T} frames, {fps} fps")

    rot_keys = [
        'global_orient', 'body_pose',
        'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose',
    ]

    raw_params: dict[str, np.ndarray] = {}
    smoothed_params: dict[str, np.ndarray] = {}

    print("[postprocess] Applying One-Euro filter...")
    for k, v in params.items():
        val = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
        raw_params[k] = val
        if k in rot_keys:
            N = val.shape[1] // 3
            val_r = val.reshape(T, N, 3)
            smoothed_params[k] = filter_rotations(val_r, fps).reshape(T, -1)
        elif k == 'transl':
            smoothed_params[k] = filter_translations(val, fps)
        else:
            smoothed_params[k] = val.copy()

    # print(f"[postprocess] Loading SMPL-X model: {model_path}")
    # smplx_model = smplx_lib.create(
    #     str(model_path), model_type='smplx',
    #     use_pca=False, batch_size=1,
    # )

    # print("[postprocess] Computing raw FK for swing analysis...")
    # _, joints_raw = smplx_forward_chunked(smplx_model, raw_params, T, chunk=64)

    # print("[postprocess] Analyzing swing phases...")
    # phases, stroke_types = classify_swing_phases(joints_raw, fps=fps, is_right_handed=True)

    # smoothed_params = fix_two_handed_grip_ik(smplx_model, smoothed_params, stroke_types, phases)

    # print("[postprocess] Applying XPBD constraints...")
    # constrained_body_pose, total_violations = apply_xpbd_constraints(
    #     smoothed_params['body_pose'],
    #     fps=fps,
    #     compliance=0.001,
    #     num_iterations=8,
    #     num_substeps=4,
    # )
    # smoothed_params['body_pose'] = constrained_body_pose

    # 결과 저장
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"[postprocess] Saving: {output_npz}")
    np.savez(str(output_npz), **smoothed_params)
    print(f"[postprocess] Done. {T} frames saved.")
    return output_npz


# ---------------------------------------------------------------------------
# 2) export_fbx  —  Blender 서브프로세스 호출
# ---------------------------------------------------------------------------

def _find_blender(blender_path: str | Path | None = None) -> Path:
    """Blender 실행파일 경로를 찾는다: 인자 → 환경변수 → 기본 경로."""
    if blender_path is not None:
        p = Path(blender_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Blender not found: {p}")

    env_path = os.environ.get("BLENDER_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"BLENDER_PATH set but not found: {p}")

    # Windows 기본 설치 경로 탐색
    if sys.platform == "win32":
        base = Path(r"C:\Program Files\Blender Foundation")
        if base.exists():
            candidates = sorted(base.iterdir(), reverse=True)
            for candidate in candidates:
                exe = candidate / "blender.exe"
                if exe.exists():
                    return exe

    raise FileNotFoundError(
        "Blender를 찾을 수 없습니다. 다음 중 하나를 사용하세요:\n"
        "  --blender-path 인자\n"
        "  BLENDER_PATH 환경변수\n"
        "  C:\\Program Files\\Blender Foundation\\<version>\\blender.exe 설치"
    )


def export_fbx(
    input_npz: str | Path,
    output_dir: str | Path | None = None,
    blender_path: str | Path | None = None,
    engine: str = "unreal",
    naming: str = "mixamo",
    fps: float = 30.0,
) -> Path:
    """
    Blender를 호출하여 후처리된 NPZ를 Humanoid FBX로 변환한다.

    Parameters
    ----------
    input_npz    : 후처리된 .npz 파일 경로
    output_dir   : FBX 출력 디렉토리 (None이면 프로젝트 Result 폴더)
    blender_path : Blender 실행파일 경로 (None이면 자동 탐색)
    engine       : 'unreal' 또는 'unity'
    naming       : 'mixamo' 또는 'unity'
    fps          : 프레임 레이트

    Returns
    -------
    Path : 출력 디렉토리 경로
    """
    input_npz = Path(input_npz).resolve()
    if not input_npz.exists():
        raise FileNotFoundError(f"NPZ not found: {input_npz}")

    blender_exe = _find_blender(blender_path)
    blend_file = PROJ_ROOT / "smplx_template.blend"
    export_script = PROJ_ROOT / "export_smplx_to_humanoid_fbx.py"

    if not blend_file.exists():
        raise FileNotFoundError(f"Blend template not found: {blend_file}")
    if not export_script.exists():
        raise FileNotFoundError(f"Export script not found: {export_script}")

    if output_dir is None:
        output_dir = PROJ_ROOT / "Result"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(blender_exe),
        "-b", str(blend_file),
        "-P", str(export_script),
        "--",
        str(input_npz),
        str(output_dir),
        "--engine", engine,
        "--naming", naming,
        "--fps", str(fps),
    ]

    print(f"[export_fbx] Blender: {blender_exe}")
    print(f"[export_fbx] Input:   {input_npz}")
    print(f"[export_fbx] Output:  {output_dir}")
    print(f"[export_fbx] Running Blender...")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

    # Blender 출력에서 마지막 줄들만 표시
    if result.stdout:
        lines = result.stdout.strip().splitlines()
        for line in lines[-20:]:
            print(f"  [blender] {line}")

    if result.returncode != 0:
        err_msg = result.stderr[-500:] if result.stderr else "unknown error"
        print(f"[export_fbx] ERROR: Blender exited with code {result.returncode}")
        print(f"  {err_msg}")
        raise RuntimeError(f"Blender FBX export failed (exit code {result.returncode})")

    print(f"[export_fbx] FBX conversion complete: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# 3) visualize  —  로컬 전용 PyVista 시각화 (서버에서는 호출하지 않음)
# ---------------------------------------------------------------------------

def visualize(
    input_pt: str | Path,
    model_path: str | Path = "model/SMPLX_NEUTRAL.npz",
    fps: int = 30,
) -> None:
    """
    PyVista로 Before/After 시각화 + 라켓 메시를 표시한다.
    창을 닫으면 자동으로 리턴된다 (무한 루프 없음).
    """
    import pyvista as pv
    import smplx as smplx_lib
    from xpbd_constraints import apply_xpbd_constraints
    from swing_classifier import classify_swing_phases, PHASE_NAMES

    # VTK 경고 억제
    try:
        import vtk
        vtk.vtkObject.GlobalWarningDisplayOff()
        if hasattr(vtk, 'vtkLogger'):
            vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    except Exception:
        pass

    input_pt = Path(input_pt)
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJ_ROOT / model_path

    print(f"[visualize] Loading: {input_pt}")
    data = torch.load(str(input_pt), map_location='cpu')
    params = data['smpl_params_global']

    T = params['body_pose'].shape[0]
    print(f"[visualize] {T} frames, {fps} fps")

    rot_keys = [
        'global_orient', 'body_pose',
        'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose',
    ]

    raw_params: dict[str, np.ndarray] = {}
    smoothed_params: dict[str, np.ndarray] = {}

    for k, v in params.items():
        val = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
        raw_params[k] = val
        if k in rot_keys:
            N = val.shape[1] // 3
            val_r = val.reshape(T, N, 3)
            smoothed_params[k] = filter_rotations(val_r, fps).reshape(T, -1)
        elif k == 'transl':
            smoothed_params[k] = filter_translations(val, fps)
        else:
            smoothed_params[k] = val.copy()

    smplx_model = smplx_lib.create(
        str(model_path), model_type='smplx',
        use_pca=False, batch_size=1,
    )
    faces = smplx_model.faces

    print("[visualize] Computing raw FK...")
    verts_raw, joints_raw = smplx_forward_chunked(smplx_model, raw_params, T, chunk=64)

    print("[visualize] Analyzing swing phases...")
    phases, stroke_types = classify_swing_phases(joints_raw, fps=fps, is_right_handed=True)

    smoothed_params = fix_two_handed_grip_ik(smplx_model, smoothed_params, stroke_types, phases)

    constrained_body_pose, _ = apply_xpbd_constraints(
        smoothed_params['body_pose'],
        fps=fps, compliance=0.001, num_iterations=8, num_substeps=4,
    )
    smoothed_params['body_pose'] = constrained_body_pose

    print("[visualize] Computing smoothed FK...")
    verts_smooth, joints_smooth = smplx_forward_chunked(smplx_model, smoothed_params, T, chunk=64)

    # 스윙 페이즈 통계
    from collections import Counter
    phase_counts = Counter(phases)
    for p, count in sorted(phase_counts.items()):
        print(f"  {PHASE_NAMES[p]:>16s}: {count:>5d} frames ({100 * count / T:.1f}%)")

    # ── Y-up → Z-up 좌표 변환 (언리얼 기준) ──
    # SMPL-X: X=right, Y=up, Z=forward
    # Unreal:  X=forward, Y=right, Z=up
    # 90° rotation around X-axis: (x, y, z) → (x, -z, y)
    rot_to_zup = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float32)
    verts_raw    = np.einsum('ij,fnj->fni', rot_to_zup, verts_raw)
    verts_smooth = np.einsum('ij,fnj->fni', rot_to_zup, verts_smooth)

    # PyVista 설정
    print("[visualize] Initializing PyVista plotter...")
    pv_faces = np.column_stack(
        (np.full(len(faces), 3, dtype=np.int64), faces)
    ).flatten()

    mesh_raw    = pv.PolyData(verts_raw[0].copy(), pv_faces)
    mesh_smooth = pv.PolyData(verts_smooth[0].copy(), pv_faces)

    pl = pv.Plotter(
        shape=(1, 2), window_size=(1600, 800),
        title="SMPL-X Post-Processing: Before vs After",
    )

    pl.subplot(0, 0)
    pl.add_text("Before", color='#ff7b72', font_size=14)
    pl.add_mesh(mesh_raw, color='#ffa0a0', smooth_shading=True,
                specular=0.5, specular_power=30)

    pl.subplot(0, 1)
    pl.add_text("After", color='#79c0ff', font_size=14)
    pl.add_mesh(mesh_smooth, color='#a0d0ff', smooth_shading=True,
                specular=0.5, specular_power=30)

    pl.link_views()
    center = verts_raw[0].mean(axis=0)
    cam_offset = np.array([0, -5.0, 0])   # Z-up이므로 -Y 방향에서 바라봄
    pl.camera.focal_point = center
    pl.camera.position    = center + cam_offset
    pl.camera.up          = (0, 0, 1)      # Z-up (언리얼 기준)

    # ── 애니메이션 상태 ──
    anim_state = {"idx": 0, "paused": False}

    def on_timer(obj, _event):
        """VTK TimerEvent 콜백 — 매 프레임 메시 갱신."""
        if anim_state["paused"]:
            return
        idx = anim_state["idx"] % T
        mesh_raw.points    = verts_raw[idx]
        mesh_smooth.points = verts_smooth[idx]

        new_center = verts_raw[idx].mean(axis=0)
        pl.camera.focal_point = new_center
        pl.camera.position    = new_center + cam_offset

        anim_state["idx"] += 1
        obj.GetRenderWindow().Render()

    def on_keypress(obj, _event):
        """스페이스바로 일시정지/재생 토글."""
        key = obj.GetKeySym()
        if key == "space":
            anim_state["paused"] = not anim_state["paused"]
            status = "PAUSED" if anim_state["paused"] else "PLAYING"
            print(f"[visualize] {status}")

    print(f"[visualize] Playing animation: {T} frames at {fps} fps")
    print("[visualize] Press SPACE to pause/resume, close window to exit.")

    # show()를 interactive_update=False (블로킹)으로 사용하되,
    # show 전에 interactor를 초기화하고 VTK 네이티브 타이머를 직접 등록
    pl.show(interactive_update=True)

    iren = pl.iren.interactor
    interval_ms = max(1, int(1000 / fps))

    # VTK 네이티브 옵저버 등록
    iren.AddObserver("TimerEvent", on_timer)
    iren.AddObserver("KeyPressEvent", on_keypress)
    timer_id = iren.CreateRepeatingTimer(interval_ms)

    # 블로킹 이벤트 루프 시작 — 창을 닫으면 자동으로 여기서 리턴
    iren.Start()

    # 정리
    iren.DestroyTimer(timer_id)
    print("[visualize] Window closed.")


# ---------------------------------------------------------------------------
# 4) run_pipeline  —  postprocess + export_fbx 통합 호출
# ---------------------------------------------------------------------------

def run_pipeline(
    input_pt: str | Path,
    output_dir: str | Path | None = None,
    model_path: str | Path = "model/SMPLX_NEUTRAL.npz",
    blender_path: str | Path | None = None,
    engine: str = "unreal",
    naming: str = "mixamo",
    fps: int = 30,
) -> dict[str, Path]:
    """
    후처리 → FBX 변환을 한 번에 실행한다.

    Returns
    -------
    dict with keys: 'npz', 'fbx_dir'
    """
    input_pt = Path(input_pt)
    if output_dir is None:
        output_dir = PROJ_ROOT / "Result"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_npz = output_dir / (input_pt.stem + "_postprocessed.npz")

    print("=" * 60)
    print("SMPL-X Post-Processing Pipeline")
    print("=" * 60)
    print(f"  Input  : {input_pt}")
    print(f"  Output : {output_dir}")
    print(f"  Engine : {engine}")
    print("=" * 60)

    # Step 1: 후처리
    npz_path = postprocess(
        input_pt=input_pt,
        output_npz=output_npz,
        model_path=model_path,
        fps=fps,
    )

    # Step 2: FBX 변환
    fbx_dir = export_fbx(
        input_npz=npz_path,
        output_dir=output_dir,
        blender_path=blender_path,
        engine=engine,
        naming=naming,
        fps=float(fps),
    )

    print("=" * 60)
    print("Pipeline complete!")
    print(f"  NPZ: {npz_path}")
    print(f"  FBX: {fbx_dir}")
    print("=" * 60)

    return {"npz": npz_path, "fbx_dir": fbx_dir}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SMPL-X 모션 후처리 + FBX 변환 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 후처리 + FBX (기본)
  python animate_postprocess.py smplx_merged_hamer.pt

  # 출력 디렉토리 지정
  python animate_postprocess.py input.pt --output-dir ./result

  # 후처리만 (FBX 변환 생략)
  python animate_postprocess.py input.pt --skip-fbx

  # 시각화 포함 (로컬 전용)
  python animate_postprocess.py input.pt --visualize

  # Blender 경로 직접 지정
  python animate_postprocess.py input.pt --blender-path "C:\\Blender\\blender.exe"
""",
    )
    parser.add_argument(
        "input_pt",
        help="GVHMR+HaMeR merged .pt file path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: input file 옆)",
    )
    parser.add_argument(
        "--model-path",
        default="model/SMPLX_NEUTRAL.npz",
        help="SMPL-X model path (default: model/SMPLX_NEUTRAL.npz)",
    )
    parser.add_argument(
        "--blender-path",
        default=None,
        help="Blender executable path (or set BLENDER_PATH env var)",
    )
    parser.add_argument(
        "--engine",
        choices=["unreal", "unity"],
        default="unreal",
        help="Target engine (default: unreal)",
    )
    parser.add_argument(
        "--naming",
        choices=["mixamo", "unity", "smplx"],
        default="mixamo",
        help="Bone naming convention (default: mixamo)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate (default: 30)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open PyVista visualization window (local only)",
    )
    parser.add_argument(
        "--skip-fbx",
        action="store_true",
        help="Skip Blender FBX conversion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_pt = Path(args.input_pt).resolve()
    if not input_pt.exists():
        print(f"Error: Input file not found: {input_pt}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (PROJ_ROOT / "Result")
    output_npz = output_dir / (input_pt.stem + "_postprocessed.npz")

    # Step 1: 후처리
    npz_path = postprocess(
        input_pt=input_pt,
        output_npz=output_npz,
        model_path=args.model_path,
        fps=args.fps,
    )

    # Step 2: FBX 변환 (선택)
    if not args.skip_fbx:
        export_fbx(
            input_npz=npz_path,
            output_dir=output_dir,
            blender_path=args.blender_path,
            engine=args.engine,
            naming=args.naming,
            fps=float(args.fps),
        )

    # Step 3: 시각화 (선택, 로컬 전용)
    if args.visualize:
        visualize(
            input_pt=input_pt,
            model_path=args.model_path,
            fps=args.fps,
        )

    print("\nAll done!")


if __name__ == "__main__":
    main()