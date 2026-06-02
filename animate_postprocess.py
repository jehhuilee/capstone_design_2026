# -*- coding: utf-8 -*-
"""
animate_postprocess.py
- SMPL-X 데이터(.pt) 로드
- 후처리 파이프라인:
  1) One-Euro 필터 (스무딩)
  2) XPBD 관절 제한 (해부학적 범위 초과 보정)
  3) 스윙 단계 분류
- PyVista로 Before / After 3D 메쉬 애니메이션 비교
"""

import argparse
import os
import torch
import numpy as np
import pyvista as pv
import smplx
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from rotation_postprocess import smooth_rotations, stabilize_body_pose
from swing_classifier import (
    classify_swing_phases,
    PHASE_NAMES, STROKE_NAMES,
)


# ═══════════════════════════════════════════════════════════════
# 1. One-Euro 필터
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# 2. 필터링 유틸
# ═══════════════════════════════════════════════════════════════

def filter_rotations(rot_seq, fps=30):
    """rot_seq: (T, N, 3) rotvecs → smoothed (T, N, 3) rotvecs"""
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


def filter_translations(trans_seq, fps=30):
    """trans_seq: (T, 3) → smoothed (T, 3)"""
    T = trans_seq.shape[0]
    dt  = 1.0 / fps
    oef = OneEuroFilter(0.0, trans_seq[0].copy(), min_cutoff=0.8, beta=0.03)
    out = np.zeros_like(trans_seq)
    for i in range(T):
        out[i] = oef(i * dt, trans_seq[i])
    return out


# ═══════════════════════════════════════════════════════════════
# 3. SMPL-X Forward (배치 청크)
# ═══════════════════════════════════════════════════════════════

def smplx_forward_chunked(model, params_np, T, chunk=64):
    """
    메모리 절약을 위해 chunk 단위로 forward.
    반환: verts (T, V, 3), joints (T, J, 3)
    """
    verts_list  = []
    joints_list = []
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


# ═══════════════════════════════════════════════════════════════
# 4. 메인 애니메이션
# ═══════════════════════════════════════════════════════════════

def run_animation(data_path=None, model_path=None, naming=None, visualize=True):
    if data_path is None:
        data_path = r"c:\Users\user\Desktop\CG\캡스톤\4k_tennis\smplx_merged_hamer.pt"
    if model_path is None:
        model_path = r"model\SMPLX_NEUTRAL.npz"

    # ── 데이터 로드 ──────────────────────────────────────────
    print(f"데이터 로딩: {data_path}")
    data   = torch.load(data_path, map_location='cpu')
    params = data['smpl_params_global']

    T   = params['body_pose'].shape[0]
    fps = 30
    print(f"   총 {T} 프레임, {fps} fps")

    # ── SMPL-X 파라미터 → numpy ──────────────────────────────
    rot_keys = [
        'global_orient', 'body_pose',
        'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose',
    ]

    raw_params      = {}
    smoothed_params = {}

    # Step 1: 회전 후처리 — 연속성 → Savitzky-Golay 스무딩 → 각속도 클램핑.
    #   (기존 One-Euro(Euler) 대체. Euler 기반은 짐벌락/언랩 불안정으로 트위스트 유발.)
    #   body_pose는 추가로 스윙-트위스트 ROM(unwrap)으로 손목/팔뚝 과도 트위스트를 제한
    #   → 기존 XPBD(Euler-XYZ 클램핑) 대체. 실측 비교(postprocess_lab)에서 최적 검증.
    print("Step 1: 회전 후처리(연속성/SG/스윙-트위스트 ROM/각속도) 적용 중...")
    for k, v in params.items():
        val = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
        raw_params[k] = val
        if k == 'body_pose':
            val_r = val.reshape(T, 21, 3)
            smoothed_params[k] = stabilize_body_pose(val_r, fps=fps).reshape(T, -1)
        elif k in rot_keys:
            val_r = val.reshape(T, -1, 3)
            smoothed_params[k] = smooth_rotations(val_r, fps=fps).reshape(T, -1)
        elif k == 'transl':
            smoothed_params[k] = filter_translations(val, fps)
        else:
            smoothed_params[k] = val.copy()

    # ── SMPL-X 모델 로드 ─────────────────────────────────────
    print(f"SMPL-X 모델 로딩: {model_path}")
    smplx_model = smplx.create(
        model_path, model_type='smplx',
        use_pca=False, batch_size=1,
    )
    faces = smplx_model.faces

    # ── FK 계산 ──────────────────────────────────────────────
    print("Raw FK 계산 중...")
    verts_raw, joints_raw = smplx_forward_chunked(smplx_model, raw_params, T, chunk=64)
    print(f"   verts_raw = {verts_raw.shape}")

    print("Smoothed + XPBD FK 계산 중...")
    verts_smooth, joints_smooth = smplx_forward_chunked(smplx_model, smoothed_params, T, chunk=64)
    print(f"   verts_smooth = {verts_smooth.shape}")

    # Step 3: 스윙 분류 + 라켓 방향
    print("Step 3: 스윙 단계 분류 중...")
    phases, stroke_types = classify_swing_phases(joints_smooth, fps=fps, is_right_handed=True)

    # 단계 분포 출력
    from collections import Counter
    phase_counts = Counter(phases)
    for p, count in sorted(phase_counts.items()):
        print(f"   {PHASE_NAMES[p]:>16s}: {count:>5d} 프레임 ({100*count/T:.1f}%)")

    # ── 결과 저장 (.npz) ──────────────────────────────────────
    input_stem = Path(data_path).stem
    if naming:
        out_name = f"{input_stem}_postprocessed_{naming}"
    else:
        out_name = f"{input_stem}_postprocessed"

    result_dir = Path("Result")
    result_dir.mkdir(parents=True, exist_ok=True)
    out_npz = result_dir / f"{out_name}.npz"

    np.savez(
        str(out_npz),
        global_orient    = smoothed_params['global_orient'],
        body_pose        = smoothed_params['body_pose'],
        left_hand_pose   = smoothed_params.get('left_hand_pose', np.zeros((T, 45), dtype=np.float32)),
        right_hand_pose  = smoothed_params.get('right_hand_pose', np.zeros((T, 45), dtype=np.float32)),
        transl           = smoothed_params['transl'],
        betas            = smoothed_params.get('betas', np.zeros((T, 10), dtype=np.float32)),
        gender           = 'neutral',
    )
    print(f"후처리 결과 저장 완료: {out_npz}")

    # ── PyVista 시각화 ────────────────────────────────────────
    if not visualize:
        print("시각화 건너뜀 (--visualize 미사용).")
        return

    print("PyVista 플로터 초기화...")

    pv_faces = np.column_stack(
        (np.full(len(faces), 3, dtype=np.int64), faces)
    ).flatten()

    mesh_raw    = pv.PolyData(verts_raw[0].copy(), pv_faces)
    mesh_smooth = pv.PolyData(verts_smooth[0].copy(), pv_faces)

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800),
                     title="SMPL-X Post-Processing: Before vs After")

    # — Left: Before ——————————————————————————————————————
    pl.subplot(0, 0)
    pl.add_text("Before  (Raw)", color='#ff7b72', font_size=14)
    pl.add_mesh(mesh_raw, color='#ffa0a0', smooth_shading=True,
                specular=0.5, specular_power=30)

    # -- Right: After ------------------------------------------
    pl.subplot(0, 1)
    # 단계 텍스트 (동적으로 업데이트)
    phase_text_actor = pl.add_text(
        "After (XPBD + One-Euro) | READY | FOREHAND",
        color='#79c0ff', font_size=12,
    )
    pl.add_mesh(mesh_smooth, color='#a0d0ff', smooth_shading=True,
                specular=0.5, specular_power=30)

    # 카메라 링크 + 초기 위치
    pl.link_views()
    center = verts_raw[0].mean(axis=0)
    cam_offset = np.array([0, 0, 5.0])
    pl.camera.focal_point = center
    pl.camera.position    = center + cam_offset
    pl.camera.up          = (0, 1, 0)

    print(f"재생 시작 ({T} 프레임, {fps} fps)")
    print("   마우스 드래그로 시점 회전, 스크롤로 확대/축소")
    print("   창을 닫으면 종료됩니다.")

    # ── 애니메이션 루프 ──────────────────────────────────────
    pl.show(interactive_update=True)

    frame = 0
    try:
        while True:
            try:
                # 창이 닫혔는지 확인
                if not hasattr(pl, 'ren_win') or pl.ren_win is None:
                    break
                try:
                    if not pl.ren_win.GetNeverRendered() == 0 and pl.ren_win.GetSize()[0] == 0:
                        break
                except Exception:
                    break

                idx = frame % T
                mesh_raw.points    = verts_raw[idx]
                mesh_smooth.points = verts_smooth[idx]

                # 단계 텍스트 업데이트
                phase_name  = PHASE_NAMES[phases[idx]]
                stroke_name = STROKE_NAMES[stroke_types[idx]]
                phase_text_actor.SetText(
                    0,
                    f"After (XPBD + One-Euro) | {phase_name} | {stroke_name} | Frame {idx+1}/{T}"
                )

                # 카메라 추적
                center = verts_raw[idx].mean(axis=0)
                pl.camera.focal_point = center
                pl.camera.position    = center + cam_offset

                pl.update()
                frame += 1
                time.sleep(1.0 / fps)
            except Exception:
                break
    finally:
        try:
            pl.close()
        except Exception:
            pass
        try:
            pv.close_all()
        except Exception:
            pass

    print("시각화 종료.")
    import os
    os._exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="SMPL-X 후처리 + 시각화")
    parser.add_argument("input", nargs="?", default=None,
                        help="입력 .pt 파일 경로")
    parser.add_argument("--naming", type=str, default=None,
                        help="출력 파일 이름에 추가할 접미사 (예: smplx)")
    parser.add_argument("--visualize", action="store_true",
                        help="PyVista 시각화 활성화")
    parser.add_argument("--model-path", type=str, default=None,
                        help="SMPL-X 모델 경로 (기본: model/SMPLX_NEUTRAL.npz)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_animation(
        data_path=args.input,
        model_path=args.model_path,
        naming=args.naming,
        visualize=args.visualize,
    )