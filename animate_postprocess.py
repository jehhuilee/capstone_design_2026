import torch
import numpy as np
import pyvista as pv
import smplx
from scipy.spatial.transform import Rotation as R

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

    # 로컬 회전을 Euler로 변환 후 unwrap
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
    params_np: dict of numpy arrays, each (T, D)
    반환: (T, V, 3) numpy vertices
    """
    verts_list = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        feed = {}
        for k, v in params_np.items():
            t = torch.tensor(v[start:end]).float()
            feed[k] = t
        # expression이 없으면 betas 배치 크기에 맞게 0으로 채움
        if 'expression' not in feed:
            bs = end - start
            feed['expression'] = torch.zeros(bs, 10, dtype=torch.float32)
        with torch.no_grad():
            out = model(**feed)
        verts_list.append(out.vertices.numpy())
    return np.concatenate(verts_list, axis=0)

# ═══════════════════════════════════════════════════════════════
# 4. 메인 애니메이션
# ═══════════════════════════════════════════════════════════════

def run_animation():
    data_path  = r"c:\Users\user\Desktop\CG\캡스톤\4k_tennis\smplx_merged_hamer.pt"
    model_path = r"model\SMPLX_NEUTRAL.npz"

    # ── 데이터 로드 ──────────────────────────────────────────
    print(f"데이터 로딩: {data_path}")
    data   = torch.load(data_path, map_location='cpu')
    params = data['smpl_params_global']

    T   = params['body_pose'].shape[0]
    fps = 30
    print(f"  총 {T} 프레임, {fps} fps")

    # ── SMPL-X 파라미터를 numpy로 변환 ───────────────────────
    rot_keys = [
        'global_orient', 'body_pose',
        'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose',
    ]

    raw_params      = {}
    smoothed_params = {}

    print("One-Euro 필터 적용 중...")
    for k, v in params.items():
        val = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
        raw_params[k] = val

        if k in rot_keys:
            N     = val.shape[1] // 3
            val_r = val.reshape(T, N, 3)
            smoothed_params[k] = filter_rotations(val_r, fps).reshape(T, -1)
        elif k == 'transl':
            smoothed_params[k] = filter_translations(val, fps)
        else:
            # betas 등 – 그대로 복사
            smoothed_params[k] = val.copy()

    # ── SMPL-X 모델 로드 ─────────────────────────────────────
    print(f"SMPL-X 모델 로딩: {model_path}")
    smplx_model = smplx.create(
        model_path,
        model_type='smplx',
        use_pca=False,   # hand_pose가 45차원 full rotation
        batch_size=1,
    )
    faces = smplx_model.faces   # (F, 3)

    # ── Forward Kinematics (청크 단위) ────────────────────────
    print("Raw 포즈 FK 계산 중...")
    verts_raw = smplx_forward_chunked(smplx_model, raw_params, T, chunk=64)
    print(f"  verts_raw  shape = {verts_raw.shape}")

    print("Smooth 포즈 FK 계산 중...")
    verts_smooth = smplx_forward_chunked(smplx_model, smoothed_params, T, chunk=64)
    print(f"  verts_smooth shape = {verts_smooth.shape}")

    # ── PyVista 애니메이션 ────────────────────────────────────
    print("PyVista 플로터 초기화...")

    # faces → PyVista 형식 [3, i0, i1, i2, 3, i0, ...]
    pv_faces = np.column_stack(
        (np.full(len(faces), 3, dtype=np.int64), faces)
    ).flatten()

    mesh_raw    = pv.PolyData(verts_raw[0].copy(),    pv_faces)
    mesh_smooth = pv.PolyData(verts_smooth[0].copy(),  pv_faces)

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800),
                     title="SMPL-X Post-Processing Comparison")

    # — Left: Before ——————————————————————————————————————
    pl.subplot(0, 0)
    pl.add_text("Before  (Raw)", color='#ff7b72', font_size=14)
    pl.add_mesh(mesh_raw, color='#ffa0a0', smooth_shading=True,
                specular=0.5, specular_power=30)

    # — Right: After ——————————————————————————————————————
    pl.subplot(0, 1)
    pl.add_text("After  (One-Euro Smoothed)", color='#79c0ff', font_size=14)
    pl.add_mesh(mesh_smooth, color='#a0d0ff', smooth_shading=True,
                specular=0.5, specular_power=30)

    # 카메라 동기화
    pl.link_views()

    # 초기 카메라 — 정면에서 보기
    # SMPL-X 좌표: X=좌우, Y=상하, Z=앞뒤
    center = verts_raw[0].mean(axis=0)
    cam_offset = np.array([0, 0, 5.0])   # 조금 더 멀리서 비추기
    pl.camera.focal_point = center
    pl.camera.position    = center + cam_offset
    pl.camera.up          = (0, 1, 0)    # Y-up (위아래 반전 수정)

    print(f"재생 시작 ({T} 프레임, {fps} fps)")
    print("  마우스 드래그로 시점 회전, 스크롤로 확대/축소")
    print("  창을 닫으면 종료됩니다.")

    # ── 수동 애니메이션 루프 ──────────────────────────────────
    import time
    pl.show(interactive_update=True)

    frame = 0
    while not pl.window_size == (0, 0):
        try:
            idx = frame % T
            mesh_raw.points    = verts_raw[idx]
            mesh_smooth.points = verts_smooth[idx]

            # 카메라가 모델 중심을 따라가도록 갱신
            center = verts_raw[idx].mean(axis=0)
            pl.camera.focal_point = center
            pl.camera.position    = center + cam_offset

            pl.update()
            frame += 1
            time.sleep(1.0 / fps)
        except Exception:
            break


if __name__ == "__main__":
    run_animation()