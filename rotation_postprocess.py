# -*- coding: utf-8 -*-
"""
rotation_postprocess.py
- 손목/팔뚝 과도 트위스트(Over-twisting) + 플래싱(Flashing) 제거용 회전 후처리.
- 기존 One-Euro(Euler) + XPBD(Euler-XYZ 클램핑)를 대체.
- 실측 비교(postprocess_lab)에서 최적으로 검증된 파이프라인:
    (1) 쿼터니언 연속성  → (2) 사비츠키-골레이 스무딩
  → (3) 스윙-트위스트 ROM(unwrap)  → (4) 각속도 클램핑
- AI/딥러닝 미사용. 순수 기하/시계열.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

# body_pose(21관절) 중 트위스트 제한을 둘 관절과 허용 범위(deg).
# 값은 해부학적 pronation/supination + 안전 마진 기준의 기본값(필요 시 조정).
BODY_JOINT_NAMES = [
    'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]
# 트위스트 ROM은 "정확한 본 장축"이 필요하다. 손목/어깨는 PCA 추정축이 굽힘과 섞여
# 클램핑이 그립 방향(라켓 쥔 손)을 꺾어버리므로 제외 — 이들의 과도 트위스트/플래싱은
# 연속성+SG+각속도 클램핑(smooth_rotations)만으로 충분히 잡힌다(원본 대비 ~5° 보존).
# 팔뚝 회내(pronation)가 해부학적으로 일어나는 팔꿈치에만 트위스트 ROM을 적용한다.
BODY_TWIST_LIMITS_DEG = {
    'left_elbow':  (-90, 90),
    'right_elbow': (-90, 90),
}


# ── 쿼터니언 유틸 (wxyz) ─────────────────────────────────────────────
def _rv2q(rv):
    q = R.from_rotvec(rv).as_quat()                       # xyzw
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])

def _q2rv(q):
    return R.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])).as_rotvec()

def _qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])

def _qconj(a):
    return np.array([a[0], -a[1], -a[2], -a[3]])

def _qslerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0: b = -b; d = -d
    if d > 0.9995:
        r = a + t*(b-a); return r/np.linalg.norm(r)
    th0 = np.arccos(np.clip(d, -1, 1)); s0 = np.sin(th0)
    return (np.sin((1-t)*th0)/s0)*a + (np.sin(t*th0)/s0)*b


def _continuous_quats(rv):
    """(1) 부호 연속성: q·q_prev<0 이면 q→-q."""
    q = _rv2q(rv)
    for t in range(1, len(q)):
        if np.dot(q[t], q[t-1]) < 0:
            q[t] = -q[t]
    return q


def _savgol_quat(q, window, poly):
    """(2) 연속 쿼터니언 성분에 SG 다항 피팅 + 재정규화."""
    n = len(q) 
    if n < poly + 2:
        return q
    win = window | 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < poly + 2:
        return q
    qs = np.column_stack([savgol_filter(q[:, k], win, poly) for k in range(4)])
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return qs


def _swing_twist_decompose(qrow, axis):
    w, v = qrow[0], qrow[1:]
    proj = np.dot(v, axis) * axis
    twist = np.array([w, proj[0], proj[1], proj[2]])
    n = np.linalg.norm(twist)
    twist = np.array([1.0, 0, 0, 0]) if n < 1e-9 else twist / n
    phi = 2.0 * np.arctan2(np.dot(v, axis), w)
    phi = (phi + np.pi) % (2*np.pi) - np.pi
    swing = _qmul(qrow, _qconj(twist))
    return swing, phi


def _twist_rom(q, twist_min_deg, twist_max_deg, axis=None, smooth_window=11):
    """(3) 스윙-트위스트 ROM: twist 각을 시간축 unwrap → 스무딩 → soft-clamp → 재조립.
    프레임별 클램프와 달리 ±180° 감김 지점에서도 불연속(스냅/플래싱)이 없다."""
    rv = _q2rv(q)
    if axis is None:                                       # 데이터 주축(=팔뚝 트위스트축 근사)
        c = rv - rv.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
    axis = np.asarray(axis, float); axis /= (np.linalg.norm(axis) + 1e-12)

    swings = np.zeros((len(q), 4)); phis = np.zeros(len(q))
    for t in range(len(q)):
        swings[t], phis[t] = _swing_twist_decompose(q[t], axis)

    phis = np.unwrap(phis)
    if smooth_window and smooth_window >= 5 and len(phis) >= smooth_window:
        phis = savgol_filter(phis, smooth_window | 1, 3)
    phis = np.clip(phis, np.radians(twist_min_deg), np.radians(twist_max_deg))

    out = np.zeros_like(q)
    for t in range(len(q)):
        half = phis[t] / 2.0
        twist_c = np.array([np.cos(half), *(np.sin(half) * axis)])
        out[t] = _qmul(swings[t], twist_c)
    return out


def _angvel_clamp(q, max_deg_per_frame):
    """(4) 각속도 클램핑: 프레임 간 상대회전이 한계 초과 시 이전 프레임으로 slerp.
    잔여 초고속 스파이크(플래싱)에 대한 인과적 안전망."""
    maxr = np.radians(max_deg_per_frame)
    out = q.copy()
    for t in range(1, len(q)):
        d = np.clip(abs(np.dot(out[t-1], q[t])), -1, 1)
        ang = 2.0 * np.arccos(d)
        out[t] = _qslerp(out[t-1], q[t], maxr/ang) if (ang > maxr and ang > 1e-9) else q[t]
    return out


# ── 공개 API ─────────────────────────────────────────────────────────
def smooth_rotations(seq, fps=30, window=9, poly=3, max_deg_per_frame=30.0):
    """(T,N,3) rotvec 시퀀스를 관절별로 연속성→SG 스무딩→각속도 클램핑.
    One-Euro(Euler) 대체. 모든 회전 키에 안전하게 적용 가능."""
    seq = np.asarray(seq, float)
    out = np.empty_like(seq)
    for j in range(seq.shape[1]):
        q = _continuous_quats(seq[:, j])
        q = _savgol_quat(q, window, poly)
        if max_deg_per_frame:
            q = _angvel_clamp(q, max_deg_per_frame)
        out[:, j] = _q2rv(q)
    return out


def apply_twist_rom(body_seq, twist_limits=None, joint_names=None,
                    smooth_twist_window=11):
    """(T,21,3) body_pose에 스윙-트위스트 ROM 적용. XPBD(Euler) 대체.
    twist_limits에 등록된 관절만 트위스트 제한(나머지는 통과)."""
    body_seq = np.asarray(body_seq, float)
    twist_limits = twist_limits or BODY_TWIST_LIMITS_DEG
    joint_names = joint_names or BODY_JOINT_NAMES
    out = body_seq.copy()
    for j, name in enumerate(joint_names):
        if name not in twist_limits:
            continue
        lo, hi = twist_limits[name]
        q = _continuous_quats(body_seq[:, j])
        q = _twist_rom(q, lo, hi, smooth_window=smooth_twist_window)
        out[:, j] = _q2rv(q)
    return out


def stabilize_body_pose(body_seq, fps=30, window=9, poly=3,
                        max_deg_per_frame=30.0, twist_limits=None):
    """body_pose 전용 통합 후처리 — 검증된 최적 파이프라인을 관절별로 순서대로 적용:
        (M1) 연속성 → (M5) SG 스무딩 → (M2u) 스윙-트위스트 ROM → (M3) 각속도 클램핑.
    트위스트 ROM은 twist_limits에 등록된 관절(어깨/팔꿈치/손목)에만 적용."""
    body_seq = np.asarray(body_seq, float)
    twist_limits = twist_limits or BODY_TWIST_LIMITS_DEG
    out = np.empty_like(body_seq)
    for j, name in enumerate(BODY_JOINT_NAMES):
        q = _continuous_quats(body_seq[:, j])            # M1
        q = _savgol_quat(q, window, poly)                # M5
        if name in twist_limits:                         # M2u
            lo, hi = twist_limits[name]
            q = _twist_rom(q, lo, hi, smooth_window=window | 1)
        if max_deg_per_frame:                            # M3
            q = _angvel_clamp(q, max_deg_per_frame)
        out[:, j] = _q2rv(q)
    return out
