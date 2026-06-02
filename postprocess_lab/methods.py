# -*- coding: utf-8 -*-
"""손목/팔뚝 과도 트위스트 + 플래싱을 고치는 5가지 접근법 (순수 numpy/scipy).
모든 함수는 관절 시퀀스 rotvec (T,3) 을 입력받아 보정된 rotvec (T,3) 을 반환.
(AI 모델 미사용 — 기하/시계열 기법만.)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter


# ── 쿼터니언 유틸 (wxyz) ─────────────────────────────────────────────
def rotvec_to_quat(rv):
    q = R.from_rotvec(rv).as_quat()  # xyzw
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])  # wxyz

def quat_to_rotvec(q):
    return R.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])).as_rotvec()

def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])

def qconj(a):
    return np.array([a[0], -a[1], -a[2], -a[3]])

def qslerp(a, b, t):
    d = np.dot(a, b)
    if d < 0: b = -b; d = -d
    if d > 0.9995:
        r = a + t*(b-a); return r/np.linalg.norm(r)
    th0 = np.arccos(np.clip(d, -1, 1)); s0 = np.sin(th0)
    return (np.sin((1-t)*th0)/s0)*a + (np.sin(t*th0)/s0)*b


# ── [접근법 1] 쿼터니언 연속성 / 언랩핑 ──────────────────────────────
def m1_continuity(rv):
    """q·q_prev < 0 이면 q→-q 로 부호 정렬. 표현상의 180°/360° 점프 제거.
    rotvec 자체는 부호-정규라 결과 회전은 동일하지만, 이후 쿼터니언 연산
    (slerp/savgol)이 부호 뒤집힘 없이 동작하도록 만드는 전처리."""
    q = rotvec_to_quat(rv)
    for t in range(1, len(q)):
        if np.dot(q[t], q[t-1]) < 0:
            q[t] = -q[t]
    return quat_to_rotvec(q)

def continuous_quats(rv):
    q = rotvec_to_quat(rv)
    for t in range(1, len(q)):
        if np.dot(q[t], q[t-1]) < 0:
            q[t] = -q[t]
    return q


# ── [접근법 2] 스윙-트위스트 분해 ROM 클램핑 ─────────────────────────
def swing_twist(q, axis):
    """q = swing * twist (twist는 axis 둘레 회전). 반환 (swing, twist, twist각rad)."""
    w, v = q[0], q[1:]
    proj = np.dot(v, axis) * axis
    twist = np.array([w, proj[0], proj[1], proj[2]])
    n = np.linalg.norm(twist)
    twist = np.array([1.0, 0, 0, 0]) if n < 1e-9 else twist / n
    phi = 2.0 * np.arctan2(np.dot(v, axis), w)   # signed twist angle
    # wrap to (-pi, pi]
    phi = (phi + np.pi) % (2*np.pi) - np.pi
    swing = qmul(q, qconj(twist))
    return swing, twist, phi

def m2_swing_twist_rom(rv, axis=None, twist_min_deg=-90.0, twist_max_deg=90.0,
                       smooth_twist_window=0):
    """축(axis, 미지정 시 PCA 주축) 둘레 twist 각을 시간축으로 unwrap → (선택)스무딩
    → [min,max] soft-clamp → 재조립. Euler 분해 없이 짐벌락-free.

    핵심: 프레임별 클램프는 twist가 ±180°를 감을 때 스냅(불연속)을 만든다.
    unwrap 후 연속 신호를 클램프하면 경계에서 평평해질 뿐 불연속이 없어 플래싱이 안 생긴다."""
    q = continuous_quats(rv)
    if axis is None:
        c = rv - rv.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)

    swings = np.zeros((len(q), 4))
    phis = np.zeros(len(q))
    for t in range(len(q)):
        swings[t], _, phis[t] = swing_twist(q[t], axis)

    phis = np.unwrap(phis)                                  # 시간축 연속화
    if smooth_twist_window and smooth_twist_window >= 5:
        win = min(smooth_twist_window | 1, len(phis) - (1 - len(phis) % 2))
        if win >= 5:
            phis = savgol_filter(phis, win, 3)
    phis = np.clip(phis, np.radians(twist_min_deg), np.radians(twist_max_deg))

    out = np.zeros_like(q)
    for t in range(len(q)):
        half = phis[t] / 2.0
        twist_c = np.array([np.cos(half), *(np.sin(half) * axis)])
        out[t] = qmul(swings[t], twist_c)
    return quat_to_rotvec(out)


# ── [접근법 3] 각속도 클램핑 (Slerp 제한) ────────────────────────────
def m3_angvel_clamp(rv, max_deg_per_frame=15.0):
    """프레임 간 상대 회전이 max를 넘으면 이전 프레임으로 slerp 당겨 제한.
    물리적으로 불가능한 초고속 회전(플래싱)을 인과적으로 차단."""
    q = continuous_quats(rv)
    maxr = np.radians(max_deg_per_frame)
    out = q.copy()
    for t in range(1, len(q)):
        d = np.clip(abs(np.dot(out[t-1], q[t])), -1, 1)
        ang = 2.0 * np.arccos(d)
        if ang > maxr and ang > 1e-9:
            out[t] = qslerp(out[t-1], q[t], maxr/ang)
        else:
            out[t] = q[t]
    return quat_to_rotvec(out)


# ── [접근법 4] FABRIK + Joint Limits (위치 기반 IK) ──────────────────
def m4_fabrik(points, target, bone_lengths, iters=10, cone_deg=None):
    """순수 FABRIK (위치 IK). points:(K,3) 관절 위치, target:(3,) 말단 목표.
    주의: FABRIK은 '위치'를 풀며 본의 트위스트(roll) DOF는 결정하지 못한다 →
    팔뚝 트위스트/플래싱 교정에는 부적합(말단 위치 정렬용). 알고리즘 데모만 제공."""
    p = points.astype(float).copy()
    L = np.asarray(bone_lengths, float)
    base = p[0].copy()
    reach = L.sum()
    if np.linalg.norm(target - base) > reach:           # 도달 불가 → 직선 신장
        d = (target - base); d /= (np.linalg.norm(d) + 1e-12)
        for i in range(1, len(p)):
            p[i] = p[i-1] + d * L[i-1]
        return p
    for _ in range(iters):
        # backward
        p[-1] = target
        for i in range(len(p)-2, -1, -1):
            d = p[i] - p[i+1]; d /= (np.linalg.norm(d)+1e-12)
            p[i] = p[i+1] + d * L[i]
        # forward
        p[0] = base
        for i in range(1, len(p)):
            d = p[i] - p[i-1]; d /= (np.linalg.norm(d)+1e-12)
            if cone_deg is not None and i >= 2:          # 굽힘각 콘 제한
                prev = (p[i-1]-p[i-2]); prev /= (np.linalg.norm(prev)+1e-12)
                cos_lim = np.cos(np.radians(cone_deg))
                if np.dot(d, prev) < cos_lim:
                    # d 를 콘 경계로 투영
                    perp = d - np.dot(d, prev)*prev
                    pn = np.linalg.norm(perp)
                    if pn > 1e-9:
                        perp /= pn
                        d = cos_lim*prev + np.sqrt(1-cos_lim**2)*perp
            p[i] = p[i-1] + d * L[i-1]
    return p


# ── [접근법 5] 사비츠키-골레이 (쿼터니언) ────────────────────────────
def m5_savgol(rv, window=11, poly=3):
    """연속화한 쿼터니언 성분에 SG 다항 피팅 후 재정규화. 오프라인 스무딩.
    윈도우 내 다항식으로 노이즈 스파이크를 깎되 피크 형태는 One-Euro보다 보존."""
    q = continuous_quats(rv)
    if window % 2 == 0:
        window += 1
    window = min(window, len(q) - (1 - len(q) % 2))
    if window < poly + 2:
        return quat_to_rotvec(q)
    qs = np.column_stack([savgol_filter(q[:, k], window, poly) for k in range(4)])
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return quat_to_rotvec(qs)
