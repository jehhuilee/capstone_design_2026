# -*- coding: utf-8 -*-
"""
swing_classifier.py
- 테니스 스윙 6단계 분류 (READY, BACKSWING, ACCELERATION, CONTACT, FOLLOW_THROUGH, RECOVERY)
- 스트로크 유형 판별 (FOREHAND, BACKHAND, SERVE)
- 라켓 방향 선택 및 보간
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ═══════════════════════════════════════════════════════════════
# 1. 스윙 단계 정의
# ═══════════════════════════════════════════════════════════════

class SwingPhase:
    READY          = 0
    BACKSWING      = 1
    ACCELERATION   = 2
    CONTACT        = 3
    FOLLOW_THROUGH = 4
    RECOVERY       = 5

PHASE_NAMES = [
    'READY', 'BACKSWING', 'ACCELERATION',
    'CONTACT', 'FOLLOW_THROUGH', 'RECOVERY'
]

class StrokeType:
    FOREHAND = 0
    BACKHAND = 1
    SERVE    = 2

STROKE_NAMES = ['FOREHAND', 'BACKHAND', 'SERVE']

# ═══════════════════════════════════════════════════════════════
# 2. 라켓 방향 정의 (단계별 Euler angles, degrees)
#    손목 로컬 좌표계 기준 (pitch, yaw, roll)
# ═══════════════════════════════════════════════════════════════

# 포핸드 라켓 자세
RACKET_EULER_FOREHAND = {
    SwingPhase.READY:          np.array([  0,    0,    0]),
    SwingPhase.BACKSWING:      np.array([ 20, -100,   10]),
    SwingPhase.ACCELERATION:   np.array([  5,  -40,  -20]),
    SwingPhase.CONTACT:        np.array([ -8,    0,    0]),
    SwingPhase.FOLLOW_THROUGH: np.array([-30,   45,  -35]),
    SwingPhase.RECOVERY:       np.array([  0,    0,    0]),
}

# 백핸드 라켓 자세
RACKET_EULER_BACKHAND = {
    SwingPhase.READY:          np.array([  0,    0,    0]),
    SwingPhase.BACKSWING:      np.array([ 20,  100,  -10]),
    SwingPhase.ACCELERATION:   np.array([  5,   40,   20]),
    SwingPhase.CONTACT:        np.array([ -8,    0,    0]),
    SwingPhase.FOLLOW_THROUGH: np.array([-30,  -45,   35]),
    SwingPhase.RECOVERY:       np.array([  0,    0,    0]),
}

# 서브 라켓 자세
RACKET_EULER_SERVE = {
    SwingPhase.READY:          np.array([  0,    0,    0]),
    SwingPhase.BACKSWING:      np.array([ 30,  -60,   15]),  # 트로피 포지션
    SwingPhase.ACCELERATION:   np.array([ 10,  -30,  -25]),
    SwingPhase.CONTACT:        np.array([  0,    0,   -5]),
    SwingPhase.FOLLOW_THROUGH: np.array([-35,   50,  -40]),
    SwingPhase.RECOVERY:       np.array([  0,    0,    0]),
}

def _euler_to_quat(euler_deg):
    """Euler degrees → quaternion"""
    return R.from_euler('XYZ', np.radians(euler_deg))

def _build_racket_quats(euler_dict):
    """단계별 Euler → quaternion 딕셔너리"""
    return {phase: _euler_to_quat(e) for phase, e in euler_dict.items()}

RACKET_QUATS = {
    StrokeType.FOREHAND: _build_racket_quats(RACKET_EULER_FOREHAND),
    StrokeType.BACKHAND: _build_racket_quats(RACKET_EULER_BACKHAND),
    StrokeType.SERVE:    _build_racket_quats(RACKET_EULER_SERVE),
}


# ═══════════════════════════════════════════════════════════════
# 3. 스트로크 유형 판별
# ═══════════════════════════════════════════════════════════════

def classify_stroke_type(joints_3d, is_right_handed=True):
    """
    단일 프레임의 관절 3D 좌표에서 스트로크 유형을 판별.

    Parameters
    ----------
    joints_3d : ndarray (J, 3)
        SMPL-X 출력 관절 좌표
    is_right_handed : bool

    Returns
    -------
    int : StrokeType value
    """
    # SMPL-X joint indices
    pelvis = joints_3d[0]
    head   = joints_3d[15]

    if is_right_handed:
        wrist = joints_3d[21]     # right_wrist
        shoulder = joints_3d[17]  # right_shoulder
        opp_shoulder = joints_3d[16]
    else:
        wrist = joints_3d[20]
        shoulder = joints_3d[16]
        opp_shoulder = joints_3d[17]

    # 서브 감지: 손목이 머리 위에 있을 때
    # SMPL-X에서 Y축이 위-아래 (음수가 위)
    if wrist[1] < head[1] - 0.10:
        return StrokeType.SERVE

    # 체간 정면 방향 계산
    left_hip  = joints_3d[1]
    right_hip = joints_3d[2]
    hip_mid   = (left_hip + right_hip) / 2.0

    left_sh  = joints_3d[16]
    right_sh = joints_3d[17]
    sh_mid   = (left_sh + right_sh) / 2.0

    # 체간 전방 벡터 (hip → shoulder 투영)
    body_up = sh_mid - hip_mid
    body_up[1] = 0  # Y(상하) 제거하여 수평 성분만

    # 좌우 벡터
    body_right = right_sh - left_sh
    body_right[1] = 0

    # 손목의 좌우 위치
    wrist_vec = wrist - hip_mid
    wrist_vec[1] = 0
    lateral = np.dot(wrist_vec, body_right)

    if is_right_handed:
        if lateral > 0:
            return StrokeType.FOREHAND
        else:
            return StrokeType.BACKHAND
    else:
        if lateral < 0:
            return StrokeType.FOREHAND
        else:
            return StrokeType.BACKHAND


# ═══════════════════════════════════════════════════════════════
# 4. 스윙 단계 분류
# ═══════════════════════════════════════════════════════════════

def compute_wrist_velocities(joints_seq, fps=30, is_right_handed=True):
    """
    프레임별 손목 속도 벡터와 스칼라 속도를 계산.

    Parameters
    ----------
    joints_seq : ndarray (T, J, 3)
    fps : int
    is_right_handed : bool

    Returns
    -------
    velocities : ndarray (T, 3)
    speeds : ndarray (T,)
    """
    wrist_idx = 21 if is_right_handed else 20
    wrist_pos = joints_seq[:, wrist_idx, :]  # (T, 3)

    T = wrist_pos.shape[0]
    dt = 1.0 / fps

    velocities = np.zeros_like(wrist_pos)
    # 중앙 차분
    velocities[1:-1] = (wrist_pos[2:] - wrist_pos[:-2]) / (2 * dt)
    velocities[0]    = (wrist_pos[1] - wrist_pos[0]) / dt
    velocities[-1]   = (wrist_pos[-1] - wrist_pos[-2]) / dt

    speeds = np.linalg.norm(velocities, axis=1)
    return velocities, speeds


def compute_shoulder_hip_separation(joints_seq):
    """
    프레임별 어깨-골반 분리 각도 (trunk rotation) 계산.

    Returns
    -------
    separations : ndarray (T,) in degrees
    """
    T = joints_seq.shape[0]
    separations = np.zeros(T)

    for t in range(T):
        left_sh  = joints_seq[t, 16]
        right_sh = joints_seq[t, 17]
        left_hip = joints_seq[t, 1]
        right_hip = joints_seq[t, 2]

        sh_vec  = right_sh - left_sh
        hip_vec = right_hip - left_hip

        # XZ 평면에 투영
        sh_2d  = np.array([sh_vec[0], sh_vec[2]])
        hip_2d = np.array([hip_vec[0], hip_vec[2]])

        # 두 벡터 사이 각도
        cos_a = np.clip(
            np.dot(sh_2d, hip_2d) /
            (np.linalg.norm(sh_2d) * np.linalg.norm(hip_2d) + 1e-8),
            -1, 1
        )
        separations[t] = np.degrees(np.arccos(cos_a))

    return separations


def classify_swing_phases(joints_seq, fps=30, is_right_handed=True):
    """
    전체 시퀀스에서 각 프레임의 스윙 단계를 분류.

    Parameters
    ----------
    joints_seq : ndarray (T, J, 3)
    fps : int
    is_right_handed : bool

    Returns
    -------
    phases : ndarray (T,) int - SwingPhase values
    stroke_types : ndarray (T,) int - StrokeType values
    """
    T = joints_seq.shape[0]
    _, speeds = compute_wrist_velocities(joints_seq, fps, is_right_handed)
    separations = compute_shoulder_hip_separation(joints_seq)

    # 속도 스무딩 (노이즈 감소)
    kernel_size = min(5, T)
    if kernel_size >= 3:
        kernel = np.ones(kernel_size) / kernel_size
        speeds_smooth = np.convolve(speeds, kernel, mode='same')
    else:
        speeds_smooth = speeds.copy()

    # 적응적 임계값 (데이터 기반)
    v_median = np.median(speeds_smooth)
    v_std    = np.std(speeds_smooth)
    V_LOW  = max(v_median * 0.3, 0.1)
    V_HIGH = v_median + 0.5 * v_std

    SEP_THRESH = 10.0  # degrees

    phases       = np.full(T, SwingPhase.READY, dtype=int)
    stroke_types = np.full(T, StrokeType.FOREHAND, dtype=int)

    # 속도 기울기 (가속도)
    speed_grad = np.gradient(speeds_smooth)

    for t in range(T):
        # 스트로크 유형 판별
        stroke_types[t] = classify_stroke_type(joints_seq[t], is_right_handed)

        v = speeds_smooth[t]
        sep = separations[t]
        grad = speed_grad[t]

        if v < V_LOW and sep < SEP_THRESH:
            # 느리고 체간 회전 적음
            if t > T * 0.7:
                phases[t] = SwingPhase.RECOVERY
            else:
                phases[t] = SwingPhase.READY
        elif grad > 0 and v < V_HIGH and sep > SEP_THRESH:
            # 속도 증가 중이지만 아직 높지 않음 + 체간 회전
            phases[t] = SwingPhase.BACKSWING
        elif grad > 0 and v >= V_HIGH:
            # 빠른 가속
            phases[t] = SwingPhase.ACCELERATION
        elif v >= V_HIGH and grad <= 0:
            # 속도 피크 부근 (감속 시작)
            phases[t] = SwingPhase.CONTACT
        elif grad < 0 and v > V_LOW:
            # 감속 중
            phases[t] = SwingPhase.FOLLOW_THROUGH
        else:
            phases[t] = SwingPhase.RECOVERY

    # 시간적 스무딩 — 너무 빠른 단계 전환 방지
    phases = _temporal_smooth_phases(phases, min_duration=3)

    return phases, stroke_types


def _temporal_smooth_phases(phases, min_duration=3):
    """
    너무 짧은 단계(< min_duration 프레임)를 주변 단계로 흡수.
    """
    T = len(phases)
    smoothed = phases.copy()

    i = 0
    while i < T:
        j = i
        while j < T and smoothed[j] == smoothed[i]:
            j += 1
        duration = j - i

        if duration < min_duration and i > 0:
            # 이전 단계로 흡수
            smoothed[i:j] = smoothed[i - 1]

        i = j

    return smoothed


# ═══════════════════════════════════════════════════════════════
# 5. 라켓 방향 계산 (SLERP 보간)
# ═══════════════════════════════════════════════════════════════

def compute_racket_orientations(phases, stroke_types, blend_frames=5):
    """
    각 프레임의 라켓 방향을 스윙 단계에 맞게 계산 (SLERP 보간).

    Parameters
    ----------
    phases : ndarray (T,) int
    stroke_types : ndarray (T,) int
    blend_frames : int
        단계 전환 시 보간할 프레임 수

    Returns
    -------
    racket_rotations : list of Rotation (T,)
        각 프레임에서의 라켓 로컬 회전 (손목 기준)
    """
    T = len(phases)
    racket_quats = []

    for t in range(T):
        stroke = stroke_types[t]
        phase  = phases[t]
        quat_dict = RACKET_QUATS.get(stroke, RACKET_QUATS[StrokeType.FOREHAND])
        racket_quats.append(quat_dict[phase])

    # SLERP 보간: 단계 전환 경계에서 부드럽게 전환
    result = [racket_quats[0]]

    for t in range(1, T):
        if phases[t] != phases[t-1]:
            # 단계 전환 감지 → 이후 blend_frames 동안 보간
            start_rot = racket_quats[t-1]
            end_rot   = racket_quats[t]

            # SLERP 키프레임
            key_rots = R.concatenate([start_rot, end_rot])
            slerp = Slerp([0, 1], key_rots)

            for bf in range(min(blend_frames, T - t)):
                alpha = (bf + 1) / blend_frames
                alpha = min(alpha, 1.0)

                if t + bf < len(result):
                    result[t + bf] = slerp(alpha)
                else:
                    result.append(slerp(alpha))
        else:
            if t >= len(result):
                result.append(racket_quats[t])

    # 길이 맞추기
    while len(result) < T:
        result.append(racket_quats[-1])

    return result[:T]


# ═══════════════════════════════════════════════════════════════
# 6. 라켓 메쉬 생성
# ═══════════════════════════════════════════════════════════════

def create_racket_mesh():
    """
    간단한 테니스 라켓 메쉬 (직사각형 면 + 손잡이).

    Returns
    -------
    vertices : ndarray (N, 3) 로컬 좌표 (손목 원점)
    faces : ndarray (F, 3) 삼각형 인덱스
    """
    # 라켓 헤드 (타원 근사 — 직사각형)
    # 길이: ~28cm, 너비: ~23cm, 손잡이: ~15cm
    # 로컬 좌표: Y축이 라켓 길이 방향, Z축이 면 법선

    hw = 0.115   # 반폭 11.5cm
    hh = 0.140   # 반높이 14cm
    grip_len = 0.15  # 손잡이 길이
    grip_w   = 0.015 # 손잡이 반폭
    thickness = 0.005

    vertices = np.array([
        # 라켓 헤드 (앞면) — 0~3
        [-hw,  grip_len,          thickness],
        [ hw,  grip_len,          thickness],
        [ hw,  grip_len + 2*hh,   thickness],
        [-hw,  grip_len + 2*hh,   thickness],

        # 라켓 헤드 (뒷면) — 4~7
        [-hw,  grip_len,         -thickness],
        [ hw,  grip_len,         -thickness],
        [ hw,  grip_len + 2*hh,  -thickness],
        [-hw,  grip_len + 2*hh,  -thickness],

        # 손잡이 — 8~11
        [-grip_w, 0,          grip_w],
        [ grip_w, 0,          grip_w],
        [ grip_w, grip_len,   grip_w],
        [-grip_w, grip_len,   grip_w],

        # 손잡이 뒤 — 12~15
        [-grip_w, 0,         -grip_w],
        [ grip_w, 0,         -grip_w],
        [ grip_w, grip_len,  -grip_w],
        [-grip_w, grip_len,  -grip_w],
    ], dtype=np.float64)

    faces = np.array([
        # 헤드 앞면
        [0, 1, 2], [0, 2, 3],
        # 헤드 뒷면
        [4, 6, 5], [4, 7, 6],
        # 헤드 옆면
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
        # 손잡이 앞
        [8, 9, 10], [8, 10, 11],
        # 손잡이 뒤
        [12, 14, 13], [12, 15, 14],
        # 손잡이 옆
        [8, 12, 13], [8, 13, 9],
        [9, 13, 14], [9, 14, 10],
        [10, 14, 15], [10, 15, 11],
        [11, 15, 12], [11, 12, 8],
    ], dtype=np.int64)

    return vertices, faces


def transform_racket(racket_verts, wrist_pos, racket_rotation):
    """
    라켓 로컬 정점을 월드 좌표로 변환.

    Parameters
    ----------
    racket_verts : ndarray (N, 3)
    wrist_pos : ndarray (3,)
    racket_rotation : Rotation

    Returns
    -------
    world_verts : ndarray (N, 3)
    """
    rotated = racket_rotation.apply(racket_verts)
    return rotated + wrist_pos
