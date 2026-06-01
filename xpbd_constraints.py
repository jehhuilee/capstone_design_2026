# -*- coding: utf-8 -*-
"""
xpbd_constraints.py
- XPBD (Extended Position-Based Dynamics) 솔버
- SMPL-X body_pose 21관절에 대한 해부학적 관절 각도 제한
- Compliance 기반 부드러운 보정 (단순 clamping 대비 자연스러움)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# ═══════════════════════════════════════════════════════════════
# 1. SMPL-X body_pose 관절 인덱스 매핑 (21관절)
#    body_pose[i*3 : i*3+3] = joint (i+1) 의 axis-angle
# ═══════════════════════════════════════════════════════════════

BODY_JOINT_NAMES = [
    'left_hip',        # 0  → joint 1
    'right_hip',       # 1  → joint 2
    'spine1',          # 2  → joint 3
    'left_knee',       # 3  → joint 4
    'right_knee',      # 4  → joint 5
    'spine2',          # 5  → joint 6
    'left_ankle',      # 6  → joint 7
    'right_ankle',     # 7  → joint 8
    'spine3',          # 8  → joint 9
    'left_foot',       # 9  → joint 10
    'right_foot',      # 10 → joint 11
    'neck',            # 11 → joint 12
    'left_collar',     # 12 → joint 13
    'right_collar',    # 13 → joint 14
    'head',            # 14 → joint 15
    'left_shoulder',   # 15 → joint 16
    'right_shoulder',  # 16 → joint 17
    'left_elbow',      # 17 → joint 18
    'right_elbow',     # 18 → joint 19
    'left_wrist',      # 19 → joint 20
    'right_wrist',     # 20 → joint 21
]

# ═══════════════════════════════════════════════════════════════
# 2. 해부학적 관절 각도 제한 (degrees)
#    각 관절에 대해 (X_min, X_max, Y_min, Y_max, Z_min, Z_max)
#    X: flexion/extension, Y: abduction/adduction, Z: rotation
#    None → 해당 관절에 제한 없음
# ═══════════════════════════════════════════════════════════════

# (x_min, x_max, y_min, y_max, z_min, z_max) in degrees
JOINT_LIMITS_DEG = {
    'left_hip':        (-30, 120,  -30, 45,  -45, 45),
    'right_hip':       (-30, 120,  -45, 30,  -45, 45),
    'spine1':          (-40,  45,  -35, 35,  -30, 30),
    'left_knee':       (-10, 135,  -5,   5,   -5,  5),
    'right_knee':      (-10, 135,  -5,   5,   -5,  5),
    'spine2':          (-40,  45,  -35, 35,  -30, 30),
    'left_ankle':      (-50,  30,  -20, 20,  -20, 20),
    'right_ankle':     (-50,  30,  -20, 20,  -20, 20),
    'spine3':          (-40,  45,  -35, 35,  -30, 30),
    'left_foot':       (-30,  50,  -15, 15,  -10, 10),
    'right_foot':      (-30,  50,  -15, 15,  -10, 10),
    'neck':            (-60,  70,  -45, 45,  -80, 80),
    'left_collar':     (-15,  15,  -15, 15,  -15, 15),
    'right_collar':    (-15,  15,  -15, 15,  -15, 15),
    'head':            (-50,  40,  -35, 35,  -60, 60),
    'left_shoulder':   (-60, 180, -180, 45,  -90, 90),
    'right_shoulder':  (-60, 180,  -45, 180, -90, 90),
    'left_elbow':      (-15, 150,  -5,   5,  -90, 90),
    'right_elbow':     (-15, 150,  -5,   5,  -90, 90),
    'left_wrist':      (-70,  80,  -30, 20,  -10, 10),
    'right_wrist':     (-70,  80,  -30, 20,  -10, 10),
}

def _deg2rad(limits_deg):
    """Convert degree limits dict to radians."""
    limits_rad = {}
    for name, (x0, x1, y0, y1, z0, z1) in limits_deg.items():
        limits_rad[name] = (
            np.radians(x0), np.radians(x1),
            np.radians(y0), np.radians(y1),
            np.radians(z0), np.radians(z1),
        )
    return limits_rad

JOINT_LIMITS_RAD = _deg2rad(JOINT_LIMITS_DEG)


# ═══════════════════════════════════════════════════════════════
# 3. XPBD 솔버
# ═══════════════════════════════════════════════════════════════

class XPBDSolver:
    """
    XPBD (Extended Position-Based Dynamics) 기반 관절 각도 제한 솔버.

    단순 clamping 대비 장점:
    - Compliance(α)로 "부드러운" 제한 경계 접근
    - Lagrange 승수 누적으로 과도 보정 방지
    - 물리적 에너지 보존
    """

    def __init__(self, compliance=0.001, num_iterations=8, num_substeps=4):
        """
        Parameters
        ----------
        compliance : float
            높을수록 부드러운 제한 (0 = 완전히 단단한 제한)
            권장: 0.0001 (매우 단단) ~ 0.01 (부드러움)
        num_iterations : int
            각 서브스텝 내 반복 횟수
        num_substeps : int
            프레임 당 서브스텝 수 (높을수록 안정적)
        """
        self.compliance = compliance
        self.num_iterations = num_iterations
        self.num_substeps = num_substeps

    def solve_sequence(self, body_poses, fps=30):
        """
        전체 시퀀스에 대해 XPBD 관절 제한 적용.

        Parameters
        ----------
        body_poses : ndarray (T, 63)
            SMPL-X body_pose (21관절 × 3 axis-angle)
        fps : int
            프레임 레이트

        Returns
        -------
        constrained : ndarray (T, 63)
            관절 제한이 적용된 body_pose
        violations : ndarray (T,)
            각 프레임의 총 제한 위반량 (디버그용)
        """
        T = body_poses.shape[0]
        dt = 1.0 / fps
        sub_dt = dt / self.num_substeps

        # α̃ = α / Δt² (scaled compliance)
        alpha_tilde = self.compliance / (sub_dt * sub_dt)

        constrained = body_poses.copy()
        violations = np.zeros(T)

        for t in range(T):
            pose = constrained[t].copy()  # (63,)
            frame_violation = 0.0

            for _substep in range(self.num_substeps):
                # 각 관절에 대해 Euler 각도 제한 적용
                pose_joints = pose.reshape(21, 3)

                # Lagrange 승수 초기화 (각 서브스텝 시작 시)
                lambdas = np.zeros((21, 3))

                for _iter in range(self.num_iterations):
                    for j_idx in range(21):
                        joint_name = BODY_JOINT_NAMES[j_idx]
                        if joint_name not in JOINT_LIMITS_RAD:
                            continue

                        limits = JOINT_LIMITS_RAD[joint_name]
                        rotvec = pose_joints[j_idx]

                        # axis-angle → Euler XYZ
                        try:
                            euler = R.from_rotvec(rotvec).as_euler('XYZ')
                        except Exception:
                            continue

                        # 각 축에 대해 제한 위반 체크 및 XPBD 보정
                        for axis in range(3):
                            lo = limits[axis * 2]
                            hi = limits[axis * 2 + 1]
                            theta = euler[axis]

                            # 제약 조건 C
                            if theta < lo:
                                C = theta - lo  # 음수
                            elif theta > hi:
                                C = theta - hi  # 양수
                            else:
                                C = 0.0

                            if abs(C) < 1e-6:
                                continue

                            frame_violation += abs(C)

                            # XPBD 보정량 계산
                            # Δλ = (-C - α̃ · λ) / (w + α̃)
                            # w = 1.0 (단위 질량)
                            w = 1.0
                            delta_lambda = (-C - alpha_tilde * lambdas[j_idx, axis]) / (w + alpha_tilde)
                            lambdas[j_idx, axis] += delta_lambda

                            # 보정 적용
                            euler[axis] += delta_lambda * w

                        # Euler → axis-angle 복원
                        try:
                            pose_joints[j_idx] = R.from_euler('XYZ', euler).as_rotvec()
                        except Exception:
                            pass

                pose = pose_joints.flatten()

            constrained[t] = pose
            violations[t] = frame_violation

        return constrained, violations


def apply_xpbd_constraints(body_poses, fps=30, compliance=0.001,
                           num_iterations=8, num_substeps=4):
    """
    편의 함수: body_pose 시퀀스에 XPBD 관절 제한을 적용합니다.

    Parameters
    ----------
    body_poses : ndarray (T, 63)
    fps : int
    compliance : float
    num_iterations : int
    num_substeps : int

    Returns
    -------
    constrained_poses : ndarray (T, 63)
    total_violations : float
    """
    solver = XPBDSolver(
        compliance=compliance,
        num_iterations=num_iterations,
        num_substeps=num_substeps,
    )
    constrained, violations = solver.solve_sequence(body_poses, fps) 

    total = violations.sum()
    print(f"   XPBD: 총 {total:.2f} rad 위반 보정 완료 "
          f"(평균 {violations.mean():.4f} rad/frame)")

    return constrained, total
