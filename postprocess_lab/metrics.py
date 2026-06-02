# -*- coding: utf-8 -*-
"""회전 시퀀스 품질 지표 (트위스트/플래싱/저크/충실도).
순수 numpy+scipy. Blender 불필요."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotvec_to_quat_wxyz(rotvecs):
    """(T,3) rotvec → (T,4) [w,x,y,z], 부호 연속성 미보정(원시)."""
    q_xyzw = R.from_rotvec(rotvecs).as_quat()  # scipy: [x,y,z,w]
    return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])


def geodesic_angular_velocity_deg(rotvecs):
    """연속 프레임 간 측지 회전각(deg/frame). 반환 (T-1,)."""
    q = rotvec_to_quat_wxyz(rotvecs)
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def quaternion_flip_count(rotvecs):
    """원시 쿼터니언 부호 뒤집힘(q·q_prev < 0) 횟수 — 데이터 불연속 지표."""
    q = rotvec_to_quat_wxyz(rotvecs)
    dots = np.sum(q[1:] * q[:-1], axis=1)
    return int(np.sum(dots < 0.0))


def swing_twist_angles_deg(rotvecs, axis):
    """주어진 축(unit, 3,)에 대한 twist 각(deg) 시퀀스 (T,)."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    q = rotvec_to_quat_wxyz(rotvecs)
    w = q[:, 0]
    v = q[:, 1:]
    proj = (v @ axis)[:, None] * axis[None, :]
    tw = np.column_stack([w, proj])
    n = np.linalg.norm(tw, axis=1)
    n[n < 1e-9] = 1.0
    tw /= n[:, None]
    sign = np.sign(v @ axis)
    sign[sign == 0] = 1.0
    ang = 2.0 * np.arctan2(np.linalg.norm(proj, axis=1) * sign, tw[:, 0])
    return np.degrees(ang)


def jerk_rms_deg(rotvecs):
    """각가속도(2차차분) RMS (deg/frame²) — 떨림/플래싱 민감 지표."""
    av = geodesic_angular_velocity_deg(rotvecs)          # (T-1,)
    if len(av) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(av) ** 2)))


def fidelity_rms_deg(rotvecs, ref_rotvecs):
    """ref 대비 측지 편차 RMS(deg) — 과도한 스무딩(원본 훼손) 측정."""
    qa = R.from_rotvec(rotvecs)
    qb = R.from_rotvec(ref_rotvecs)
    rel = (qa.inv() * qb).magnitude()  # rad, (T,)
    return float(np.sqrt(np.mean(np.degrees(rel) ** 2)))


def dominant_twist_axis(rotvecs):
    """회전 벡터들의 주성분축(트위스트가 가장 큰 방향 추정)."""
    rv = np.asarray(rotvecs, float)
    rv = rv - rv.mean(0)
    _, _, vt = np.linalg.svd(rv, full_matrices=False)
    return vt[0]


def summarize(name, rotvecs, ref_rotvecs=None, spike_deg=25.0, twist_axis=None):
    av = geodesic_angular_velocity_deg(rotvecs)
    # 트위스트 비교를 위해 축은 고정(raw 기준)으로 받는다. 미지정 시 자체 PCA.
    axis = dominant_twist_axis(rotvecs) if twist_axis is None else twist_axis
    tw = swing_twist_angles_deg(rotvecs, axis)
    out = {
        "name": name,
        "angvel_mean": float(av.mean()),
        "angvel_p99": float(np.percentile(av, 99)),
        "angvel_max": float(av.max()),
        "spikes": int(np.sum(av > spike_deg)),
        "jerk_rms": jerk_rms_deg(rotvecs),
        "twist_range": float(tw.max() - tw.min()),
        "twist_absmax": float(np.abs(tw).max()),
        "flips": quaternion_flip_count(rotvecs),
    }
    if ref_rotvecs is not None:
        out["fidelity_rms"] = fidelity_rms_deg(rotvecs, ref_rotvecs)
    return out


def print_table(rows):
    cols = ["name", "angvel_mean", "angvel_p99", "angvel_max", "spikes",
            "jerk_rms", "twist_range", "twist_absmax", "flips", "fidelity_rms"]
    hdr = f"{'method':<22s} {'avμ':>7s} {'avP99':>7s} {'avMax':>7s} {'spk':>5s} {'jerk':>7s} {'twRng':>7s} {'twMax':>7s} {'flip':>5s} {'fidel':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<22s} {r['angvel_mean']:7.2f} {r['angvel_p99']:7.2f} "
              f"{r['angvel_max']:7.2f} {r['spikes']:5d} {r['jerk_rms']:7.2f} "
              f"{r['twist_range']:7.1f} {r['twist_absmax']:7.1f} {r['flips']:5d} "
              f"{r.get('fidelity_rms', float('nan')):7.2f}")
