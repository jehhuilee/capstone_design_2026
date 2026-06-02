# -*- coding: utf-8 -*-
"""5가지 접근법(+조합)을 실제 팔 관절 데이터로 비교.
실행: venv\\Scripts\\python postprocess_lab\\compare.py [data.pt]
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xpbd_constraints import apply_xpbd_constraints
from postprocess_lab.metrics import summarize, print_table
from postprocess_lab.baseline import filter_rotations, ARM_JOINTS
from postprocess_lab import methods as M


def load_arms(data_path):
    data = torch.load(data_path, map_location="cpu")
    bp = data["smpl_params_global"]["body_pose"]
    bp = bp.numpy() if isinstance(bp, torch.Tensor) else np.asarray(bp)
    T = bp.shape[0]
    return bp.reshape(T, 21, 3).astype(np.float64), T


def baseline_pipeline(bp, T):
    oe = filter_rotations(bp, fps=30)
    xp, _ = apply_xpbd_constraints(oe.reshape(T, 63), fps=30,
                                   compliance=0.001, num_iterations=8, num_substeps=4)
    return xp.reshape(T, 21, 3)


# 후보 파이프라인: 관절별 rotvec (T,3) → (T,3)
CANDIDATES = {
    "M1 continuity":        lambda x: M.m1_continuity(x),
    "M2 swingtwist":        lambda x: M.m2_swing_twist_rom(x, twist_min_deg=-90, twist_max_deg=90),
    "M3 angvel15":          lambda x: M.m3_angvel_clamp(x, max_deg_per_frame=15),
    "M5 savgol":            lambda x: M.m5_savgol(x, window=11, poly=3),
    "M1+M2":                lambda x: M.m2_swing_twist_rom(M.m1_continuity(x), twist_min_deg=-90, twist_max_deg=90),
    "M1+M3":                lambda x: M.m3_angvel_clamp(M.m1_continuity(x), 15),
    "M1+M5":                lambda x: M.m5_savgol(M.m1_continuity(x)),
    # M2u = unwrap+smooth twist (개선판)
    "M2u twiststab":        lambda x: M.m2_swing_twist_rom(x, twist_min_deg=-90, twist_max_deg=90, smooth_twist_window=11),
    "M1+M2u+M3":            lambda x: M.m3_angvel_clamp(M.m2_swing_twist_rom(M.m1_continuity(x), twist_min_deg=-90, twist_max_deg=90, smooth_twist_window=11), 15),
    "M1+M5+M2u":            lambda x: M.m2_swing_twist_rom(M.m5_savgol(M.m1_continuity(x)), twist_min_deg=-90, twist_max_deg=90, smooth_twist_window=11),
    "M1+M5+M2u+M3":         lambda x: M.m3_angvel_clamp(M.m2_swing_twist_rom(M.m5_savgol(M.m1_continuity(x)), twist_min_deg=-90, twist_max_deg=90, smooth_twist_window=11), 15),
}


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data_pt/smplx_hmp_injected_full.pt"
    bp, T = load_arms(data_path)
    print(f"data={data_path}  T={T}")

    base = baseline_pipeline(bp, T)

    from postprocess_lab.metrics import dominant_twist_axis
    for jname, jidx in ARM_JOINTS.items():
        raw = bp[:, jidx]
        ax = dominant_twist_axis(raw)   # 고정 트위스트 축(raw 기준)
        rows = [summarize("raw", raw, twist_axis=ax),
                summarize("[current] OE+XPBD", base[:, jidx], ref_rotvecs=raw, twist_axis=ax)]
        for cname, fn in CANDIDATES.items():
            rows.append(summarize(cname, fn(raw), ref_rotvecs=raw, twist_axis=ax))
        print(f"\n=== {jname} ===")
        print_table(rows)


if __name__ == "__main__":
    main()
