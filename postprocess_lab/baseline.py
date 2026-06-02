# -*- coding: utf-8 -*-
"""현재 파이프라인(One-Euro → XPBD) 베이스라인 진단.
팔(팔꿈치/손목) 관절의 트위스트/플래싱을 정량화한다.
실행: venv\\Scripts\\python postprocess_lab\\baseline.py [data.pt]
"""
import sys, os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xpbd_constraints import apply_xpbd_constraints
from postprocess_lab.metrics import summarize, print_table

# ── One-Euro 필터 (animate_postprocess.py와 동일 로직 복제) ──────────
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = x0.copy(); self.dx_prev = np.zeros_like(x0); self.t_prev = t0
    def _sf(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e; return r / (r + 1)
    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return x
        dx = (x - self.x_prev) / t_e
        ad = self._sf(t_e, self.d_cutoff)
        dx_hat = ad * dx + (1 - ad) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._sf(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat.copy(); self.dx_prev = dx_hat.copy(); self.t_prev = t
        return x_hat

def filter_rotations(rot_seq, fps=30):
    T, N, _ = rot_seq.shape
    dt = 1.0 / fps
    euler = R.from_rotvec(rot_seq.reshape(-1, 3)).as_euler('XYZ').reshape(T, N, 3)
    euler_u = np.unwrap(euler, axis=0)
    oef = OneEuroFilter(0.0, euler_u[0].flatten(), min_cutoff=0.8, beta=0.03)
    out = np.zeros_like(rot_seq)
    for i in range(T):
        sm = oef(i * dt, euler_u[i].flatten()).reshape(N, 3)
        out[i] = R.from_euler('XYZ', sm).as_rotvec().reshape(N, 3)
    return out


ARM_JOINTS = {"left_elbow": 17, "right_elbow": 18, "left_wrist": 19, "right_wrist": 20}


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "smplx_hmp_injected_full.pt"
    data = torch.load(data_path, map_location="cpu")
    params = data["smpl_params_global"]
    bp = params["body_pose"]
    bp = bp.numpy() if isinstance(bp, torch.Tensor) else np.asarray(bp)
    T = bp.shape[0]
    bp = bp.reshape(T, 21, 3).astype(np.float64)
    print(f"data={data_path}  T={T} frames")

    raw = bp.copy()
    oneeuro = filter_rotations(raw, fps=30)
    xpbd, _ = apply_xpbd_constraints(oneeuro.reshape(T, 63), fps=30,
                                     compliance=0.001, num_iterations=8, num_substeps=4)
    xpbd = xpbd.reshape(T, 21, 3)

    for jname, jidx in ARM_JOINTS.items():
        print(f"\n=== {jname} (body_pose idx {jidx}) ===")
        rows = [
            summarize("raw", raw[:, jidx]),
            summarize("one-euro", oneeuro[:, jidx], ref_rotvecs=raw[:, jidx]),
            summarize("one-euro+xpbd", xpbd[:, jidx], ref_rotvecs=raw[:, jidx]),
        ]
        print_table(rows)


if __name__ == "__main__":
    main()
