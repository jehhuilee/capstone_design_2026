import torch
import numpy as np
import pyvista as pv
import smplx
import time
from scipy.spatial.transform import Rotation as R

from xpbd_constraints import apply_xpbd_constraints
from swing_classifier import (
    classify_swing_phases,
    PHASE_NAMES, STROKE_NAMES,
)


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


def filter_rotations(rot_seq, fps=30):
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
    T = trans_seq.shape[0]
    dt  = 1.0 / fps
    oef = OneEuroFilter(0.0, trans_seq[0].copy(), min_cutoff=0.8, beta=0.03)
    out = np.zeros_like(trans_seq)
    for i in range(T):
        out[i] = oef(i * dt, trans_seq[i])
    return out


def smplx_forward_chunked(model, params_np, T, chunk=64):
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


def fix_two_handed_grip_ik(model, params_np, stroke_types, phases):
    from swing_classifier import StrokeType, SwingPhase
    T = params_np['body_pose'].shape[0]
    
    target_frames = []
    for t in range(T):
        if stroke_types[t] == StrokeType.BACKHAND and phases[t] not in [SwingPhase.RECOVERY, SwingPhase.READY]:
            target_frames.append(t)
            
    if not target_frames:
        return params_np
        
    print(f"Step 1.5: grip IK correction target: {len(target_frames)}")
    
    model.eval()
    
    fixed_params = {}
    for k, v in params_np.items():
        fixed_params[k] = torch.tensor(v).float()
    if 'expression' not in fixed_params:
        fixed_params['expression'] = torch.zeros(T, 10, dtype=torch.float32)
        
    body_pose = fixed_params['body_pose'].clone()
    
    ls_idx, le_idx, lw_idx = 16*3, 18*3, 20*3
    
    left_arm_params = torch.zeros(len(target_frames), 9, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for i, t in enumerate(target_frames):
            left_arm_params[i, 0:3] = body_pose[t, ls_idx:ls_idx+3]
            left_arm_params[i, 3:6] = body_pose[t, le_idx:le_idx+3]
            left_arm_params[i, 6:9] = body_pose[t, lw_idx:lw_idx+3]
            
    optimizer = torch.optim.Adam([left_arm_params], lr=0.05)
    
    for step in range(30):
        optimizer.zero_grad()
        
        curr_body_pose = body_pose[target_frames].clone()
        curr_body_pose[:, ls_idx:ls_idx+3] = left_arm_params[:, 0:3]
        curr_body_pose[:, le_idx:le_idx+3] = left_arm_params[:, 3:6]
        curr_body_pose[:, lw_idx:lw_idx+3] = left_arm_params[:, 6:9]
        
        feed = {k: v[target_frames] for k, v in fixed_params.items()}
        feed['body_pose'] = curr_body_pose
        
        out = model(**feed)
        joints = out.joints
        
        left_wrist = joints[:, 20]
        right_wrist = joints[:, 21]
        
        target_dist = 0.10
        dists = torch.norm(left_wrist - right_wrist, dim=1)
        loss = torch.mean((dists - target_dist)**2)
        
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        for i, t in enumerate(target_frames):
            body_pose[t, ls_idx:ls_idx+3] = left_arm_params[i, 0:3]
            body_pose[t, le_idx:le_idx+3] = left_arm_params[i, 3:6]
            body_pose[t, lw_idx:lw_idx+3] = left_arm_params[i, 6:9]
            
    params_np['body_pose'] = body_pose.numpy()
    return params_np


def run_animation():
    data_path  = r"c:\Users\user\Desktop\CG\캡스톤\4k_tennis\smplx_merged_hamer.pt"
    model_path = r"model\SMPLX_NEUTRAL.npz"

    print(f"Loading data: {data_path}")
    data   = torch.load(data_path, map_location='cpu')
    params = data['smpl_params_global']

    T   = params['body_pose'].shape[0]
    fps = 30
    print(f"Total {T} frames, {fps} fps")

    rot_keys = [
        'global_orient', 'body_pose',
        'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose',
    ]

    raw_params      = {}
    smoothed_params = {}

    print("Applying One-Euro filter...")
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
            smoothed_params[k] = val.copy()

    print(f"Loading SMPL-X model: {model_path}")
    smplx_model = smplx.create(
        model_path, model_type='smplx',
        use_pca=False, batch_size=1,
    )
    faces = smplx_model.faces

    print("Computing raw FK...")
    verts_raw, joints_raw = smplx_forward_chunked(smplx_model, raw_params, T, chunk=64)
    print(f"verts_raw = {verts_raw.shape}")

    print("Analyzing swing phases...")
    phases, stroke_types = classify_swing_phases(joints_raw, fps=fps, is_right_handed=True)
    
    smoothed_params = fix_two_handed_grip_ik(smplx_model, smoothed_params, stroke_types, phases)

    print("Applying XPBD constraints...")
    constrained_body_pose, total_violations = apply_xpbd_constraints(
        smoothed_params['body_pose'],
        fps=fps,
        compliance=0.001,
        num_iterations=8,
        num_substeps=4,
    )
    smoothed_params['body_pose'] = constrained_body_pose

    print("Computing smoothed FK...")
    verts_smooth, joints_smooth = smplx_forward_chunked(smplx_model, smoothed_params, T, chunk=64)
    print(f"verts_smooth = {verts_smooth.shape}")

    from collections import Counter
    phase_counts = Counter(phases)
    for p, count in sorted(phase_counts.items()):
        print(f"{PHASE_NAMES[p]:>16s}: {count:>5d} frames ({100*count/T:.1f}%)")

    print("Initializing PyVista plotter...")

    pv_faces = np.column_stack(
        (np.full(len(faces), 3, dtype=np.int64), faces)
    ).flatten()

    mesh_raw    = pv.PolyData(verts_raw[0].copy(), pv_faces)
    mesh_smooth = pv.PolyData(verts_smooth[0].copy(), pv_faces)

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800),
                     title="SMPL-X Post-Processing: Before vs After")

    pl.subplot(0, 0)
    pl.add_text("Before (Raw)", color='#ff7b72', font_size=14)
    pl.add_mesh(mesh_raw, color='#ffa0a0', smooth_shading=True,
                specular=0.5, specular_power=30)

    pl.subplot(0, 1)
    phase_text_actor = pl.add_text(
        "After | FOREHAND",
        color='#79c0ff', font_size=14,
    )
    pl.add_mesh(mesh_smooth, color='#a0d0ff', smooth_shading=True,
                specular=0.5, specular_power=30)

    pl.link_views()
    prev_center = verts_raw[0].mean(axis=0)
    cam_offset = np.array([0, 0, 5.0])
    pl.camera.focal_point = prev_center
    pl.camera.position    = prev_center + cam_offset
    pl.camera.up          = (0, 1, 0)

    print(f"Playing animation: {T} frames, {fps} fps")

    pl.show(interactive_update=True)

    frame = 0
    while True:
        try:
            idx = frame % T
            mesh_raw.points    = verts_raw[idx]
            mesh_smooth.points = verts_smooth[idx]

            stroke_name = STROKE_NAMES[stroke_types[idx]]
            phase_text_actor.SetText(
                0,
                f"After | {stroke_name}"
            )

            curr_center = verts_raw[idx].mean(axis=0)
            delta = curr_center - prev_center
            pl.camera.focal_point = np.array(pl.camera.focal_point) + delta
            pl.camera.position    = np.array(pl.camera.position) + delta
            prev_center = curr_center

            pl.update()
            frame += 1
            time.sleep(1.0 / fps)
        except Exception:
            break
            
    out_npz = data_path.replace('.pt', '_postprocessed.npz')
    print(f"Saving postprocessed npz: {out_npz}")
    np.savez(out_npz, **smoothed_params)
    
    import subprocess
    import os
    print("Starting Unreal Engine compatible FBX conversion...")
    blender_path = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    blend_file = os.path.join(base_dir, "smplx_template.blend")
    export_script = os.path.join(base_dir, "export_smplx_to_humanoid_fbx.py")
    result_dir = os.path.join(base_dir, "Result")
    
    cmd = [
        blender_path,
        "-b", blend_file,
        "-P", export_script,
        "--",
        out_npz,
        result_dir,
        "--engine", "unreal",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"FBX conversion success! Result path: {result_dir}")
    except Exception as e:
        print(f"FBX conversion failed: {e}")


if __name__ == "__main__":
    run_animation()