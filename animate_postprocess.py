import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os

class Particle:
    def __init__(self, pos, mass):
        self.pos = np.array(pos, dtype=float)
        self.prev_pos = np.copy(self.pos)
        self.vel = np.zeros(3)
        self.inv_mass = 0.0 if mass == 0.0 else 1.0 / mass

class DistanceConstraint:
    def __init__(self, p1_idx, p2_idx, length, compliance=0.0):
        self.p1_idx, self.p2_idx = p1_idx, p2_idx
        self.length = length
        self.compliance = compliance

    def solve(self, particles, dt):
        p1, p2 = particles[self.p1_idx], particles[self.p2_idx]
        w_sum = p1.inv_mass + p2.inv_mass
        if w_sum == 0.0: return
        dir_vec = p1.pos - p2.pos
        cur_len = np.linalg.norm(dir_vec)
        if cur_len == 0.0: return
        C = cur_len - self.length
        alpha = self.compliance / (dt * dt)
        delta_lambda = -C / (w_sum + alpha)
        move = delta_lambda * (dir_vec / cur_len)
        p1.pos += p1.inv_mass * move
        p2.pos -= p2.inv_mass * move

class TrackingConstraint:
    def __init__(self, p_idx, compliance):
        self.p_idx = p_idx
        self.target_pos = np.zeros(3)
        self.compliance = compliance

    def solve(self, particles, dt):
        p = particles[self.p_idx]
        if p.inv_mass == 0.0: return
        diff = p.pos - self.target_pos
        C_len = np.linalg.norm(diff)
        if C_len == 0.0: return
        alpha = self.compliance / (dt * dt)
        delta_lambda = -C_len / (p.inv_mass + alpha)
        p.pos += p.inv_mass * delta_lambda * (diff / C_len)

class OneEuroFilterVectorized:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0.copy()
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0: return x
        dx = (x - self.x_prev) / t_e
        alpha_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        alpha = self.smoothing_factor(t_e, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat.copy(), dx_hat.copy(), t
        return x_hat

def get_rotation_between_vectors(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    cross = np.cross(v1_u, v2_u)
    dot = np.dot(v1_u, v2_u)
    s = np.linalg.norm(cross)
    if s < 1e-8:
        return R.identity()
    axis = cross / s
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis * angle)

def process_full_tennis_pipeline(arm_positions, global_orient_3, local_pose_45, fps=30, is_swing_segment=True):
    T = local_pose_45.shape[0]
    dt = 1.0 / fps
    substeps = 10
    sub_dt = dt / substeps
    timestamps = np.arange(T) * dt

    particles = [
        Particle(arm_positions[0, 0], mass=0.0),
        Particle(arm_positions[0, 1], mass=2.5),
        Particle(arm_positions[0, 2], mass=1.8),
        Particle(arm_positions[0, 2] + np.array([0, -0.6, 0]), mass=1.2)
    ]
    
    l_upper = np.linalg.norm(arm_positions[0, 0] - arm_positions[0, 1])
    l_lower = np.linalg.norm(arm_positions[0, 1] - arm_positions[0, 2])
    l_racket = 0.6
    
    constraints = [
        DistanceConstraint(0, 1, l_upper, 0.0),
        DistanceConstraint(1, 2, l_lower, 0.0),
        DistanceConstraint(2, 3, l_racket, 0.0) 
    ]
    
    track_elbow = TrackingConstraint(1, compliance=0.002)
    track_wrist = TrackingConstraint(2, compliance=0.005)

    local_15x3 = local_pose_45.reshape(T, 15, 3)
    init_euler = R.from_rotvec(local_15x3[0]).as_euler('XYZ', degrees=False)
    one_euro = OneEuroFilterVectorized(0.0, init_euler, min_cutoff=0.5, beta=0.05)
    
    all_euler = R.from_rotvec(local_15x3.reshape(-1, 3)).as_euler('XYZ', degrees=False).reshape(T, 15, 3)
    all_euler_unwrapped = np.unwrap(all_euler, axis=0)

    final_global_orient_3 = np.zeros_like(global_orient_3)
    final_local_pose_45 = np.zeros_like(local_pose_45)

    target_grip_quat = R.from_rotvec(np.ones((15, 3)) * 0.2).as_quat() 

    print("🚀 통합 파이프라인 시작...")

    for i in range(T):
        particles[0].pos = particles[0].prev_pos = arm_positions[i, 0]
        track_elbow.target_pos = arm_positions[i, 1]
        track_wrist.target_pos = arm_positions[i, 2]
        target_racket_pos = track_wrist.target_pos + np.array([0, -0.6, 0])

        for _ in range(substeps):
            for p in particles:
                if p.inv_mass > 0:
                    p.vel += np.array([0, -9.81, 0]) * sub_dt
                    p.prev_pos = np.copy(p.pos)
                    p.pos += p.vel * sub_dt
            for _ in range(3):
                for c in constraints: c.solve(particles, sub_dt)
                track_elbow.solve(particles, sub_dt)
                track_wrist.solve(particles, sub_dt)
            for p in particles:
                if p.inv_mass > 0: p.vel = (p.pos - p.prev_pos) / sub_dt

        physics_wrist_pos = particles[2].pos
        physics_racket_pos = particles[3].pos

        orig_dir = target_racket_pos - track_wrist.target_pos
        physics_dir = physics_racket_pos - physics_wrist_pos
        
        delta_rot = get_rotation_between_vectors(orig_dir, physics_dir)
        orig_wrist_rot = R.from_rotvec(global_orient_3[i])
        new_wrist_rot = delta_rot * orig_wrist_rot
        final_global_orient_3[i] = new_wrist_rot.as_rotvec()

        smoothed_euler = one_euro(timestamps[i], all_euler_unwrapped[i])
        smoothed_quats = R.from_euler('XYZ', smoothed_euler.reshape(-1, 3), degrees=False).as_quat()

        refined_quats = np.zeros_like(smoothed_quats)
        for j in range(15):
            if is_swing_segment:
                key_quats = R.from_quat([smoothed_quats[j], target_grip_quat[j]])
                refined_quats[j] = Slerp([0, 1], key_quats)([0.8])[0].as_quat()
            else:
                refined_quats[j] = smoothed_quats[j]

        final_local_pose_45[i] = R.from_quat(refined_quats).as_rotvec().reshape(45)

    final_48d_pose = np.concatenate([final_global_orient_3, final_local_pose_45], axis=1)
    print("✅ 처리 완료!")
    return final_48d_pose

def run_animation():
    T_frames = 150
    fps = 30
    
    npz_path = "interhand_val.npz"
    local_pose_45 = np.zeros((T_frames, 45))
    
    if os.path.exists(npz_path):
        print(f"📦 실제 데이터셋 로드: {npz_path}")
        data = np.load(npz_path)
        hand_pose_all = data['hand_pose']
        length = min(T_frames, len(hand_pose_all))
        T_frames = length
        local_pose_45[:length] = hand_pose_all[:length]
        
        local_pose_45 += np.random.randn(*local_pose_45.shape) * 0.1
    else:
        print(f"⚠️ {npz_path} 없음. 가짜 흔들림(Jitter) 데이터를 임의 생성합니다.")
        t = np.linspace(0, 4 * np.pi, T_frames)
        for i in range(45):
            local_pose_45[:, i] = np.sin(t * (1 + i * 0.1)) * 0.3 + np.random.randn(T_frames) * 0.1

    dummy_arm_pos = np.zeros((T_frames, 3, 3)) 
    dummy_arm_pos[:, 1] = [0, -0.3, 0]
    dummy_arm_pos[:, 2] = [0, -0.6, 0]
    
    swing_t = np.linspace(0, 2 * np.pi, T_frames)
    dummy_arm_pos[:, 2, 0] = np.sin(swing_t) * 0.6
    dummy_arm_pos[:, 2, 1] = -0.6 + np.sin(swing_t * 2) * 0.2
    dummy_arm_pos[:, 2, 2] = np.cos(swing_t) * 0.5
    
    dummy_global = np.zeros((T_frames, 3))
    dummy_global[:, 0] = np.sin(swing_t) * 0.5
    
    final_48d_pose = process_full_tennis_pipeline(
        arm_positions=dummy_arm_pos,
        global_orient_3=dummy_global,
        local_pose_45=local_pose_45,
        fps=fps,
        is_swing_segment=True
    )
    
    try:
        from smplx import MANO
        print("💡 smplx 모듈이 감지되었습니다. 3D 관절 렌더링을 준비합니다...")
        device = torch.device("cpu")
        mano_layer = MANO(model_path="mano", is_rhand=True, use_pca=False).to(device)
        
        orig_global = torch.tensor(dummy_global, dtype=torch.float32)
        orig_local = torch.tensor(local_pose_45, dtype=torch.float32)
        with torch.no_grad():
            orig_output = mano_layer(global_orient=orig_global, hand_pose=orig_local)
            orig_joints = orig_output.joints.numpy()
            
        proc_global = torch.tensor(final_48d_pose[:, :3], dtype=torch.float32)
        proc_local = torch.tensor(final_48d_pose[:, 3:], dtype=torch.float32)
        with torch.no_grad():
            proc_output = mano_layer(global_orient=proc_global, hand_pose=proc_local)
            proc_joints = proc_output.joints.numpy()
            
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        ax1.set_title("Before")
        ax2.set_title("After")
        
        scat_orig = ax1.scatter([], [], [], c='r', s=20)
        scat_proc = ax2.scatter([], [], [], c='b', s=20)

        def set_axes(ax):
            ax.set_xlim([-0.8, 0.8])
            ax.set_ylim([-0.8, 0.8])
            ax.set_zlim([-0.8, 0.8])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        set_axes(ax1)
        set_axes(ax2)
        
        def update(frame):
            scat_orig._offsets3d = (
                orig_joints[frame, :, 0], 
                orig_joints[frame, :, 1], 
                orig_joints[frame, :, 2]
            )
            scat_proc._offsets3d = (
                proc_joints[frame, :, 0], 
                proc_joints[frame, :, 1], 
                proc_joints[frame, :, 2]
            )
            return scat_orig, scat_proc

        ani = animation.FuncAnimation(fig, update, frames=T_frames, interval=1000/fps, blit=False)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("smplx 모듈이 없어 3D 렌더링을 할 수 없습니다.")
        print("그래프 시각화 없이 내부 처리만 정상 완료되었습니다.")

if __name__ == "__main__":
    run_animation()