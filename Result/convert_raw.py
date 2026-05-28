import torch
import numpy as np
import sys

pt_path = r"c:\Users\user\Desktop\CG\캡스톤\cap_pipeline\front_4\smplx_merged_hamer_post.pt"
out_path = r"c:\Users\user\Desktop\CG\캡스톤\cap_pipeline\front_4\smplx_merged_hamer_post_raw.npz"

data = torch.load(pt_path, map_location='cpu')
params = data['smpl_params_global']

out_dict = {}
for k, v in params.items():
    val = v.numpy() if hasattr(v, 'numpy') else np.array(v)
    out_dict[k] = val

np.savez(out_path, **out_dict)
print(f"Saved raw npz to {out_path}")
