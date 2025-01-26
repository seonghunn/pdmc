import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
from pdmc import DMC

# Load SDF from file
sdf_file = "./sdf_small.npy"
if not os.path.exists(sdf_file):
    raise FileNotFoundError(f"File {sdf_file} not found!")
sdf_data = np.load(sdf_file)
# Convert SDF to PyTorch tensor
device = "cuda:0"
sdf = torch.tensor(sdf_data, dtype=torch.float32, device=device)
sdf = sdf.requires_grad_(True)  # Enable gradients for backward pass

os.makedirs("out", exist_ok=True)

# Create grid deformation
deform = torch.nn.Parameter(
    torch.rand(
        (sdf.shape[0], sdf.shape[1], sdf.shape[2], 3),
        dtype=torch.float32,
        device=device,
    ),
    requires_grad=True,
)

# Initialize iso-surface extractors
diffdmc = DMC(dtype=torch.float32)

verts, faces, tris = diffdmc(sdf, isovalue=0)
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=tris.cpu().numpy(), process=False)
mesh.export("out/out.obj")

print("forward results saved to out/")
