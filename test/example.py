import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
from pdmc import DMC

# define a sphere SDF
class SphereSDF:
    def __init__(self, center, radius, margin):
        self.center = center
        self.radius = radius
        self.aabb = torch.stack([center - radius - margin, center + radius + margin], dim=-1)

    def __call__(self, points):
        return torch.norm(points - self.center, dim=-1) - self.radius

device = "cuda:0"
os.makedirs("out", exist_ok=True)

# define a sphere
s_x = 0.
s_y = 0.
s_z = 0.
radius = 0.5
sphere = SphereSDF(torch.tensor([s_x, s_y, s_z]), radius, 1/64)


# create a grid
dimX, dimY, dimZ = 64, 64, 64
grids = torch.stack(
    torch.meshgrid(
        torch.linspace(0, 1, dimX),
        torch.linspace(0, 1, dimY),
        torch.linspace(0, 1, dimZ),
        indexing="ij",
    ),
    dim=-1,
)
grids[..., 0] = (
    grids[..., 0] * (sphere.aabb[0, 1] - sphere.aabb[0, 0]) + sphere.aabb[0, 0]
)
grids[..., 1] = (
    grids[..., 1] * (sphere.aabb[1, 1] - sphere.aabb[1, 0]) + sphere.aabb[1, 0]
)
grids[..., 2] = (
    grids[..., 2] * (sphere.aabb[2, 1] - sphere.aabb[2, 0]) + sphere.aabb[2, 0]
)

# query the SDF input
sdf = sphere(grids).to(device)

os.makedirs("out", exist_ok=True)

dmc = DMC(dtype=torch.float32).to(device)

verts, faces = dmc(sdf, isovalue=0.0, return_quads=False, normalize=True)
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(), process=False)
mesh.export("out/sphere.obj")

print("results saved to out/")
