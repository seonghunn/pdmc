# Parallel Dual Marching Cube for Intersection-free Mesh

# Installation
Requirements: torch (must be compatible with CUDA version), trimesh
```
pip install pdmc
```

# Quick Start
You can effortlessly try the following command, which converts a sphere SDF into triangle mesh using different algorithms. The generated results are saved in `out/`.
```
python test/example.py
```

# Usage
All the functions share the same interfaces. Firstly you should build an iso-surface extractor as follows:
```
from pdmc import DMC
dmc = DMC(dtype=torch.float32).cuda() # or dtype=torch.float64
```
Then use its `forward` function to generate a single mesh:
```
verts, faces = dmc(sdf, isovalue=0.0, return_quads=False, normalize=True)
```

Input
* `sdf`: queries SDF values on the grid vertices (see the `test/example.py` for how to create the grid). The gradient will be back-propagated to the source that generates the SDF values. (**[N, N, N, 3]**)
* `normalize`: whether to normalize the output vertices, default=True. If set to **True**, the vertices are normalized to [0, 1]. When **False**, the vertices remain unnormalized as [0, dim-1], 
* `return_quads`: whether return quad meshes; If set to **True**, the function returns quad meshes (**[F, 4]**).

Output
* `verts`: mesh vertices within the range of [0, 1] or [0, dim-1]. (**[V, 3]**)
* `faces`: mesh face indices (starting from 0). (**[F, 3]**).



# Reference