# 3D Gaussian Splatting from Scratch

> A scene is a cloud of millions of 3D Gaussians. Each one has a position, orientation, scale, opacity, and a colour that depends on viewing direction. Rasterise them, backprop through the rasterisation, done.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 13 (3D Vision & NeRF), Phase 1 Lesson 12 (Tensor Operations)
**Time:** ~90 minutes

## Learning Objectives

- Explain why 3DGS replaced NeRF as the production default
- State the six per-Gaussian parameters and how many floats each contributes
- Implement a 2D Gaussian splatting rasteriser with alpha compositing
- Use `nerfstudio` or `gsplat` to reconstruct a scene from photos

## The Problem

NeRF stores a scene as MLP weights. Training hours, rendering seconds, cannot edit. 3DGS stores explicit 3D Gaussians — rendering at 100+ fps, training minutes, edit by translating a subset.

## The Concept

### What a Gaussian carries

```
position    mu         (3,)
rotation    q          (4,)   unit quaternion
scale       s          (3,)   log-scales
opacity     alpha      (1,)   [0, 1]
SH coeffs   c_lm       (3 * (L+1)^2,)
```

### Rasterisation, not ray marching

```mermaid
flowchart LR
    SCENE["Millions of 3D Gaussians"] --> PROJ["Project to 2D"]
    PROJ --> TILES["Assign to tiles"]
    TILES --> SORT["Depth-sort per tile"]
    SORT --> ALPHA["Alpha-composite"]
    ALPHA --> PIX["Pixel colour"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

### Projection

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T
```

### Alpha compositing

```
C_pixel = sum_i alpha_i * T_i * c_i
T_i = prod_{j < i} (1 - alpha_j)
```

### Densification and pruning

Clone where gradient high and scale small. Split large Gaussians. Prune low-opacity ones.

## Build It

### Step 1: 2D Gaussian evaluation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def eval_2d_gaussian(means, covs, points):
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

### Step 2: 2D splatting rasteriser

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

### Step 3: Trainable 2D splat scene

```python
import math

class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

### Step 4: Fit to target image

```python
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)

target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

### Step 5: From 2D to 3D

Additions: quaternion rotation, 3x3 covariance, camera projection, spherical harmonics colour.

### Step 6: SH evaluation (degree 3)

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]
    return result
```

## Use It

```bash
pip install nerfstudio gsplat
ns-train splatfacto --data path/to/data
```

Export: `.ply`, `.splat`, glTF `KHR_gaussian_splatting`, OpenUSD.

## Ship It

- `outputs/prompt-3dgs-capture-planner.md`
- `outputs/skill-3dgs-export-router.md`

## Exercises

1. **(Easy)** Train 2D splats with num_splats in [16, 64, 256]; plot MSE.
2. **(Medium)** Add view-dependent colour with degree-2 harmonics.
3. **(Hard)** Train splatfacto on a 20-photo capture; export to glTF.

## Key Terms

## Further Reading

- [3DGS (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)
- [KHR_gaussian_splatting](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/22-3d-gaussian-splatting)
