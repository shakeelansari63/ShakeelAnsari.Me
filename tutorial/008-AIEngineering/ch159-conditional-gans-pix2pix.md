# Conditional GANs & Pix2Pix

> The first big unlock of 2014-2017 was controlling what a GAN makes. Attach a label, or an image, or a sentence. Pix2Pix did the image version and it still beats every generic text-to-image model on narrow image-to-image tasks.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## The Problem

An unconditional GAN samples arbitrary faces. Useful for a demo, useless in production. You want: *map a sketch to a photo*, *map a map to an aerial photo*, *map a daytime scene to nighttime*, *colorize a grayscale image*. In all of these, you are given an input image `x` and must output `y` with some semantic correspondence. There are many plausible `y`s per `x`. Mean-squared error flattens them into mush. An adversarial loss doesn't, because "looks real" is sharp.

Conditional GAN (Mirza & Osindero, 2014) adds a condition `c` as an input to both `G` and `D`. Pix2Pix (Isola et al., 2017) specialized this: condition is a full input image, generator is a U-Net, discriminator is a *patch-based* classifier (PatchGAN), and loss is adversarial + L1.

## The Concept

```mermaid
graph LR
    A[Condition c] --> B[Generator G<br>U-Net]
    C[Noise z] --> B
    B --> D[Fake Output ŷ]
    E[Real y, Condition c] --> F[Discriminator D<br>PatchGAN]
    D --> F
    A --> F
    F --> G[Patch-wise Real/Fake]
    D -- L1 --> H[||y - ŷ||₁]
```

**Conditional G.** `G(x, z) → y`. In Pix2Pix, `z` is dropout inside G.

**Conditional D.** `D(x, y) → [0, 1]`. Input is the *pair* (condition, output). D must judge whether `y` is consistent with `x`.

**U-Net generator.** Encoder-decoder with skip connections across the bottleneck. Critical for tasks where input and output share low-level structure.

**PatchGAN discriminator.** Instead of outputting a single real/fake score, D outputs an `N×N` grid where each cell judges a receptive field of ~70×70 pixels.

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

The L1 term stabilizes training and pushes G toward the known target. L1 gives sharper edges than L2 (medians, not means). `λ = 100` was the Pix2Pix default.

## CycleGAN — When You Don't Have Pairs

Pix2Pix needs paired `(x, y)` data. CycleGAN (Zhu et al., 2017) drops this requirement at the cost of an extra loss: the *cycle consistency* loss. Two generators `G: X → Y` and `F: Y → X`. Train them so `F(G(x)) ≈ x` and `G(F(y)) ≈ y`.

## Build It

`code/main.py` implements a tiny conditional GAN on 1-D data. The condition `c` is a class label (0 or 1).

### Step 1: append condition to both G and D inputs

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

### Step 2: train conditional

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

### Step 3: verify per-class output

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Pitfalls

- **Condition ignored.** G learns to marginalize, D never penalizes because condition signal is weak. Fix: condition D more aggressively, use projection discriminator.
- **L1 weight too low.** G drifts to arbitrary real-looking outputs, not faithful ones. Start λ≈100.
- **L1 weight too high.** G produces blurry outputs. Anneal down once training stabilizes.
- **Ground-truth leakage in D.** Concatenate `(x, y)` as D input, not just `y`.
- **Mode collapse per class.** Each class can collapse independently.

## Use It — 2026 Image-to-Image Tasks

| Task | Best approach |
|------|---------------|
| Sketch → photo, same domain, paired data | Pix2Pix / Pix2PixHD |
| Sketch → photo, unpaired | ControlNet with a Scribble conditioning model |
| Semantic seg → photo | SPADE / GauGAN2 or SD + ControlNet-Seg |
| Style transfer | Diffusion with IP-Adapter or LoRA |
| Depth → photo | ControlNet-Depth over Stable Diffusion |
| Super-resolution | Real-ESRGAN (GAN), ESRGAN-Plus, or SD-Upscale |
| Colorization | ColTran, diffusion-based colorizers, or Pix2Pix-color |

## Production Note — Pix2Pix as a Latency-Bound Baseline

| Path | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix wins on throughput in static batches. The modern play is often to ship a Pix2Pix-style distilled model for the narrow task and a diffusion fallback for tail inputs.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Conditional GAN | "GAN with labels" | G(z, c), D(x, c). Both networks see the condition. |
| Pix2Pix | "Image-to-image GAN" | Paired cGAN with U-Net G and PatchGAN D + L1 loss. |
| U-Net | "Encoder-decoder with skips" | Symmetric conv network; skips preserve high-freq. |
| PatchGAN | "Local-realism classifier" | D outputs per-patch score instead of global score. |
| CycleGAN | "Unpaired image translation" | Two G's + cycle-consistency loss; no paired data. |
| SPADE | "GauGAN" | Normalizes intermediate activations with the semantic map. |
| FiLM | "Feature-wise linear modulation" | Per-feature affine transform from the condition. |

## Exercises

1. **Easy.** Modify `code/main.py` to add a third class. Confirm G still maps each class's noise to the correct mode.
2. **Medium.** Replace L1 with a perceptual-style loss in the 1-D setting. Does it change sharpness of the conditional distribution?
3. **Hard.** Sketch a CycleGAN in the 1-D setting: two distributions, two generators, cycle loss.

## Further Reading

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585)
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/08-generative-ai/04-conditional-gans-pix2pix)
