# Diffusion Models — DDPM from Scratch

> Ho, Jain, Abbeel (2020) gave the field a recipe it could not quit. Destroy the data with noise over a thousand small steps. Train one neural net to predict the noise. Reverse the process at inference. Today every mainstream image, video, 3D, and music model runs on this loop, possibly with flow matching or consistency tricks on top.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## The Problem

You want a sampler for `p_data(x)`. GANs play a minimax game that often diverges. VAEs produce blurry samples from a Gaussian decoder. What you really want is a training objective that is (a) a single stable loss (no saddle point, no minimax), (b) a lower bound on `log p(x)` (so you have likelihoods), and (c) samples that match SOTA quality.

Sohl-Dickstein et al. (2015) had a theoretical answer: define a Markov chain `q(x_t | x_{t-1})` that gradually adds Gaussian noise, and train a reverse chain `p_θ(x_{t-1} | x_t)` to denoise. Ho, Jain, Abbeel (2020) showed the loss could be simplified to one line — predict the noise.

## The Concept

```mermaid
graph LR
    A[x₀] --> B[x₁]
    B --> C[...]
    C --> D[x_T]
    D --> E[ε_θ]
    E --> F[x_{T-1}]
    F --> G[...]
    G --> H[x̂₀]
    A -- forward q --> D
    D -- reverse p_θ --> H
```

**Forward process `q`.** Add Gaussian noise in `T` small steps. The closed form:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

where `α̅_t = ∏_{s=1..t} (1 - β_s)` for a schedule of `β_t`. Pick `β_t` from 1e-4 to 0.02 linearly over T=1000 steps and `x_T` is approximately `N(0, I)`.

**Reverse process `p_θ`.** Learn a neural net `ε_θ(x_t, t)` that predicts the noise that was added.

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

**Training loss.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Sample `x_0` from data, pick a random `t`, sample `ε ~ N(0, I)`, compute the noisy `x_t` in one shot via the closed form, and regress on the noise. One loss, no minimax, no KL, no reparameterization tricks.

**Sampling.** Start `x_T ~ N(0, I)`. Iterate the reverse step from `t = T` to `1`. Done.

## Build It

`code/main.py` implements a 1-D DDPM. Data is a two-mode mixture. The "net" is a tiny MLP that takes `(x_t, t)` and outputs predicted noise.

### Step 1: the forward schedule (closed form)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Step 2: sample `x_t` in one shot

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Step 3: one training step

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Step 4: reverse sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

## Time Conditioning

The net needs to know which timestep it is denoising. Two standard options:

- **Sinusoidal embedding.** Like Transformer positional encoding.
- **FiLM / group-norm conditioning.** Project embedding to per-channel scale/bias at each block.

```python
def sin_embed(t, T, dim=8):
    out = []
    half = dim // 2
    for i in range(half):
        freq = 1.0 / (10000 ** (i / max(half - 1, 1)))
        out.append(math.sin(t * freq))
        out.append(math.cos(t * freq))
    return out[:dim]
```

## Pitfalls

- **Schedule matters a lot.** Linear `β` is the DDPM default but cosine schedule gives better FID.
- **Timestep embedding is fragile.** Always use a proper embedding, not raw `t`.
- **V-prediction vs ε-prediction.** V-prediction (`v = α·ε - σ·x`) is more stable; SDXL, SD3, and Flux use it.
- **Classifier-free guidance.** At inference, compute both conditional and unconditional `ε`, then `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` with `w ≈ 3-7`.
- **1000 steps is a lot.** Production uses DDIM (20-50 steps), DPM-Solver (10-20 steps), or distillation (1-4 steps).

## Use It

| Role | Typical stack in 2026 |
|------|-----------------------|
| Image pixel-space diffusion (small, toy) | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net or DiT |
| Video latent diffusion | Spatiotemporal DiT (Sora, Veo, WAN) |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science (molecules, proteins, physics) | Equivariant diffusion (EDM, RFdiffusion) |

## Production Note

The DDPM paper runs T=1000 reverse steps. Nobody ships that in production.

1. **Faster sampler, same model.** DDIM (20-50 steps), DPM-Solver++ (10-20), UniPC (8-16). Cuts latency 20-50×.
2. **Distillation.** Train a student to match the teacher in fewer steps: Progressive Distillation, Consistency Models, LCM, SDXL-Turbo.
3. **Caching and compilation.** `torch.compile(unet, mode="reduce-overhead")`, TensorRT-LLM, `xformers`/SDPA attention, bf16 weights. Cuts per-step latency ~2×.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Forward process | "Adding noise" | Fixed Markov chain `q(x_t \| x_{t-1})` that destroys the data. |
| Reverse process | "Denoising" | Learned chain `p_θ(x_{t-1} \| x_t)` that reconstructs the data. |
| β schedule | "The noise ladder" | Per-step variance; linear, cosine, or sigmoid. |
| α̅ | "Alpha bar" | Cumulative product `∏(1 - β)`; gives closed-form `x_t` from `x_0`. |
| Simple loss | "MSE on noise" | `\|\|ε - ε_θ(x_t, t)\|\|²`; all variational derivations collapse to this. |
| ε-prediction | "Predict noise" | Output is the noise added; standard DDPM. |
| V-prediction | "Predict velocity" | Output is `α·ε - σ·x`; better conditioning across t. |
| DDPM | "The paper" | Ho et al. 2020; linear β, 1000 steps, U-Net. |
| DDIM | "Deterministic sampler" | Non-Markov sampler, 20-50 steps, same training objective. |
| Classifier-free guidance | "CFG" | Mix conditional and unconditional noise predictions to amplify conditioning. |

## Exercises

1. **Easy.** Change T from 40 to 10 in `code/main.py`. How does sample quality degrade?
2. **Medium.** Switch from ε-prediction to v-prediction. Re-derive the reverse step. Compare final sample quality.
3. **Hard.** Add classifier-free guidance. Condition on a class label `c ∈ {0, 1}`, drop it 10% of the time during training, and at sampling time use `ε = (1+w)·ε_cond - w·ε_uncond`.

## Further Reading

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672)
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/08-generative-ai/06-diffusion-ddpm-from-scratch)
