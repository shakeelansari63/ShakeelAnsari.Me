# Quantization: Making Models Fit

> A 70B model in FP16 needs 140GB. Two A100s just for weights. Quantize to FP8: one 80GB GPU. INT4: a MacBook.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lessons 01-10 (LLMs from Scratch)
**Time:** ~120 minutes

## Learning Objectives

- Implement symmetric and asymmetric quantization from FP16 to INT8 and INT4, including per-tensor and per-channel scaling
- Calculate the memory savings from quantization and determine which precision fits a given GPU's VRAM
- Explain the difference between post-training quantization (PTQ) and quantization-aware training (QAT)
- Apply GPTQ or AWQ to quantize a real model and measure the accuracy-memory tradeoff on a benchmark

## The Problem

Llama 3 70B has 70 billion parameters. Each parameter is a 16-bit floating point number. That is 140GB. A single A100 has 80GB of VRAM. You cannot even load the weights.

But 16 bits per parameter is wasteful. Most weights in a neural network cluster near zero. The full dynamic range of FP16 is almost entirely unused. 95% of values fall between -0.1 and +0.1. You are burning 16 bits to represent values that could fit in 4.

Quantization replaces high-precision numbers with lower-precision ones. FP16 to INT4 cuts memory to a quarter. That 140GB model becomes 35GB. It fits on a single consumer GPU.

The cost is accuracy. A well-quantized INT4 model retains 95-99% of the original's quality on most benchmarks.

## The Concept

### Number Formats

```
FP32:  [1 sign] [8 exponent] [23 mantissa]  = 32 bits
FP16:  [1 sign] [5 exponent] [10 mantissa]  = 16 bits
BF16:  [1 sign] [8 exponent] [7  mantissa]  = 16 bits
FP8:   [1 sign] [4 exponent] [3  mantissa]  = 8  bits (E4M3)
INT8:  [1 sign] [7 value]                   = 8  bits
INT4:  [1 sign] [3 value]                   = 4  bits (16 levels)
```

- **FP32**: Full precision, used for training accumulation
- **FP16**: Halved bits, needs loss scaling for training
- **BF16**: Same range as FP32, less precision -- modern training default
- **FP8**: H100 native, 30-50% speedup over FP16
- **INT8**: Integer format, faster arithmetic, needs scale factor
- **INT4**: 16 possible values, requires GPTQ/AWQ for quality

### How Quantization Works

```
scale = max(abs(tensor)) / max_int_value
quantized = round(tensor / scale)
reconstructed = quantized * scale
```

**Per-tensor vs per-channel.** Per-tensor uses one scale for the entire matrix. Simple but lossy when values vary across channels. Per-channel uses one scale per output channel -- more overhead but dramatically better quality.

**Asymmetric quantization** adds a zero-point offset for distributions not centered at zero: `quantized = round(tensor / scale) + zero_point`.

### Sensitivity Hierarchy

| Component | Sensitivity | Best Precision |
|-----------|-------------|----------------|
| Weights | Most robust | INT4 works well |
| Activations | Moderate | INT8 with care |
| KV cache | High | FP8 or INT8 |
| Attention logits | Most sensitive | Keep in FP16 |

### PTQ vs QAT

| Aspect | PTQ | QAT |
|--------|-----|-----|
| Cost | Minutes to hours | Full training run |
| Quality at INT8 | Excellent | Excellent |
| Quality at INT4 | Good with GPTQ/AWQ | Better |
| Calibration data | 128-1024 examples | Full training dataset |

### GPTQ, AWQ, GGUF

**GPTQ** quantizes weights one layer at a time using the Hessian to measure weight importance. Weights that matter more get quantized more carefully.

**AWQ** identifies the ~1% of weights that are disproportionately important (those multiplying with large activations) and scales them up before quantization.

**GGUF** is the llama.cpp format supporting mixed quantization across layers.

| Model | Format | Size | Perplexity | MMLU | Tokens/sec |
|-------|--------|------|------------|------|------------|
| Llama 3 70B | FP16 | 140GB | 3.12 | 79.5% | 38 |
| Llama 3 70B | FP8 | 70GB | 3.14 | 79.3% | 55 |
| Llama 3 70B | GPTQ INT4 | 35GB | 4.32 | 77.8% | 72 |

## Build It

### Step 1: Number Format Representations

```python
import numpy as np

def float_to_fp32_bits(value):
    bits = np.float32(value).view(np.uint32)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '023b'),
            "value": float(value), "actual_exponent": int(exponent) - 127}

def float_to_fp16_bits(value):
    fp16 = np.float16(value)
    bits = fp16.view(np.uint16)
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '05b'),
            "mantissa_bits": format(int(mantissa), '010b'),
            "value": float(fp16), "actual_exponent": int(exponent) - 15}

def float_to_bf16_bits(value):
    fp32_bits = np.float32(value).view(np.uint32)
    bf16_bits = (fp32_bits >> 16).astype(np.uint16)
    sign = (bf16_bits >> 15) & 1
    exponent = (bf16_bits >> 7) & 0xFF
    mantissa = bf16_bits & 0x7F
    reconstructed = np.uint32(bf16_bits.astype(np.uint32) << 16).view(np.float32)
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '07b'),
            "value": float(reconstructed), "actual_exponent": int(exponent) - 127}
```

### Step 2: Symmetric and Asymmetric Quantization

```python
def quantize_symmetric(tensor, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int32), 1.0
    scale = abs_max / qmax
    quantized = np.clip(np.round(tensor / scale), qmin, qmax).astype(np.int32)
    return quantized, float(scale)

def dequantize_symmetric(quantized, scale):
    return quantized.astype(np.float64) * scale

def quantize_per_channel(tensor, num_bits=8, axis=0):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    if axis == 0:
        abs_max = np.max(np.abs(tensor), axis=1, keepdims=True)
    else:
        abs_max = np.max(np.abs(tensor), axis=0, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = abs_max / qmax
    quantized = np.clip(np.round(tensor / scales), qmin, qmax).astype(np.int32)
    return quantized, scales.squeeze()

def dequantize_per_channel(quantized, scales, axis=0):
    if axis == 0:
        return quantized.astype(np.float64) * scales.reshape(-1, 1)
    else:
        return quantized.astype(np.float64) * scales.reshape(1, -1)

def quantize_asymmetric(tensor, num_bits=8):
    qmin, qmax = 0, 2 ** num_bits - 1
    t_min, t_max = np.min(tensor), np.max(tensor)
    if t_max == t_min:
        return np.zeros_like(tensor, dtype=np.int32), 1.0, 0
    scale = (t_max - t_min) / (qmax - qmin)
    zero_point = int(np.round(qmin - t_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    quantized = np.clip(np.round(tensor / scale + zero_point), qmin, qmax).astype(np.int32)
    return quantized, float(scale), int(zero_point)

def dequantize_asymmetric(quantized, scale, zero_point):
    return (quantized.astype(np.float64) - zero_point) * scale
```

### Step 3: Quality Measurement

```python
def quantization_error(original, reconstructed):
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(np.abs(diff)))
    signal_power = float(np.mean(original ** 2))
    snr_db = 10 * np.log10(signal_power / max(mse, 1e-20))
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    norm_orig = np.linalg.norm(orig_flat)
    norm_recon = np.linalg.norm(recon_flat)
    cosine_sim = float(np.dot(orig_flat, recon_flat) / (norm_orig * norm_recon)) if norm_orig > 0 and norm_recon > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "max_error": max_error,
            "snr_db": float(snr_db), "cosine_similarity": cosine_sim}


def compare_quantization_methods(tensor, num_bits=8):
    q_pt, s_pt = quantize_symmetric(tensor, num_bits)
    recon_pt = dequantize_symmetric(q_pt, s_pt)
    err_pt = quantization_error(tensor, recon_pt)

    q_pc, s_pc = quantize_per_channel(tensor, num_bits, axis=0)
    recon_pc = dequantize_per_channel(q_pc, s_pc, axis=0)
    err_pc = quantization_error(tensor, recon_pc)

    q_asym, s_asym, zp = quantize_asymmetric(tensor, num_bits)
    recon_asym = dequantize_asymmetric(q_asym, s_asym, zp)
    err_asym = quantization_error(tensor, recon_asym)

    print(f"\n  Quantization Comparison ({num_bits}-bit):")
    for name, err in [("Per-tensor sym", err_pt), ("Per-channel sym", err_pc), ("Asymmetric", err_asym)]:
        print(f"  {name:<20} MSE: {err['mse']:.8f}  SNR: {err['snr_db']:.2f}dB  Cos: {err['cosine_similarity']:.6f}")
```

### Step 4: Bit-Width Sweep

```python
def bit_width_sweep(tensor):
    print(f"\n  Bit-Width Sweep:")
    for bits in [2, 3, 4, 8, 16]:
        q, s = quantize_per_channel(tensor, bits, axis=0)
        recon = dequantize_per_channel(q, s, axis=0)
        err = quantization_error(tensor, recon)
        compression = 32.0 / bits
        print(f"  {bits}b -> {2**bits:>4} levels  MSE: {err['mse']:.8f}  SNR: {err['snr_db']:.2f}dB  {compression:.1f}x")
```

### Step 5: Sensitivity Experiment

```python
def simulate_transformer_layer(input_data, weights):
    hidden = input_data @ weights["qkv"]
    d_model = weights["qkv"].shape[1] // 3
    q, k, v = hidden[:, :, :d_model], hidden[:, :, d_model:2*d_model], hidden[:, :, 2*d_model:]
    attn_scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_model)
    attn_max = np.max(attn_scores, axis=-1, keepdims=True)
    attn_weights = np.exp(attn_scores - attn_max) / np.sum(np.exp(attn_scores - attn_max), axis=-1, keepdims=True)
    attn_output = attn_weights @ v
    output = attn_output @ weights["out"]
    return output, {"q": q, "k": k, "v": v, "attn_scores": attn_scores}


def sensitivity_experiment(batch_size=2, seq_len=16, d_model=64, num_bits=8):
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, d_model) * 0.1
    weights = {"qkv": np.random.randn(d_model, 3*d_model) * np.sqrt(2/d_model),
               "out": np.random.randn(d_model, d_model) * np.sqrt(2/d_model)}

    baseline_output, baseline_internals = simulate_transformer_layer(input_data, weights)

    q_qkv, s_qkv = quantize_per_channel(weights["qkv"], num_bits, axis=0)
    q_out, s_out = quantize_per_channel(weights["out"], num_bits, axis=0)
    quant_w = {"qkv": dequantize_per_channel(q_qkv, s_qkv, axis=0),
               "out": dequantize_per_channel(q_out, s_out, axis=0)}
    w_err = quantization_error(baseline_output, simulate_transformer_layer(input_data, quant_w)[0])

    _, fresh = simulate_transformer_layer(input_data, weights)
    q_act, s_act = quantize_per_channel(fresh["attn_output"].reshape(-1, d_model), num_bits, axis=0)
    q_k, s_k = quantize_per_channel(fresh["k"].reshape(-1, d_model), num_bits, axis=0)
    q_v, s_v = quantize_per_channel(fresh["v"].reshape(-1, d_model), num_bits, axis=0)

    print(f"\n  Sensitivity ({num_bits}-bit):")
    for name, err in [("Weights", w_err)]:
        print(f"  {name:<20} MSE: {err['mse']:.8f}  SNR: {err['snr_db']:.2f}dB")
```

### Step 6: Simulated GPTQ

```python
def simulated_gptq(weight_matrix, calibration_inputs, num_bits=4):
    n_in, n_out = weight_matrix.shape
    qmin, qmax = -(2**(num_bits-1)), 2**(num_bits-1) - 1

    H = np.zeros((n_in, n_in))
    for x in calibration_inputs:
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        for row in range(x.shape[0]):
            xi = x[row].reshape(-1, 1)
            H += xi @ xi.T
    H /= len(calibration_inputs)
    H += np.eye(n_in) * 1e-4

    quantized = np.zeros_like(weight_matrix, dtype=np.int32)
    scales = np.zeros(n_out)
    W = weight_matrix.copy()

    for col in range(n_out):
        w_col = W[:, col]
        abs_max = np.max(np.abs(w_col))
        if abs_max == 0:
            scales[col] = 1.0
            continue
        scale = abs_max / qmax
        scales[col] = scale
        q_col = np.clip(np.round(w_col / scale), qmin, qmax).astype(np.int32)
        quantized[:, col] = q_col
        quant_error = w_col - q_col * scale
        if col < n_out - 1:
            for next_col in range(col + 1, min(col + 4, n_out)):
                W[:, next_col] += quant_error * 0.1

    return quantized, scales
```

### Step 7: AWQ Simulation

```python
def simulated_awq(weight_matrix, calibration_inputs, num_bits=4, salient_fraction=0.01):
    n_in, n_out = weight_matrix.shape
    activation_magnitudes = np.zeros(n_in)
    for x in calibration_inputs:
        activation_magnitudes += np.mean(np.abs(x), axis=0) if x.ndim > 1 else np.abs(x)
    activation_magnitudes /= len(calibration_inputs)

    n_salient = max(1, int(n_in * salient_fraction))
    salient_indices = np.argsort(activation_magnitudes)[-n_salient:]

    scale_factors = np.ones(n_in)
    for idx in salient_indices:
        col_max = np.max(np.abs(weight_matrix[idx, :]))
        if col_max > 0:
            scale_factors[idx] = min(4.0, np.mean(np.abs(weight_matrix)) / (col_max + 1e-8))

    scaled_weights = weight_matrix * scale_factors.reshape(-1, 1)
    q, s = quantize_per_channel(scaled_weights, num_bits, axis=0)
    recon = dequantize_per_channel(q, s, axis=0) / scale_factors.reshape(-1, 1)
    return recon, quantization_error(weight_matrix, recon)
```

### Memory Calculator

```python
def memory_calculator(num_params_billions, bits_per_param):
    total_gb = num_params_billions * 1e9 * (bits_per_param / 8) / (1024 ** 3)
    return total_gb

def print_memory_table():
    print("\n  Memory Requirements:")
    for name, params in [("7B", 7), ("13B", 13), ("70B", 70), ("405B", 405)]:
        fp16 = memory_calculator(params, 16)
        int4 = memory_calculator(params, 4)
        print(f"  {name:<8} FP16: {fp16:>6.1f}G  INT4: {int4:>6.1f}G")
```

## Use It

```python
# AutoGPTQ
# model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

# AutoAWQ
# model = AutoAWQForCausalLM.from_pretrained(model_id)
# model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})

# vLLM serving with quantized model
# vllm serve model-awq --quantization awq --dtype half

# GGUF with llama.cpp
# python convert_hf_to_gguf.py model --outtype q4_k_m --outfile model.gguf
```

## Ship It

This lesson produces `outputs/skill-quantization.md` -- a decision framework for choosing the right quantization strategy.

## Exercises

1. Implement group quantization with group sizes of 32, 64, 128, 256.
2. Build a mixed-precision quantizer: first/last layers at INT8, middle layers at INT4.
3. Implement the straight-through estimator (STE) for QAT on a simple network.
4. Build an outlier-aware quantizer (LLM.int8() style): detect outlier channels, keep in FP16.
5. Implement a quantization quality dashboard with weight distribution histogram, error distribution, per-channel scales.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| FP16 | "Half precision" | 16-bit float, 5 exponent + 10 mantissa bits, standard inference format |
| BF16 | "Brain float" | 16-bit float, 8 exponent bits (same range as FP32), training default |
| INT8 | "8-bit integer" | 256 uniformly spaced values from -128 to 127 |
| INT4 | "4-bit integer" | 16 levels, requires GPTQ/AWQ for quality |
| GPTQ | "Hessian method" | Post-training quantization using second-order info per layer |
| AWQ | "Activation-aware" | Scales salient weights before quantization |
| PTQ | "Post-training quantization" | Quantize after training, fast but limited at extreme compression |
| QAT | "Quantization-aware training" | Insert fake quantization during training, better at INT4/INT2 |

## Further Reading

- [Frantar et al., 2022 -- "GPTQ"](https://arxiv.org/abs/2210.17323)
- [Lin et al., 2023 -- "AWQ"](https://arxiv.org/abs/2306.00978)
- [Dettmers et al., 2022 -- "LLM.int8()"](https://arxiv.org/abs/2208.07339)
- [Xiao et al., 2023 -- "SmoothQuant"](https://arxiv.org/abs/2211.10438)
