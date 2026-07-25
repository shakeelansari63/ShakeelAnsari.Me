# Real-Time Vision — Edge Deployment

> Edge inference is the discipline of getting a 90-accuracy model to run at 30 fps on a device with 2 GB of RAM. Every percentage point of accuracy is traded against milliseconds of latency.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 04 (Image Classification), Phase 10 Lesson 11 (Quantization)
**Time:** ~75 minutes

## Learning Objectives

- Measure latency, peak memory, and throughput correctly
- Quantise to INT8 with PyTorch PTQ and verify accuracy loss < 1%
- Export to ONNX and compile with ONNX Runtime or TensorRT
- Pick the right edge architecture for a budget

## The Problem

A training model is 100M params, 10 GFLOPs, 2 GB VRAM. Edge devices have 100x less.

## The Concept

### Three budgets

- Latency: p50, p95, p99
- Peak memory: max, not average
- Power: mJ per inference

### Measurement rules

1. Warm up 5-10 passes
2. Synchronise CUDA
3. Fix input size to production resolution

### FLOPs as proxy

Use FLOPs for architecture search, on-device latency for deployment decisions.

### Quantisation

```
Types: dynamic (easiest), static PTQ (95% benefit), QAT (best accuracy)
```

### Edge architecture picker

| Budget | Model |
|--------|-------|
| < 3M params | MobileNetV3-Small |
| 3-10M | EfficientNet-Lite-B0 |
| 10-20M | ConvNeXt-Tiny |
| 20-30M | MobileViT-S |

## Build It

### Step 1: Measure latency

```python
import time
import torch

def measure_latency(model, input_shape, device="cpu", warmup=10, iters=50):
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "mean_ms": sum(times) / len(times),
    }
```

### Step 2: Parameter and FLOP counts

```python
def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def flops_estimate(model, input_shape):
    total = 0
    def conv_hook(m, inp, out):
        nonlocal total
        c_out, c_in, kh, kw = m.weight.shape
        h, w = out.shape[-2:]
        total += 2 * c_in * c_out * kh * kw * h * w
    def linear_hook(m, inp, out):
        nonlocal total
        total += 2 * m.in_features * m.out_features
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_shape))
    for h in hooks:
        h.remove()
    return total
```

### Step 3: Post-training static quantisation

```python
def quantise_ptq(model, calibration_loader, backend="x86"):
    import torch.ao.quantization as tq
    model = model.eval().cpu()
    model.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)
    tq.convert(model, inplace=True)
    return model
```

### Step 4: Export to ONNX

```python
def export_onnx(model, sample_input, path="model.onnx"):
    model = model.eval()
    torch.onnx.export(
        model, sample_input, path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return path
```

### Step 5: Benchmark regimes

```python
from torchvision.models import mobilenet_v3_small

def compare_regimes():
    model = mobilenet_v3_small(weights=None, num_classes=10)
    params = parameter_count(model)
    flops = flops_estimate(model, (1, 3, 224, 224))
    lat_fp32 = measure_latency(model, (1, 3, 224, 224), device="cpu")
    print(f"FP32 MobileNetV3-Small: {params:,} params  {flops/1e9:.2f} GFLOPs  "
          f"p50={lat_fp32['p50_ms']:.2f}ms")
```

## Use It

Production paths:
- Web/serverless: PyTorch -> ONNX -> ONNX Runtime
- NVIDIA edge: PyTorch -> ONNX -> TensorRT
- Mobile: PyTorch -> ONNX -> Core ML (iOS) or TFLite (Android)

## Ship It

- `outputs/prompt-edge-deployment-planner.md`
- `outputs/skill-latency-profiler.md`

## Exercises

1. **(Easy)** Measure p50 for resnet18, mobilenet_v3_small, efficientnet_v2_s, convnext_tiny.
2. **(Medium)** Apply PTQ to mobilenet_v3_small; report FP32 vs INT8 latency and accuracy.
3. **(Hard)** Export convnext_tiny to ONNX; compare vs PyTorch eager.

## Key Terms

## Further Reading

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244)
- [TensorRT Optimization Guide](https://developer.nvidia.com/blog/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users/)
- [ONNX Runtime docs](https://onnxruntime.ai/docs/)

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/15-real-time-edge)
