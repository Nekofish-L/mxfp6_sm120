<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mxfp6-wordmark-dark.png">
    <img src="docs/assets/mxfp6-wordmark-light.png" alt="MXFP6 SM120" width="520">
  </picture>
</p>

# Native MXFP6 and MXFP8 for NVIDIA SM120

`mxfp6-sm120` is a PyTorch CUDA kernel library for NVIDIA compute capability 12.0. MXFP6 W6A8 uses packed E3M2 weights; MXFP8 W8A8 uses E4M3 weights. Both consume E4M3 activations with one E8M0 power-of-two scale per 32 values and accumulate in FP32 on SM120 block-scaled Tensor Cores.

The package provides Dense GEMM, Qwen3.5 routed MoE, activation producers and CUDA Graph workspaces. [vLLM Mach](https://github.com/troycheng/vllm-mach) supplies checkpoint loading, model profiles and complete serving integration.

## Model serving performance

The model-size comparison measures complete Mach serving configurations against their recorded FP8 baselines on RTX 5090. These gains include the configured attention, communication, recurrent-state and LM-head paths; they are not gains attributable to an isolated native kernel. Parallelism, runtime and baseline differ by model and remain explicit in the [performance report](docs/performance.md).

| Model / profile | GPUs | Measured gain over its FP8 baseline | Details |
|---|---|---:|---|
| Qwen3.5-2B · MXFP8 · Triton + hybrid | 1 · TP1 | +17.53% to +25.89% | [Data](docs/performance.md#qwen35-2b) |
| Qwen3.5-4B · MXFP8 champion | 1 · TP1 | +20.83% to +36.77% | [Data](docs/performance.md#qwen35-4b) |
| Qwen3.8-27B Dense · MXFP6 default / full | 2 · TP2 | default +30.41% to +43.69%; full +42.18% to +55.40% | [Data](docs/performance.md#qwen38-27b) |
| Qwen3.5-35B-A3B MoE · MXFP6 default / full | 2 · TP2 | default +48.34% to +67.01%; full +61.34% to +79.32% | [Data](docs/performance.md#qwen35-35b-a3b) |

![Model serving throughput](docs/assets/model-throughput.svg)

[Detailed service data and methodology](docs/performance.md) · [Machine-readable data](docs/data/model-performance.json)

Earlier frozen Qwen3.5 TP2 serving measurements, GEMM timings and complete MoE-layer measurements remain in [component and historical benchmarks](docs/benchmarks.md). The separate [Qwen3.8-27B study](docs/qwen38-27b.md) covers throughput, BF16 token-logprob fidelity and memory capacity. Operator timings, reference fidelity and service throughput have distinct measurement boundaries.

## Install

Requirements: Linux, SM120, Python 3.10+, CUDA Toolkit 12.8+, CUDA-enabled PyTorch with FP8 dtype support, CMake 3.24+ and a C++17 compiler. Install PyTorch and Triton first, then build against the runtime PyTorch ABI:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
python3 -m pip install 'setuptools>=77' wheel
bash scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

For an existing clone, run `git submodule update --init third_party/cutlass`. The wheel script applies the checked-in CUTLASS runtime patches and builds both extensions. `MAX_JOBS` defaults to 2. See [development and compatibility](docs/development.md) for ABI, build and upgrade policies.

Source metadata reports `0.2.1`; producer and dual-activation APIs are newer source additions. Pin the integration's required source commit and rebuild when replacing an older wheel.

## Python API

Quantize weights once and reuse them for `x[M,K] @ weight[N,K].T`:

```python
import torch
import mxfp6
import mxfp6.mxfp8 as mxfp8

x = torch.randn((32, 2560), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((4096, 2560), device="cuda", dtype=torch.bfloat16)
w6 = mxfp6.quantize_mxfp6(weight)
w8 = mxfp8.quantize_mxfp8(weight)
y6 = mxfp6.gemm(x, w6, out_dtype=torch.bfloat16)
y8 = mxfp8.gemm(x, w8)

# Quantize an activation explicitly for reuse with either format.
qa = mxfp6.quantize_activation(x)
y6 = mxfp6.gemm_w6a8(qa, w6, out_dtype=torch.bfloat16)
y8 = mxfp8.gemm_w8a8(qa, w8)
```

| Operator | Supported boundary | Guide |
|---|---|---|
| Dense W6A8 | Positive M/N/K; N multiple of 8, K multiple of 128; FP16/BF16 output | [Dense GEMM](docs/dense-gemm.md) |
| Dense W8A8 | Positive M; N/K multiples of 128; BF16 output, alpha=1 | [MXFP8](docs/mxfp8.md) |
| Dual W8A8 activation | Every M=1–128; N=12288 or 18432, K=2560; explicit opt-in | [Dual API](docs/mxfp8-dual.md) |
| Routed MoE | Qwen3.5-35B-A3B TP2 router, routed/shared experts and reduction | [MoE API](docs/qwen35-moe.md) |
| Activation producers | SwiGLU, GDN norm and attention gate with specified rounding/shape contracts | [Producer APIs](docs/producer-fusion.md) |

Packed operands require contiguous CUDA storage on one SM120 device and the documented scale layout; public quantizers construct it. Before CUDA Graph capture, plan all expected shapes, freeze workspace capacity and warm each stream. Retain input buffers and update them in place between replays. Concurrent prepared calls require separate objects. The API guides specify buffer alignment, workspace ownership and numerical contracts.

## Serving integration

Use [Mach's installation and launch guides](https://github.com/troycheng/vllm-mach) for current model profiles and dependency revisions. The library supplies operators and layouts; Mach owns model admission, scheduling and serving configuration.

The [vLLM v0.28 TP2 example](examples/vllm/README.md) is a separate version-locked reproducer for Qwen3.5-27B and Qwen3.5-35B-A3B. Its validation and earlier baseline comparison are recorded in [benchmarks](docs/benchmarks.md#public-reproduction-history).

## Documentation

| Entry | Contents |
|---|---|
| [Performance](docs/performance.md) | Model-size service comparison, absolute throughput, workloads and sources |
| [API guides](docs/README.md#operators) | Dense W6A8/W8A8, dual activations, MoE and producer contracts |
| [Checkpoint conversion](docs/quantize-checkpoints.md) | Validated Quark conversion, metadata checks and layout/ABI boundary |
| [Development](docs/development.md) | Platform, source build, ABI, integration and validation |
| [Benchmark evidence](docs/README.md#measurement-evidence) | Component results, earlier service snapshots and fidelity studies |
| [Contributing](CONTRIBUTING.md) | Contribution and review requirements |

## License

[BSD 3-Clause](LICENSE). CUTLASS and other dependencies retain their own licenses.
