<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mxfp6-wordmark-dark.png">
    <img src="docs/assets/mxfp6-wordmark-light.png" alt="MXFP6 SM120" width="520">
  </picture>
</p>

<h3 align="center">
  Native MXFP6 W6A8 and MXFP8 W8A8 kernels for NVIDIA SM120
</h3>

<p align="center">
  <a href="#build">Build</a> ·
  <a href="#python-api">API</a> ·
  <a href="#validated-scope">Support</a> ·
  <a href="#execution-model">Design</a> ·
  <a href="#vllm-integration">vLLM Mach</a> ·
  <a href="#performance">Performance</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-BSD--3--Clause-blue"></a>
  <img alt="GPU" src="https://img.shields.io/badge/GPU-SM120-7657F6">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-%E2%89%A512.8-20C8F6">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C">
</p>

`mxfp6-sm120` is a PyTorch CUDA kernel library for native OCP microscaling on
NVIDIA compute capability 12.0. It provides **MXFP6 W6A8** with packed E3M2
weights and **MXFP8 W8A8** with E4M3 weights. Both consume E4M3 activations with
one E8M0 power-of-two scale per 32 values and use SM120 block-scaled Tensor Core
MMA with FP32 accumulation.

| Capability | Package entrypoint |
|---|---|
| Dense W6A8, FP16/BF16 output | `mxfp6.gemm` |
| Dense W8A8, BF16 output | `mxfp6.mxfp8.gemm` |
| Fixed-shape dual-activation W8A8 | `mxfp6.mxfp8_dual` |
| Qwen3.5 routed MoE | Router, W1, SiLU/mul, quantization, W2 and routed/shared reduction |
| Producer fusion and graph replay | Fused MXFP8 preparation, native dispatch and persistent workspaces |

The library supplies operators and packed tensor layouts. Model loading,
scheduler configuration and serving profiles are provided by
[vLLM Mach](https://github.com/troycheng/vllm-mach).

## Build

Requirements: Linux, Python 3.10+, CUDA Toolkit 12.8+, CUDA-enabled PyTorch
with FP8 dtype support, CMake 3.24+ and a C++17 compiler. Install PyTorch and
Triton in the target environment first, then build against that PyTorch ABI:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
python3 -m pip install 'setuptools>=77' wheel
bash scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

For an existing clone, run `git submodule update --init third_party/cutlass`
before building. The wheel script applies the checked-in runtime CUTLASS patches
and builds both `mxfp6_torch` and `mxfp8_torch`. `MAX_JOBS` controls compilation
parallelism; the default is 2. See [compatibility](docs/compatibility.md) and
[development](docs/development.md) for toolchain and source-build details.

Current source still reports version `0.2.1`, while producer and dual-activation
APIs are newer source additions. Pin the source commit required by your
integration, rebuild, and use `--force-reinstall` when replacing an older wheel.

## Python API

Quantize weights once, then reuse them for `x[M,K] @ weight[N,K].T`:

```python
import torch
import mxfp6
import mxfp6.mxfp8 as mxfp8

m, n, k = 32, 4096, 2560
x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

w6 = mxfp6.quantize_mxfp6(weight)
w8 = mxfp8.quantize_mxfp8(weight)
y6 = mxfp6.gemm(x, w6, out_dtype=torch.bfloat16)
y8 = mxfp8.gemm(x, w8, out_dtype=torch.bfloat16)

# Reuse an explicitly quantized activation with either weight format.
qa = mxfp6.quantize_activation(x)
y6 = mxfp6.gemm_w6a8(qa, w6, out_dtype=torch.bfloat16)
y8 = mxfp8.gemm_w8a8(qa, w8)
```

Both backends accept FP16/BF16 activations or the shared `MXFP8Tensor` type.
MXFP6 defaults to FP16 output; MXFP8 supports BF16 output and `alpha=1`.
Their Python wrappers call native C++ dispatch and workspace management.

### CUDA Graph preparation

Plan every expected shape and dtype, freeze the workspace capacity, then warm
each capture stream. The two backends maintain independent workspace pools:

```python
backend = mxfp8  # Use mxfp6 with w6 for the same planning workflow.
backend.begin_workspace_planning()
backend.warmup(qa, w8, out_dtype=torch.bfloat16)
backend.finalize_workspace_planning()
backend.warmup(qa, w8, out_dtype=torch.bfloat16)
```

Retain input buffers and update their contents in place between replays. For
MXFP8, `prepare(qa, w8)` also returns a callable with private output/workspace;
concurrent prepared calls need separate objects. See [Dense GEMM](docs/dense-gemm.md),
[MXFP8](docs/mxfp8.md) and [autotuning](docs/autotuning.md) for complete contracts.

## Validated scope

| Component | Supported boundary |
|---|---|
| Platform | Linux, NVIDIA SM120; binaries target `sm_120a` |
| Dense W6A8 | `M,N,K > 0`, `N % 8 == 0`, `K % 128 == 0` |
| Dense W8A8 | `M > 0`, positive N/K multiples of 128; BF16 output, `alpha=1` |
| Packed operands | Contiguous CUDA storage on one device; packed E8M0/32 scales |
| Routed MoE | Qwen3.5-35B-A3B TP2 router, routed/shared experts and reduction |
| Graph replay | Prewarmed dispatch, planned capacity and persistent buffers |
| Model integration | Model-scoped checkpoint layouts and runtime profiles in Mach |

W8A8 raw operands and scales require 16-byte-aligned storage and the padded
CUTLASS 128×4 scale swizzle. The public quantizers create these layouts.
[Checkpoint conversion](docs/quantize-checkpoints.md) and
[checkpoint format](docs/checkpoint-format.md) describe the MXFP6 model layouts.

## Execution model

```text
FP16/BF16 activation ── quantize E4M3 + E8M0/32 ──┐
                                                │
MXFP6 packed E3M2 or MXFP8 E4M3 weight + E8M0/32 ──┤ SM120 MMA
                                                │
                           FP32 accumulation ────┘
                                  │
                   W6A8: FP16/BF16; W8A8: BF16
```

MXFP6 keeps four E3M2 values in three bytes throughout the mainloop. Its Dense
portfolio combines swapped small-M tiles, cooperative/persistent schedules and
Stream-K, with tuned Qwen projection overrides and a general fallback policy.
MXFP8 selects kernels through a compiled M/N/K interval policy; its default
quantizer and GEMM use native C++/CUDA and CUTLASS/CuTe.

Qwen3.5 MoE operators cover small-batch and grouped schedules, including routing,
W1, activation/quantization, W2 and routed/shared combine. Caller-owned workspaces
support graph replay; see the [MoE guide](docs/qwen35-moe.md).

Dense producer APIs feed packed activations directly into W6A8:
[`gemm_from_swiglu`](docs/tp2-swiglu.md) preserves the rounded SwiGLU contract,
and `gemm_from_gdn` fuses gated RMSNorm/MXFP8 preparation for BF16 `[M,24,128]`
at M1/2/4/8/16/24/32 using the pinned vLLM 0.29 norm reduction layout.
The [attention-gate producer](docs/attention-gate.md) is an explicit API.
Mach documents model-level eligibility in its
[producer integration guide](https://github.com/troycheng/vllm-mach/blob/main/docs/dense-producer-fusion.md).

## Fixed-shape dual MXFP8 activations

The opt-in `mxfp6.mxfp8_dual` module quantizes finite BF16 activations into a
high E4M3 limb and a BF16-rounded residual E4M3 limb, each with its own scales.
GEMM uses independent FP32 accumulators, adds them after reduction and rounds
once to BF16. It supports exactly `(M,N,K) = (32,18432,2560)`,
`(32,12288,2560)` and `(64,12288,2560)`.

```python
import torch
import mxfp6.mxfp8 as mxfp8
from mxfp6 import mxfp8_dual

x_dual = torch.randn((32, 2560), device="cuda", dtype=torch.bfloat16)
w_dual = mxfp8.quantize_mxfp8(
    torch.randn((12288, 2560), device="cuda", dtype=torch.bfloat16)
)
hi, s_hi, res, s_res = mxfp8_dual.quantize(x_dual)
out = torch.empty((32, 12288), device="cuda", dtype=torch.bfloat16)
mxfp8_dual.gemm_out(
    hi, w_dual.dequantized_values(), s_hi, w_dual.scales, res, s_res, out
)
```

The dual quantizer uses Triton; the three GEMMs are native kernels linked into
`mxfp8_torch`. `quantize_out` and `gemm_out` support preallocated graph replay.
This API is selected explicitly and retains quantized weights and activations;
BF16 output does not imply BF16-equivalent numerics. See the
[dual-activation guide](docs/mxfp8-dual.md) for buffer contracts and validation.

## vLLM integration

[vLLM Mach](https://github.com/troycheng/vllm-mach) provides the current official
serving integration, including checkpoint admission, graph/workspace lifecycle
and model-specific optimizations. Its README describes three routes:

| Mach route | Kernel relationship |
|---|---|
| Native MXFP6 TP2 Dense/MoE | This library's W6A8 and Qwen MoE operators |
| Qwen3.5-4B MXFP8 TP1 Champion | Native W8A8, including selected dual-activation kernels |
| Qwen3.5-4B existing block-FP8 TP1 | Existing block-FP8 execution path |

Follow Mach's installation and launch instructions for the required source
revisions and runtime dependencies. Mach's full profiles combine kernel,
communication, attention, recurrent-state and LM-head choices; their results
measure the complete serving configuration.

[`examples/vllm`](examples/vllm/README.md) remains a version-locked vLLM v0.28.0
TP2 reproducer for Qwen3.5-27B and Qwen3.5-35B-A3B.
[vLLM issue #52347](https://github.com/vllm-project/vllm/issues/52347) tracks the
upstream native MXFP6 work.

## Performance

The following snapshots have distinct measurement scopes. Speedup is baseline
latency divided by native latency; serving percentages are output-throughput
gains. MXFP6 figures retain the recorded runtime revisions, and the MXFP8
figures use the cache policy in its guide.

| Scope and workload | Baseline | Recorded result | Evidence |
|---|---|---|---|
| W6A8 GEMM, five 27B TP2 projections × 14 M values, warm cache | vLLM block FP8 | **1.688×** geometric mean over 70 shapes; activation quantization excluded | [Dense artifact](benchmarks/results/qwen35_dense_gemm_current_main.json) |
| Complete 35B-A3B TP2 MoE layer, M1–96, CUDA Graph including custom all-reduce | Official FP8 layer | **1.241–1.787×** across nine batches | [MoE artifact](benchmarks/results/qwen35_moe_layer_current_main.json) |
| Updated 35B-A3B TP2 MoE B4 cluster reduction and vector loads | Official FP8 layer / previous MXFP6 service | **1.313×** layer speedup; **+4.41%** service throughput over the previous MXFP6 implementation at c4 | [B4 update](benchmarks/results/qwen35_moe_b4_cluster_tp2.json) |
| W8A8 GEMM, 2B/4B TP1 shapes, 240 cases, cold cache | FlashInfer MXFP8 / vLLM block FP8 | Approx. **1.14× / 1.26×** geometric mean; quantization excluded | [MXFP8 methodology](docs/mxfp8.md#approximate-performance) |
| Frozen 27B TP2 service, c1–c32 | Official FP8, same internal Champion runtime | **+17.43–21.36%** output throughput | [Service artifact](benchmarks/results/qwen35_champion_concurrency_tp2.json) |
| Frozen 35B-A3B TP2 service, c1–c32 | Official FP8, same internal Champion runtime | **+17.14–32.92%** output throughput | [Service artifact](benchmarks/results/qwen35_champion_concurrency_tp2.json) |
| Qwen3.8-27B TP2 service, c1–c32 | Official FP8, frozen runtime | **+18.26–22.43%** throughput; model load 14.09 → 11.46 GiB/GPU | [Qwen3.8 report](docs/qwen38-27b.md) |

All measurements above use RTX 5090. Service snapshots are environment-bound;
operator and layer speedups are separate from full-service gains. The
[benchmark report](docs/benchmarks.md) contains request contracts, absolute
latencies, repeat spread and numerical checks.

For Qwen3.8-27B, 256 teacher-forced records gave sample-mean token-logprob MAE
against BF16 of 0.05719 (FP8), 0.09078 (MXFP6) and 0.17661 (NVFP4). MXFP6 was
13.55–16.74% slower than NVFP4 in the fixed TP2 concurrency sweep. These fidelity,
throughput and capacity trade-offs are detailed in the
[Qwen3.8 report](docs/qwen38-27b.md).

## Documentation

| Guide | Contents |
|---|---|
| [Dense GEMM](docs/dense-gemm.md) | W6A8 formats, shapes, operators and workspaces |
| [MXFP8](docs/mxfp8.md) / [dual activations](docs/mxfp8-dual.md) | W8A8 API, dispatch, graph capture and measurements |
| [Qwen3.5 MoE](docs/qwen35-moe.md) | Routing, expert execution and graph-safe APIs |
| [Checkpoint conversion](docs/quantize-checkpoints.md) / [format](docs/checkpoint-format.md) | Public Qwen3.5 conversion and Quark layouts |
| [Autotuning](docs/autotuning.md) | W6A8 dispatch policy and cache generation |
| [Benchmarks](docs/benchmarks.md) / [Qwen3.8-27B](docs/qwen38-27b.md) | Operator, layer, serving and fidelity evidence |
| [Compatibility](docs/compatibility.md) / [development](docs/development.md) | Platform, ABI, source build and tests |
| [Runtime integration](docs/runtime-integration.md) | Runtime interfaces and upstream status |

## License

MXFP6 SM120 is released under the [BSD 3-Clause License](LICENSE). CUTLASS and
other dependencies retain their own licenses.
