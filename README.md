<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mxfp6-wordmark-dark.png">
    <img src="docs/assets/mxfp6-wordmark-light.png" alt="MXFP6 SM120" width="520">
  </picture>
</p>

<h3 align="center">
  Native OCP W6A8 Execution on NVIDIA SM120 Tensor Cores
</h3>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#execution-model">Design</a> ·
  <a href="#build-and-minimal-use">Build</a> ·
  <a href="#operator-surface">Operators</a> ·
  <a href="#runtime-integration">vLLM Integration</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-BSD--3--Clause-blue"></a>
  <img alt="GPU" src="https://img.shields.io/badge/GPU-SM120-7657F6">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-%E2%89%A512.8-20C8F6">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C">
</p>

## Overview

`mxfp6-sm120` is a PyTorch CUDA extension for OCP W6A8 inference on NVIDIA
compute capability 12.0. It keeps model weights in packed MXFP6, quantizes
activations to MXFP8 at runtime, and executes the mixed-precision operation with
native SM120 block-scaled Tensor Core MMA.

The repository contains a general Dense GEMM engine and a Qwen3.5 MoE schedule.
The MoE implementation owns the GPU path from routing through W1, activation
requantization, W2, shared-expert combine and final reduction. The checked-in
benchmarks measure both isolated kernels and the complete TP2 MoE layer.

## Performance

### End-to-end serving on public vLLM

| Model and workload | Official FP8 | MXFP6 | Output throughput gain |
|---|---:|---:|---:|
| Qwen3.5-27B, 3000 input / 1000 output, c32 | 1229.90 tok/s | 1341.97 tok/s | **+9.11%** |
| Qwen3.5-35B-A3B, public `random-mm`, c4 | 678.37 tok/s | 793.33 tok/s | **+16.95%** |

Both tests used two PIX-connected RTX 5090 GPUs, tensor parallel size 2, and
four fresh service runs in FP8/MXFP6/MXFP6/FP8 order. See the
[public vLLM reproducer](examples/vllm/README.md) and the
[four-block result artifact](benchmarks/results/qwen35_public_vllm_tp2.json).

### Native kernel and MoE layer

| Scope | Result | Baseline |
|---|---:|---|
| Qwen3.5 Dense GEMM, five shapes × 10 batch sizes | **1.633×** geometric mean | vLLM block-FP8 GEMM |
| Complete Qwen3.5 MoE layer, B16-B96 | **1.278×** geometric mean | vLLM FP8 MoE layer |

Kernel, layer and serving results have different boundaries and are not
additive. The [benchmark methodology](docs/benchmarks.md) records the full
contracts, per-shape data and latency metrics. All checked-in data is indexed in
[benchmark artifacts](benchmarks/results/README.md).

<details>
<summary>Internal vLLM reference results</summary>

The internal runtime includes additional vLLM 0.25.1 optimizations and uses
frozen workloads. These numbers are retained as optimization references, not as
results for the public reproducer.

| Model and workload | Official FP8 | MXFP6 | Output throughput gain |
|---|---:|---:|---:|
| Qwen3.5-27B, 3000 input / 1000 output, c32 | 1157.79 tok/s | 1417.63 tok/s | **+22.44%** |
| Qwen3.5-35B-A3B, frozen real multimodal, c4 | 589.41 tok/s | 669.45 tok/s | **+13.58%** |

The 35B-A3B artifact includes a 742-case reference-output comparison. Its paired
intervals did not detect a significant fidelity loss. This measures reference
fidelity, not audited task accuracy.

</details>

## Execution model

### W6A8 data path

```text
BF16/FP16 activation
        │ dynamic E4M3 quantization + UE8M0 scale (1:32 along K)
        ▼
MXFP8 activation ───────────────────────────────────────────┐
                                                            ├─► SM120 block-scaled MMA
packed MXFP6 E3M2 weight + UE8M0 scale (1:32 along K) ──────┘            │
                                                                         ▼
                                                        FP32 accumulate ─► FP16/BF16 output
```

Weights remain in the six-bit representation during inference. The GEMM does
not expand them to FP8 or FP16 before the mainloop.

| Operand | Representation | Scale granularity |
|---|---|---|
| Weight | MXFP6 E3M2, four values packed into three bytes | UE8M0 per 32 values along K |
| Activation | MXFP8 E4M3 generated from FP16/BF16 | UE8M0 per 32 values along K |
| Accumulator | FP32 | n/a |
| Output | FP16 or BF16 | n/a |

### Dense dispatch

The Dense operator computes `D[M,N] = A[M,K] @ B[N,K].T`. Its SM120 dispatch
portfolio is shape-aware rather than a single universal kernel:

| Region | Kernel policy |
|---|---|
| Small M | Swapped x8/x16/x32 tiles with ping-pong, cooperative and static-persistent schedules |
| Large M | Normal 64x64, 64x128 and 128x128 Tensor Core tiles |
| Selected shapes | Stream-K scheduling with a reusable persistent workspace |

Exact overrides cover the five Qwen3.5-27B TP2 linear shapes at ten measured
batch sizes. Other valid shapes use the native fallback policy or a generated
autotune cache.

### Qwen3.5 MoE schedule

```text
hidden states
    │
    ├─► router: gate GEMV + softmax/top-k + input quantization
    │
    └─► W1 gate/up ─► SiLU-and-mul ─► MXFP8 requantization
                                      │
                                      ▼
                   W2 over 8 routed experts + shared expert
                                      │
                                      ▼
                         routed/shared weighted reduction
                                      │
                                      ▼
                               rank-local output
```

Batch 1/2/4 uses allocation-free routing, split-K W1 and fused W2 reduction.
Larger batches use indirect routing, a TMA W1 path over the interleaved gate/up
projection and grouped W2. Caller-owned workspaces keep every production path
allocation-free during CUDA Graph capture and replay.

### Validated scope

| Component | Tested boundary |
|---|---|
| Dense W6A8 GEMM | Native SM120 dispatch for `M > 0`, `N % 8 == 0`, `K % 128 == 0` |
| Qwen3.5 MoE | Specialized router, W1, W2, shared-expert and reduction paths; validated with Qwen3.5-35B-A3B TP2 |
| CUDA Graphs | Prewarmed dispatch and persistent workspaces; no capture-time allocation or tuning |
| Checkpoints | Two tested model-scoped Quark layouts; no stable general checkpoint ABI yet |
| Serving runtime | Version-locked public vLLM v0.25.1 prototype for Qwen3.5-27B and Qwen3.5-35B-A3B |

The package targets SM120. Devices, checkpoint layouts and runtime
configurations outside this table have not been tested.

## Build and minimal use

Build the extension against the CUDA-enabled PyTorch installation that will load
the resulting wheel:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
./scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

For an existing clone, initialize the pinned CUTLASS dependency before building:

```bash
git submodule update --init third_party/cutlass
```

Quantize a persistent weight once, then reuse it across calls:

```python
import torch
import mxfp6

m, n, k = 32, 8192, 5120
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

packed_weight = mxfp6.quantize_mxfp6(weight)
output = mxfp6.gemm(a, packed_weight, out_dtype=torch.bfloat16)
```

`out_dtype` accepts `torch.float16` and `torch.bfloat16`. Production runtimes
should call the warmup and workspace-planning APIs before CUDA Graph capture;
see [Dense GEMM](docs/dense-gemm.md) and [Autotuning](docs/autotuning.md).

## Operator surface

| Operation | Public entry point | Contract |
|---|---|---|
| Weight quantization and packing | `quantize_mxfp6` | FP16/BF16 to persistent E3M2/UE8M0 storage |
| Dense W6A8 GEMM | `gemm`, `gemm_w6a8` | Native mixed-precision GEMM with FP16/BF16 output |
| Activation quantization | `quantize_activation` | Reusable dynamic E4M3/UE8M0 activation tensors |
| Dispatch preparation | `warmup_w6a8`, configuration APIs | Resolve a supported kernel before graph capture |
| Workspace planning | persistent workspace APIs | Allocate stable storage for graph replay |
| Qwen3.5 MoE | Qwen-specific workspace and layer ops | Allocation-free router/W1/W2/reduction paths within the documented topology |

Checked-in dense overrides cover the primary Qwen3.5-27B TP2 linear shapes.
Other valid shapes use the native dispatcher or a generated autotune cache.
Qwen3.5 MoE shapes, tensor ordering and workspace contracts are documented in
[Qwen3.5 MoE kernels](docs/qwen35-moe.md).

## Runtime integration

The wheel exports standalone PyTorch operators. `examples/vllm` contains a
version-locked vLLM v0.25.1 compatibility layer that loads and serves the two
measured Qwen3.5 profiles without internal infrastructure. It is a public
reproducer, not a general checkpoint loader or an upstream backend.

vLLM already recognizes Quark/OCP MXFP6 checkpoints, but its CUDA path currently
uses software emulation. The public prototype connects the existing loading
path to the native Dense and MoE operators for the two tested models. It does
not change vLLM's behavior outside that version-locked setup. See
[Runtime integration](docs/runtime-integration.md) and
[Checkpoint format status](docs/checkpoint-format.md).

## Compatibility

- NVIDIA GPU with compute capability 12.0 (SM120).
- CUDA Toolkit 12.8 or newer.
- CUDA-enabled PyTorch.
- CMake 3.24 or newer and a C++17 compiler.
- Ninja is recommended.

CUTLASS is pinned at `e6233cbac5d7c7a865c19c91cd684ceece19513c` with versioned
SM120 runtime patches from this repository. Portable prebuilt CUDA wheels are
not currently published. See [Compatibility](docs/compatibility.md) for the
measured toolchain and upgrade policy.

## Documentation

| Guide | Scope |
|---|---|
| [Dense GEMM](docs/dense-gemm.md) | Data formats, public API, supported shapes and workspace model |
| [Qwen3.5 MoE](docs/qwen35-moe.md) | Specialized execution paths, TP2 evidence and graph-safe API |
| [Benchmark methodology](docs/benchmarks.md) | Baselines, workloads, metrics, full result tables and commands |
| [Autotuning](docs/autotuning.md) | Dispatch policy, cache generation and serving controls |
| [Checkpoint format](docs/checkpoint-format.md) | Current in-memory layout and persistent-format acceptance bar |
| [Runtime integration](docs/runtime-integration.md) | vLLM/SGLang status and proposed upstream boundary |
| [Development](docs/development.md) | Source build, CUTLASS patch queue, tests and repository layout |

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change. Performance
PRs must include correctness coverage, environment identity, paired samples and
a claim explicitly bounded to kernel, layer or end-to-end service scope.

## License and acknowledgements

MXFP6 SM120 is released under the [BSD 3-Clause License](LICENSE). NVIDIA
CUTLASS retains its upstream license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices.
