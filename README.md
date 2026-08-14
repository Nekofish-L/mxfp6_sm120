<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mxfp6-wordmark-dark.png">
    <img src="docs/assets/mxfp6-wordmark-light.png" alt="MXFP6 SM120" width="520">
  </picture>
</p>

<h3 align="center">
  Native 6-bit Tensor Core Kernels for NVIDIA Blackwell GeForce
</h3>

<p align="center">
  Dense GEMM and Qwen3.5 MoE kernels with persistent MXFP6 weights<br>
  and dynamic MXFP8 activations.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#performance-highlights">Performance</a> ·
  <a href="#supported-operations">Operations</a> ·
  <a href="docs/runtime-integration.md">vLLM Integration</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-BSD--3--Clause-blue"></a>
  <img alt="GPU" src="https://img.shields.io/badge/GPU-SM120-7657F6">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-%E2%89%A512.8-20C8F6">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C">
</p>

## Performance highlights

### Public vLLM reproduction

| Model and workload | Official FP8 | MXFP6 | Output throughput gain |
|---|---:|---:|---:|
| Qwen3.5-27B, 3000 input / 1000 output, c32 | 1229.90 tok/s | 1341.97 tok/s | **+9.11%** |
| Qwen3.5-35B-A3B, public `random-mm`, c4 | 678.37 tok/s | 793.33 tok/s | **+16.95%** |

Both comparisons used two RTX 5090 GPUs, TP2 and four fresh service lifecycles
in FP8/MXFP6/MXFP6/FP8 order. Mean TPOT improved by 8.79% and 16.06%,
respectively. The 35B-A3B P99 TTFT regressed by 21.24%.

See the [public reproducer](examples/vllm/README.md) for commands and the
[benchmark artifact](benchmarks/results/qwen35_public_vllm_tp2.json) for the
complete measurement contract and all four per-block metric summaries.

### Internally optimized vLLM reference

| Model and workload | Official FP8 | MXFP6 | Output throughput gain |
|---|---:|---:|---:|
| Qwen3.5-27B, 3000 input / 1000 output, c32 | 1157.79 tok/s | 1417.63 tok/s | **+22.44%** |
| Qwen3.5-35B-A3B, frozen real multimodal, c4 | 589.41 tok/s | 669.45 tok/s | **+13.58%** |

These results used the internally maintained vLLM 0.25.1 stack. They are useful
optimization references, but are not claims about unmodified community vLLM
or generally available workloads. Their complete environment and measurement
boundaries are retained in [Benchmark methodology](docs/benchmarks.md).

### Kernel and layer results

| Scope | Result | Baseline |
|---|---:|---|
| Dense GEMM | **1.633×** | vLLM block-FP8 |
| Complete Qwen3.5 MoE layer | **1.278×** | vLLM FP8 MoE |

> [!NOTE]
> **Integration status**
>
> The standalone kernel package is usable today. A version-locked
> [vLLM v0.25.1 reproducer](examples/vllm/README.md) loads Qwen3.5-27B and
> Qwen3.5-35B-A3B. Upstream vLLM and SGLang integration is not yet available.

## What is MXFP6 SM120?

MXFP6 SM120 is a PyTorch CUDA extension for six-bit weight inference on NVIDIA
Blackwell GeForce GPUs. It keeps model weights in packed E3M2 format with one
UE8M0 scale per 32 values, dynamically quantizes FP16 or BF16 activations to
E4M3/UE8M0, and executes native SM120 block-scaled Tensor Core kernels.

The project includes general dense GEMM and specialized Qwen3.5 MoE execution
paths. It is intentionally narrow: the current package targets compute
capability 12.0 rather than presenting a portable GPU abstraction.

## Quick start

### Build and install

The extension is built from source against the active CUDA-enabled PyTorch ABI:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
./scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

For an existing clone, initialize the pinned CUTLASS dependency first:

```bash
git submodule update --init third_party/cutlass
```

### Run a dense GEMM

Quantize a persistent weight once, then call the native kernel with FP16 or
BF16 activations:

```python
import torch
import mxfp6

m, n, k = 32, 8192, 5120
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

packed_weight = mxfp6.quantize_mxfp6(weight)
output = mxfp6.gemm(a, packed_weight, out_dtype=torch.bfloat16)
```

`out_dtype` accepts `torch.float16` or `torch.bfloat16`; both are written
directly by the GEMM epilogue. See the [dense GEMM guide](docs/dense-gemm.md)
for explicit activation quantization, prewarming and workspace planning.

## Supported operations

| Operation | Public entry point | Purpose |
|---|---|---|
| Weight packing | `quantize_mxfp6` | Convert FP16/BF16 weights to persistent E3M2/UE8M0 storage |
| Dense W6A8 GEMM | `gemm`, `gemm_w6a8` | Run native mixed-precision GEMM with FP16/BF16 output |
| Activation quantization | `quantize_activation` | Produce reusable dynamic E4M3/UE8M0 activations |
| Qwen3.5 MoE | Qwen-specific workspace and layer ops | Allocation-free router, W1, W2 and reduction paths |
| Deployment preparation | `warmup_w6a8`, workspace planning APIs | Resolve dispatch and allocate graph-safe persistent state |

The dense operator accepts `M > 0`, `N % 8 == 0` and `K % 128 == 0`.
Checked-in dispatch overrides cover the primary Qwen3.5-27B TP2 linear shapes;
other supported shapes use the native dispatcher or a pre-generated autotune
cache. Qwen3.5 MoE contracts and benchmark modes are documented separately in
the [MoE guide](docs/qwen35-moe.md).

## Runtime integration status

The wheel provides standalone PyTorch operators. The version-locked
[vLLM reproducer](examples/vllm/README.md) supports the two measured Qwen3.5
profiles, but it is not a general checkpoint ABI or an upstream runtime
backend. See [Checkpoint format](docs/checkpoint-format.md) and
[Runtime integration](docs/runtime-integration.md) for the current boundary.

## Architecture

The native mixed-precision operation is:

```text
FP16/BF16 activation
        │ dynamic E4M3 + UE8M0
        ▼
SM120 block-scaled Tensor Core GEMM ◀── packed E3M2 + UE8M0 weight
        │ FP32 accumulation
        ▼
    FP16/BF16 output
```

The 16-to-8 activation mapping is deliberate: it removes repeated
activation-side six-bit unpacking from the MMA mainloop while preserving the
six-bit storage and bandwidth advantage of persistent weights. Specialized MoE
paths fuse routing-adjacent work, W1 activation processing and W2 reduction
where the Qwen3.5 topology permits it. See [Dense GEMM](docs/dense-gemm.md) and
[Qwen3.5 MoE](docs/qwen35-moe.md) for the implementation contracts.

## Compatibility

- NVIDIA GPU with compute capability 12.0 (SM120).
- CUDA Toolkit 12.8 or newer.
- CUDA-enabled PyTorch.
- CMake 3.24 or newer and a C++17 compiler.
- Ninja is recommended.

CUTLASS is pinned at `e6233cbac5d7c7a865c19c91cd684ceece19513c` and carries
versioned SM120 runtime patches from this repository. Portable prebuilt CUDA
wheels are not currently published. See [Compatibility](docs/compatibility.md)
for the measured environment and upgrade policy.

## Documentation

| Guide | Scope |
|---|---|
| [Benchmark methodology](docs/benchmarks.md) | Full result tables, baselines, workloads, metrics and reproduction commands |
| [Dense GEMM](docs/dense-gemm.md) | Data formats, public API, supported shapes and workspace model |
| [Qwen3.5 MoE](docs/qwen35-moe.md) | Specialized layer paths, TP2 evidence and graph-safe API |
| [Autotuning](docs/autotuning.md) | Dispatch policy, cache generation and serving controls |
| [Checkpoint format](docs/checkpoint-format.md) | Current in-memory layout and persistent-format acceptance bar |
| [Runtime integration](docs/runtime-integration.md) | vLLM/SGLang delivery status and upstream acceptance criteria |
| [Development](docs/development.md) | Source build, CUTLASS patch queue, tests and repository layout |

## Contributing

Contributions are welcome, especially reproducible SM120 kernels, numerical
validation, checkpoint-format work and maintainable runtime integration. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change; performance PRs
must include correctness results, environment provenance and paired evidence.

## License and acknowledgements

MXFP6 SM120 is released under the [BSD 3-Clause License](LICENSE). NVIDIA
CUTLASS retains its upstream license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices.
