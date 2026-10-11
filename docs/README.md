# Documentation

## Start here

- [Model performance](performance.md): current complete Mach serving configurations by model size, absolute throughput, baseline and workload.
- [Checkpoint conversion and format](quantize-checkpoints.md): validated model layouts, metadata admission and persistent ABI limits.
- [Development and compatibility](development.md): source build, toolchain, ABI, runtime surfaces and validation.

## Operators

| Guide | Contract |
|---|---|
| [Dense W6A8](dense-gemm.md) | Quantization, shapes, outputs and persistent workspaces |
| [Dense W8A8](mxfp8.md) | MXFP8 dispatch, prepared calls, graph and PDL APIs |
| [Dual activations](mxfp8-dual.md) | High/residual limbs, M1–128 and graph buffers |
| [Qwen3.5 MoE](qwen35-moe.md) | Routing, experts, shared reduction and workspace ownership |
| [Activation producers](producer-fusion.md) | SwiGLU, GDN norm, attention gate and scale initialization |
| [Autotuning](autotuning.md) | W6A8 selection policy and offline cache generation |

## Measurement evidence

| Report | Scope |
|---|---|
| [Component and earlier service benchmarks](benchmarks.md) | Native GEMM/MoE measurements and frozen Qwen3.5 TP2 service snapshots |
| [Qwen3.8-27B](qwen38-27b.md) | Separate FP8/MXFP6/NVFP4 fidelity, service and capacity study |
| [Dual performance](mxfp8-dual-performance.md) | Synthetic and real activations; cache-dependent latency and approximation error |
| [PDL performance](mxfp8-pdl-performance.md) | Incremental launch-mode HTTP and dependency-chain comparisons |
| [Raw artifacts](../benchmarks/results/README.md) | Scope and source index for checked-in JSON/CSV |

Serving gains, component speedups and reference fidelity answer different questions. Use each report's baseline, units, workload and runtime identity.
