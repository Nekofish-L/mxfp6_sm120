# Documentation

## Start here

- [Model performance](performance.md): per-model FP8/NVFP4 throughput charts, current Mach configurations and precision against BF16.
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
| [Kernel benchmarks](benchmarks.md) | Native Dense GEMM and complete TP2 MoE layer |
| [Qwen3.8-27B](qwen38-27b.md) | FP8/MXFP6/NVFP4 fidelity and capacity |
| [Dual measurements](mxfp8-dual.md#performance-and-activation-fidelity) | Synthetic and real activations; cache-dependent latency and approximation error |
| [PDL measurements](mxfp8.md#pdl-measurements) | Incremental launch-mode HTTP and dependency-chain comparisons |
| [Raw artifacts](../benchmarks/results/README.md) | Scope and source index for checked-in JSON/CSV |

Serving gains, component speedups and reference fidelity answer different questions. Use each report's baseline, units, workload and runtime identity.
