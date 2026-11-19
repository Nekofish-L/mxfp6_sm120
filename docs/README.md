# Documentation

## Start here

- [Native 0.3.2 installation](../README.md#install) covers the release wheel and source build.
- [Mach 0.4.0 model setup](https://github.com/troycheng/vllm-mach/blob/v0.4.0/docs/README.md#model-setup) covers checkpoints and launch commands for vLLM 0.34.0 and 0.33.0.
- [Checkpoint conversion and format](quantize-checkpoints.md) covers validated model layouts, metadata admission and persistent ABI limits.
- [Development and compatibility](development.md) covers source build, toolchain, ABI, runtime surfaces and validation.

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
| [vLLM 0.34 throughput](performance.md#vllm-034-throughput) and [precision](https://github.com/troycheng/vllm-mach/blob/v0.4.0/docs/installation.md#vllm-034-qualification) | Eight serving points across five Mach profiles |
| [vLLM 0.33 concurrency curves](performance.md#vllm-0330-concurrency-curves) and [protocol](data/evidence-vllm033/README.md) | Per-concurrency throughput and precision |
| [Kernel benchmarks](benchmarks.md) | Native Dense GEMM and complete TP2 MoE layer |
| [Qwen3.8-27B](qwen38-27b.md) | FP8/MXFP6/NVFP4 fidelity and capacity |
| [Dual measurements](mxfp8-dual.md#performance-and-activation-fidelity) | Synthetic and real activations; cache-dependent latency and approximation error |
| [PDL measurements](mxfp8.md#pdl-measurements) | Incremental launch-mode HTTP and dependency-chain comparisons |
| [Raw artifacts](../benchmarks/results/README.md) | Scope and source index for checked-in JSON/CSV |

[Historical vLLM 0.29 performance](https://github.com/troycheng/vllm-mach/blob/v0.4.0/docs/performance-vllm029.md) retains the earlier FP8/NVFP4 throughput and precision comparisons.
