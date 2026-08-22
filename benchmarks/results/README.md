# Benchmark artifacts

## Current evidence

| Artifact | Measurement boundary | Main result |
|---|---|---:|
| `qwen35_champion_concurrency_tp2.json` | Qwen3.5-27B and 35B-A3B TP2 full service, c1 through c32 | +9.59% to +32.70% output tok/s |
| `qwen35_dense_gemm_current_main.json` | Five 27B TP2 Dense shapes at 14 values of M | 1.592x geometric mean over 70 shapes |
| `qwen35_moe_layer_current_main.json` | Complete 35B-A3B TP2 MoE layer | 1.241x to 1.787x |

These three files are environment-bound integration snapshots, not public
reproducers. The service artifact compares official FP8 and MXFP6 on one frozen
Champion runtime and contains two opposite-order lifecycles per format, all
absolute latencies, token totals and repeat spread. The layer artifacts use
`mxfp6_sm120` commit `eb1e582`.

## Public reproduction

| Artifact | Contents |
|---|---|
| `qwen35_public_vllm_tp2.json` | Public vLLM v0.25.1 FP8/MXFP6 service comparison for both models |
| `qwen35_public_conversion_validation.json` | Public conversion, checkpoint and TP2 load validation |

The public service workload differs from the internal Champion sweep. Its
results remain useful for independent setup but should not be combined with the
Champion numbers.

## Earlier and specialized results

| Artifact | Contents |
|---|---|
| `qwen35_27b_service_tp2.json` | Earlier internal 27B service comparison |
| `qwen35_moe_service_tp2.json` | Earlier internal 35B-A3B service and reference-fidelity comparison |
| `native_w6a8_dispatch.json` | Earlier Dense dispatch matrix |
| `qwen35_moe_tp2.json` | Earlier complete MoE layer sweep |
| `qwen35_moe_b4_vector.json` | B4 packed-load implementation comparison |
| `persistent_workspace.json` | Persistent workspace comparison |
| `runtime_autotune_v4.json` | Runtime autotuning evidence |

Serving, complete-layer and isolated-GEMM artifacts have different measurement
boundaries. Their speedups are not additive. Machine-local autotune caches,
server logs and profiler reports are not checked in.
