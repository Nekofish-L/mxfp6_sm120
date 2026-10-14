# Benchmark artifacts

[Current model serving performance](../../docs/performance.md) has its own source index. The native operator and precision studies below retain their measured runtime and workload.

| Evidence | Artifacts | Report |
|---|---|---|
| MXFP8 decode, Qwen3.5-2B/4B TP1, 170 points at BS1–64 | [Initial policy](mxfp8_decode_20261013.json), [final policy](mxfp8_decode_20261014.json), [original 120-point study](mxfp8_small_batch_20261013.json) | [Decode results](../../docs/mxfp8.md#decode-results-october-14-2026) |
| Dense W6A8, five TP2 projection shapes ×14 batches | [70-shape matrix](qwen35_dense_gemm_current_main.json) | [Kernel benchmarks](../../docs/benchmarks.md#dense-gemm) |
| Complete TP2 MoE layer | [Nine-batch sweep](qwen35_moe_layer_current_main.json), [B4 cluster update](qwen35_moe_b4_cluster_tp2.json) | [MoE benchmarks](../../docs/benchmarks.md#complete-qwen35-moe-layer) |
| Dual activation, all M1–128 | [Synthetic JSON](mxfp8_dual_bs1_128.json), [CSV](mxfp8_dual_bs1_128.csv), [real-model JSON](mxfp8_dual_actual_inference_bs1_128.json), [CSV](mxfp8_dual_actual_inference_bs1_128.csv) | [Dual API and measurements](../../docs/mxfp8-dual.md#performance-and-activation-fidelity) |
| PDL, rotating chain and incremental HTTP comparison | [Validation JSON](mxfp8_pdl_batch_validation.json) | [PDL measurements](../../docs/mxfp8.md#pdl-measurements) |
| Qwen3.8-27B fidelity and capacity | [Fidelity](qwen38_27b_quality_fidelity.json), [TP2 metadata](qwen38_27b_concurrency_tp2.json), [TP1 capacity](qwen38_27b_capacity_tp1.json), [environment](qwen38_27b_environment.json) | [Precision and capacity](../../docs/qwen38-27b.md) |
| Public checkpoint / v0.28 adapter validation | [Conversion](qwen35_public_conversion_validation.json), [integration](vllm_v028_public_reproducer_validation.json) | [Checkpoint conversion](../../docs/quantize-checkpoints.md), [example](../../examples/vllm/README.md) |

Earlier service, dispatch, tuning and workspace JSON remain available as raw measurements in this directory. Machine-local server logs and profiler reports are not checked in.
