# Measurement data and protocol

The current report covers vLLM 0.33, Torch 2.14 and Triton 3.8 on RTX 5090. All serving charts use 3,000 input and 1,000 output tokens; rates are output tokens/s.

## Model records

| Serving profile | Throughput aggregates | Precision |
|---|---|---|
| 2B MXFP8 | [FP32](two_b_fp32_default_composite.json), [BF16](two_b_bf16_final.json) | [Fixed-gold results](precision-aggregate.json), [FP32 intervals](two_fp32_derived_ci95.json) |
| 4B MXFP8 | [FP32](four_fp32.json) | [Fixed-gold results](precision-aggregate.json) |
| 4B block-FP8 | [Four-point results](block_fp8.json) | [BF16-reference validation](block_fp8.json) |
| 27B Dense MXFP6 | [FP32](dense_current.json), [FP16 FULL](dense_full.json) | [Fixed-gold results](precision-aggregate.json) |
| 35B-A3B MoE MXFP6 | [FP32](moe_current.json), [FP16 FULL](moe_full_final.json) | [Fixed-gold results](precision-aggregate.json) |

[Normalized chart data](../model-performance-vllm033.json) records every plotted value and its source. The [upgrade and additional configuration results](migration-status.json) retain the paired vLLM 0.29 → 0.33 comparisons, including optional 4B BF16 and 2B headgraph-OFF settings.

## Throughput

A window's throughput is completion tokens divided by duration. Pooled throughput is total completion tokens divided by total duration. The five-window upgrade records retain AA, A0, B0, B1 and A1; OLD pools A0/A1 and NEW pools B0/B1. Request counts include AA.

Chart gains use `100 * (Mach / community - 1)`. Community results pool independent repeats on vLLM 0.33; upgrade gains use the previous Mach configuration as their denominator. Block-FP8 uses the same observed checkpoint/tokenizer path, config, exact prompt-token IDs and request contract as its community reference. The four feature switches distinguish the configurations. Full model-file byte identity was not independently frozen in the community receipt.

| Community reference | Records |
|---|---|
| 2B FP8 | [vLLM 0.33](reference_community-rest-v1_2B_COMMUNITY033_REFERENCE.json) |
| 4B FP8 | [vLLM 0.33](reference_4B_COMMUNITY033_VS_R15_V1.json) |
| 27B Dense | [FP8](ref_DENSE_FP8.json), [NVFP4](ref_DENSE_NVFP4.json) |
| 35B-A3B MoE | [FP8](ref_MOE_FP8.json), [NVFP4](ref_MOE_NVFP4.json) |

The 2B FP32 curve combines five accepted points from R27 with the final R42 c24 result. Block-FP8 combines accepted R40 c4/c16 with R51 c32/c64. MoE FULL combines accepted R49 c4 with R53 c16/c24/c32. Each aggregate retains the original source hashes and windows.

## Precision

Precision is the mean of per-query gold-token logprob MAEs against the model's BF16 anchor. Fixed-gold datasets contain 256 queries and 10,479 tokens. Where shown, 95% intervals use 20,000 query-bootstrap replicates with seed 20260910. Block-FP8's chart shows point estimates against its matching BF16 anchors; its current repeated runs are bitwise stable.

Stock FP8/NVFP4 measurements use vLLM 0.33 and the frozen historical BF16 anchor. Their checkpoint byte identity with the anchor checkpoint was not established. Aggregate JSON preserves the reference identity and model configuration for each result; query text and private raw logs are not published.

## Earlier configurations

[Dense head OFF](dense_default.json), [MoE head OFF](moe_default.json), [original 2B FP32 points](two_fp32_original.json) and [final 2B c24](two_fp32_c24_final.json) retain earlier configuration and per-point records. The [historical vLLM 0.29 report](https://github.com/troycheng/vllm-mach/blob/main/docs/performance-vllm029.md) contains the original FP8/NVFP4 comparisons.
