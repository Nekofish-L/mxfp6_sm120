# vLLM 0.33 aggregate evidence

These JSON files contain published aggregate numbers, request/window counts and original evidence hashes. They reproduce the report calculations; private raw logs and query text are not included.

Each window throughput is completion tokens divided by duration. The formal five-window records retain the initial AA window followed by A0, B0, B1, A1: OLD pools A0 and A1, NEW pools B0 and B1. Community throughput pools the recorded independent repeats. Pooling means total completion tokens divided by total duration. Report gains use `100 * (selected / community - 1)`; migration gains compare NEW with OLD and have a different denominator. Request counts include every saved window, including AA.

The 2B FP32 default joins five accepted R27 points with R42 c24. It is a source-scoped composite, not a new full campaign. Native M32 precision is the unweighted mean of 256 query token-logprob MAEs against the original model-specific BF16 anchor (10,479 scored tokens); confidence intervals use 20,000 query bootstrap replicates, seed 20260910. Stock FP8/NVFP4 results use vLLM 0.33; checkpoint byte identity with the historical anchor checkpoint has not been established.

[Normalized report data](../model-performance-vllm033.json) · [Precision aggregates](precision-aggregate.json) · [Other measured configurations](migration-status.json)

- [dense_current](dense_current.json)
- [dense_default](dense_default.json)
- [dense_full](dense_full.json)
- [four_fp32](four_fp32.json)
- [moe_current](moe_current.json)
- [moe_full_final](moe_full_final.json)
- [moe_default](moe_default.json)
- [ref_DENSE_FP8](ref_DENSE_FP8.json)
- [ref_DENSE_NVFP4](ref_DENSE_NVFP4.json)
- [ref_MOE_FP8](ref_MOE_FP8.json)
- [ref_MOE_NVFP4](ref_MOE_NVFP4.json)
- [reference_4B_COMMUNITY033_VS_R15_V1](reference_4B_COMMUNITY033_VS_R15_V1.json)
- [reference_community-rest-v1_2B_COMMUNITY033_REFERENCE](reference_community-rest-v1_2B_COMMUNITY033_REFERENCE.json)
- [two_b_bf16_final](two_b_bf16_final.json)
- [two_b_fp32_default_composite](two_b_fp32_default_composite.json)
- [two_fp32_c24_final](two_fp32_c24_final.json)
- [two_fp32_derived_ci95](two_fp32_derived_ci95.json)
- [two_fp32_original](two_fp32_original.json)
- [verified_complete](verified_complete.json)
