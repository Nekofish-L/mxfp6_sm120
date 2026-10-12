# Model serving performance

Measured complete configurations on RTX 5090: **3,000 input / 1,000 output tokens**, text-only, non-speculative inference. Throughput is **output tokens/s**; `c` is concurrent requests, and physical kernel rows are denoted `M`.

Each model has its own throughput chart. The 27B Dense and 35B-A3B MoE comparisons include FP8, NVFP4 and both MXFP6 profiles, followed by a separate precision chart against BF16.

MXFP6 **default** retains FP32 recurrent state and the BF16 head. **Full** adds FP16 state and NVFP4 candidate search with BF16 refinement. The 2B MXFP8 champion uses BF16 KV and K128 hybrid head at M≤32; 4B uses calibrated FP8 KV and its own refined head.

<a id="qwen35-2b"></a>
## Qwen3.5-2B — MXFP8 · Triton + hybrid

Mach profile: 2026-10-11 · 1 × RTX 5090 · TP1

![Qwen3.5-2B output throughput by concurrency](assets/qwen35-2b-throughput.svg)

| Concurrency | Community FP8 | Previous FA2/full-head | Current Triton + hybrid | vs FP8 | vs previous |
|---|---:|---:|---:|---:|---:|
| c4 | 1261.89 | 1395.31 | 1588.27 | +25.86% | +13.83% |
| c16 | 3814.33 | 4295.10 | 4801.96 | +25.89% | +11.80% |
| c24 | 5069.68 | 5594.33 | 6209.52 | +22.48% | +11.00% |
| c32 | 5870.22 | 6595.49 | 7259.99 | +23.67% | +10.08% |
| c48 | 6837.60 | 7896.46 | 8036.47 | +17.53% | +1.77% |
| c64 | 7578.86 | 8702.23 | 9008.34 | +18.86% | +3.52% |

| Concurrency | Selected mean TTFT (ms) | Selected mean TPOT (ms) |
|---|---:|---:|
| c4 | 98.90 | 2.421 |
| c16 | 134.09 | 3.198 |
| c24 | 192.81 | 3.666 |
| c32 | 224.49 | 4.174 |
| c48 | 277.45 | 5.678 |
| c64 | 333.26 | 6.743 |

[Profile / study](https://github.com/troycheng/vllm-mach/blob/main/docs/mxfp8-2b-champion.md) · [Source measurements](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/mxfp8-2b-triton-hybrid-sixpoint-20261011.json)

<a id="qwen35-4b"></a>
## Qwen3.5-4B — MXFP8 champion

Mach profile: 2026-10-06 · 1 × RTX 5090 · TP1

![Qwen3.5-4B output throughput by concurrency](assets/qwen35-4b-throughput.svg)

| Concurrency | FP8 baseline | Selected profile | Gain |
|---|---:|---:|---:|
| c4 | 683.96 | 880.13 | +28.68% |
| c16 | 2053.55 | 2481.30 | +20.83% |
| c24 | 2559.26 | 3189.42 | +24.62% |
| c32 | 2887.82 | 3672.70 | +27.18% |
| c48 | 3220.55 | 4168.04 | +29.42% |
| c64 | 3402.76 | 4653.93 | +36.77% |

| Concurrency | Selected mean TTFT (ms) | Selected mean TPOT (ms) |
|---|---:|---:|
| c4 | 173.56 | 4.375 |
| c16 | 237.62 | 6.211 |
| c24 | 347.70 | 7.167 |
| c32 | 410.70 | 8.286 |
| c48 | 540.36 | 10.947 |
| c64 | 688.76 | 13.017 |

[Profile / study](https://github.com/troycheng/vllm-mach/blob/main/docs/mxfp8-champion.md) · [Source measurements](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/mxfp8-qualification-20261006.json)

<a id="qwen38-27b"></a>
## Qwen3.8-27B Dense — MXFP6 default / full

Mach profile: 2026-09-20 · 2 × RTX 5090 · TP2

![Qwen3.8-27B Dense output throughput by concurrency](assets/qwen38-27b-throughput.svg)

| Concurrency | FP8 | NVFP4 | MXFP6 default | MXFP6 full |
|---|---:|---:|---:|---:|
| c4 | 264.57 | 378.97 | 380.17 | 411.14 |
| c16 | 802.79 | 1142.87 | 1046.93 | 1141.40 |
| c24 | 1011.43 | 1424.61 | 1374.53 | 1498.93 |
| c32 | 1150.97 | 1607.56 | 1553.09 | 1718.24 |

| Concurrency | Default vs FP8 | Full vs FP8 | Default vs NVFP4 | Full vs NVFP4 |
|---|---:|---:|---:|---:|
| c4 | +43.69% | +55.40% | +0.32% | +8.49% |
| c16 | +30.41% | +42.18% | -8.39% | -0.13% |
| c24 | +35.90% | +48.20% | -3.52% | +5.22% |
| c32 | +34.94% | +49.29% | -3.39% | +6.89% |

### Qwen3.8-27B Dense precision

![Qwen3.8-27B Dense gold-token logprob MAE against BF16](assets/qwen38-27b-precision.svg)

| FP8 MAE | NVFP4 MAE | MXFP6 default MAE | MXFP6 full MAE |
|---:|---:|---:|---:|
| 0.06143 | 0.17328 | 0.09064 | 0.08962 |

M32 · 256 queries / 10,479 fixed gold tokens · query-mean logprob MAE against BF16. Lower is better. Error bars show 95% query-bootstrap intervals.

[NVFP4 throughput (2026-09-15)](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/native-serving.json) · [Precision measurements](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/profile-fidelity-20260917.json)

[Profile / study](https://github.com/troycheng/vllm-mach/blob/main/docs/dense-gains-aligned-baseline-20260920.md) · [Source measurements](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/dense-gains-aligned-baseline-20260920.json)

<a id="qwen35-35b-a3b"></a>
## Qwen3.5-35B-A3B MoE — MXFP6 default / full

Mach profile: 2026-09-28 · 2 × RTX 5090 · TP2

![Qwen3.5-35B-A3B MoE output throughput by concurrency](assets/qwen35-35b-a3b-throughput.svg)

| Concurrency | FP8 | NVFP4 | MXFP6 default | MXFP6 full |
|---|---:|---:|---:|---:|
| c4 | 679.89 | 730.60 | 1008.56 | 1096.93 |
| c16 | 1471.58 | 1654.20 | 2338.97 | 2561.85 |
| c24 | 1774.64 | 1985.31 | 2920.74 | 3073.56 |
| c32 | 1962.14 | 2202.11 | 3277.02 | 3518.44 |

| Concurrency | Default vs FP8 | Full vs FP8 | Default vs NVFP4 | Full vs NVFP4 |
|---|---:|---:|---:|---:|
| c4 | +48.34% | +61.34% | +38.05% | +50.14% |
| c16 | +58.94% | +74.09% | +41.40% | +54.87% |
| c24 | +64.58% | +73.19% | +47.12% | +54.82% |
| c32 | +67.01% | +79.32% | +48.81% | +59.78% |

### Qwen3.5-35B-A3B MoE precision

![Qwen3.5-35B-A3B MoE gold-token logprob MAE against BF16](assets/qwen35-35b-a3b-precision.svg)

| FP8 MAE | NVFP4 MAE | MXFP6 default MAE | MXFP6 full MAE |
|---:|---:|---:|---:|
| 0.05534 | 0.22836 | 0.07261 | 0.07409 |

M32 · 256 queries / 10,479 fixed gold tokens · query-mean logprob MAE against BF16. Lower is better. Error bars show 95% query-bootstrap intervals.

[NVFP4 throughput (2026-09-17)](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/qwen35-default-full-20260917.json) · [Precision measurements](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/profile-fidelity-20260917.json)

[Profile / study](https://github.com/troycheng/vllm-mach/blob/main/docs/moe-gains-fp8-baseline-20260920.md) · [Source measurements](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/moe-recheck-20260928.json)

<a id="qwen35-4b-block-fp8"></a>
## Qwen3.5-4B block-FP8 — Block-FP8 · all four features

Mach profile: 2026-10-06 · 1 × RTX 5090 · TP1

![Qwen3.5-4B block-FP8 output throughput by concurrency](assets/qwen35-4b-block-fp8-throughput.svg)

| Concurrency | FP8 baseline | Selected profile | Gain |
|---|---:|---:|---:|
| c4 | 678.54 | 739.84 | +9.03% |
| c16 | 2018.89 | 2143.60 | +6.18% |
| c32 | 2834.84 | 3004.00 | +5.97% |
| c64 | 3363.73 | 3598.84 | +6.99% |

| Concurrency | Selected mean TTFT (ms) | Selected mean TPOT (ms) |
|---|---:|---:|
| c4 | 192.46 | 5.218 |
| c16 | 260.16 | 7.205 |
| c32 | 451.17 | 10.182 |
| c64 | 771.87 | 16.955 |

[Profile / study](https://github.com/troycheng/vllm-mach/blob/main/docs/fp8-profile.md) · [Source measurements](https://github.com/troycheng/vllm-mach/blob/eaa0acd6f5700913d10a422f6d4075614b9fb6be/docs/data/fp8-qualification-20261006.json)

## Numerical fidelity

Precision is measured as token-logprob MAE against BF16 using 256 queries / 10,479 tokens per physical-row setting. Lower MAE means closer agreement with BF16.

| Profile | Physical rows | Numerical result |
|---|---|---|
| 2B MXFP8 Triton + hybrid | M4/16/24/32/48/64 | BF16-anchor MAE 0.04783–0.04947; M32 +2.58% versus FA2. Actual-generation head checks: 129,878 decisions, no omitted BF16 winner or lower-scored selection, one equal-score tie. |
| 4B MXFP8 | M32/64 | Same-M BF16 MAE 0.04904 / 0.05329. |
| 4B block-FP8 | M4/32/64 | Fixed-gold records byte-identical to stock block-FP8. |
| 27B Dense MXFP6 default / full | M32 | BF16 MAE 0.09064 / 0.08962; FP8 0.06143, NVFP4 0.17328. |
| 35B-A3B MoE MXFP6 default / full | M32 | BF16 MAE 0.07261 / 0.07409; FP8 0.05534, NVFP4 0.22836. |

[2B precision and generation coverage](https://github.com/troycheng/vllm-mach/blob/main/docs/mxfp8-2b-champion.md#precision) · [4B qualification](https://github.com/troycheng/vllm-mach/blob/main/docs/mxfp8-champion.md#validation) · [MXFP6 fidelity methodology](https://github.com/troycheng/vllm-mach/blob/main/docs/profile-fidelity-20260917.md)

### Measurement protocol

The linked studies and [source records](data/model-performance.json) retain hardware identities, software revisions, request counts and baseline dates. MXFP6 precision results use the September 17 evaluation. MXFP6 and 2B logprob checks use the complete BF16 head; hybrid-head validation uses actual generation. The fixed-gold corpus is private.

The 27B NVFP4 throughput reference is the September 15 token-ID sweep. The 35B-A3B NVFP4 throughput reference is the September 17 user service; its launch configuration was not recorded. The separate 35B-A3B precision evaluation used the NVFP4 ScaleSweep checkpoint. Source records preserve these identities and measurement dates.

## Data and regeneration

The [normalized data](data/model-performance.json) record source revisions, SHA-256 hashes, unrounded throughputs and per-point ratios. Gain is `100 × (selected / baseline − 1)`.

These figures are generated by [vLLM Mach's renderer](https://github.com/troycheng/vllm-mach/blob/main/tools/render_performance_docs.py). The native-only kernel and earlier service studies remain in [benchmarks](benchmarks.md).
