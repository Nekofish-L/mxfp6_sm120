# Benchmark methodology and results

The project reports three measurement layers. Each answers a different
question:

| Layer | Question | Primary metric |
|---|---|---|
| Full service | Does MXFP6 improve the complete vLLM request path? | Output tokens/s, TPOT and TTFT |
| Complete MoE layer | Does the routed-expert implementation reduce GPU time? | TP2 CUDA Graph latency |
| Dense GEMM | Which projection shapes benefit from the native matrix kernel? | Isolated GEMM latency |

Layer and GEMM speedups are diagnostic evidence. They are never added to the
full-service gain.

The separate [Qwen3.8-27B snapshot](qwen38-27b.md) compares BF16 fidelity,
TP2 serving and TP1 capacity across official FP8, MXFP6 and standard NVFP4
W4A4. Its integration environment and workload differ from the Qwen3.5
Champion results below, so the numbers should not be combined.

## Champion full-service comparison

### Runtime

FP8 and MXFP6 ran in the same frozen environment:

| Component | Configuration |
|---|---|
| GPUs | 2 x NVIDIA GeForce RTX 5090, SM120, PIX topology |
| Parallelism | Tensor parallel size 2 |
| vLLM | 0.25.1 internally optimized Champion |
| vLLM build | `752a3a504485790a2e8491cacbb35c137339ad34` |
| Container image | `sha256:075bb856626daacf090e20d39b01329a752914aae8dade2339f8b4bb2f908e9f` |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| FlashInfer | 0.6.18 with the validated Qwen3.5 TP2 GDN registry and local #4634 |
| CUDA Graph sizes | 1, 2, 4, 8, 16, 24, 32 |
| Scheduler | 4096 maximum batched tokens, asynchronous scheduling |
| Collective | FlashInfer TRT-LLM AllReduce/RMSNorm fusion |
| Disabled | Prefix caching and speculative decoding |
| mxfp6-sm120 | `c123379e6d02f4d74d3e0b85879a1ebf9eda6064` |

Service logs confirmed the intended full CUDA Graph captures, fused GDN path,
AllReduce/RMSNorm backend and quantized kernels. MXFP6 runs additionally logged
the native Dense or native grouped-MoE backend.

### Workloads

Qwen3.5-27B used deterministic fixed-token requests with 3000 input and 1000
output tokens. The request count was 32 at concurrency 1, 2, 4 and 8; 64 at 16;
96 at 24; and 128 at 32.

Qwen3.5-35B-A3B used a frozen real multimodal request set. Sampling parameters,
per-request seeds and requested output lengths were fixed. Concurrency 1 through
16 used the same 64 records, concurrency 24 used 96, and concurrency 32 used
128. The first 64 records contain 203,631 prompt and 59,193 completion tokens.

### Acquisition and aggregation

The comparison uses four fresh service lifecycles:

1. FP8 A1, concurrency ascending;
2. MXFP6 B1, concurrency descending;
3. MXFP6 B2, concurrency ascending;
4. FP8 A2, concurrency descending.

Each point had a concurrency-matched warmup. The current update reran both FP8
and MXFP6 c4 lifecycles for the 35B-A3B B4 cluster optimization. The unchanged
35B-A3B c1/c2 and grouped-from-B5 c8-c32 points were retained, as were all 27B
points. Every reported point remains the mean of two opposite-order
service-lifecycle samples; no best-run selection is used. All points completed
their request and token contracts with zero failures.

![Full-service FP8 and MXFP6 throughput curves](assets/full-service-throughput.svg)

The figure is generated directly from the service artifact with
[`plot_champion_concurrency.py`](../benchmarks/plot_champion_concurrency.py).
Error bars show the range of the two opposite-order lifecycles.

### Qwen3.5-27B results

| c | FP8 tok/s | MXFP6 tok/s | Throughput gain | Mean TPOT reduction | P99 TPOT reduction | P99 TTFT change | Repeat spread, FP8 / MXFP6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.93 | 100.70 | +18.57% | 15.51% | 15.51% | -21.31% | 0.32% / 0.21% |
| 2 | 160.12 | 193.09 | +20.59% | 16.73% | 16.95% | -27.70% | 0.04% / 0.15% |
| 4 | 292.78 | 355.31 | +21.36% | 16.72% | 17.33% | -28.26% | 0.13% / 0.14% |
| 8 | 514.85 | 618.93 | +20.21% | 16.19% | 17.11% | -20.95% | 0.06% / 0.37% |
| 16 | 822.24 | 965.53 | +17.43% | 14.13% | 14.82% | -19.41% | 0.39% / 0.31% |
| 24 | 1033.50 | 1229.07 | +18.92% | 15.35% | 16.03% | -20.42% | 0.21% / 0.33% |
| 32 | 1179.66 | 1395.28 | +18.28% | 14.86% | 15.77% | -19.78% | 0.18% / 0.28% |

`c` is request concurrency. It does not imply that every projection executes
with token batch `M=c`.

### Qwen3.5-35B-A3B results

| c | FP8 tok/s | MXFP6 tok/s | Throughput gain | Mean TPOT reduction | P99 TPOT reduction | P99 TTFT change | Repeat spread, FP8 / MXFP6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 270.94 | 329.13 | +21.48% | 17.48% | 17.21% | -59.59% | 3.89% / 0.46% |
| 2 | 415.02 | 526.67 | +26.90% | 21.92% | 19.91% | -10.65% | 0.86% / 0.68% |
| 4 | 697.59 | 817.15 | +17.14% | 14.37% | 12.28% | -0.71% | 1.18% / 0.24% |
| 8 | 1010.63 | 1262.86 | +24.96% | 20.68% | 17.76% | -13.48% | 1.21% / 0.19% |
| 16 | 1365.96 | 1766.50 | +29.32% | 23.47% | 22.33% | -4.28% | 1.34% / 0.37% |
| 24 | 1660.71 | 2182.02 | +31.39% | 23.64% | 18.21% | -5.36% | 2.10% / 0.04% |
| 32 | 1757.01 | 2335.39 | +32.92% | 24.65% | 9.88% | -14.30% | 6.68% / 0.64% |

The larger FP8 spread at c1 and c32 tracks lifecycle order. A point run first in
a fresh FP8 service was slower than the same point after the rest of the curve.
The ABBA design places each format in both orders and reports their mean. MXFP6
did not show the same sensitivity.

The complete machine-readable result is
[`qwen35_champion_concurrency_tp2.json`](../benchmarks/results/qwen35_champion_concurrency_tp2.json).
It includes all 56 service points, absolute latency metrics, token totals and
repeat spread.

## Dense GEMM

The Dense benchmark covers the five Qwen3.5-27B TP2 projection shapes:

| Layer | N | K |
|---|---:|---:|
| GDN input projection | 8192 | 5120 |
| GDN output projection | 5120 | 3072 |
| Full-attention QKV/gate projection | 7168 | 5120 |
| MLP gate/up projection | 17408 | 5120 |
| MLP down projection | 5120 | 8704 |

It used BF16 input, FP16 output, 20 warmups, 100 measured iterations, warm L2
state and disabled runtime autotuning. The ratio compares vLLM block-scaled FP8
GEMM latency with MXFP6 GEMM latency; dynamic activation quantization is reported
separately and excluded from the ratio.

| M | Geometric mean speedup over five shapes |
|---:|---:|
| 1 | 1.977x |
| 2 | 1.983x |
| 4 | 1.981x |
| 8 | 1.998x |
| 16 | 1.603x |
| 24 | 1.514x |
| 32 | 1.532x |
| 64 | 1.232x |
| 96 | 1.555x |
| 512 | 1.829x |
| 1024 | 1.761x |
| 2048 | 1.697x |
| 4096 | 1.590x |
| 8192 | 1.588x |
| Overall | 1.688x |

All 70 native outputs matched the decoded FP32 matmul reference exactly after
FP16 output conversion. Exact static dispatch now covers the five measured
projection shapes at `M=24`, improving that batch's geometric-mean ratio from
0.717x to 1.514x. The full rows are in
[`qwen35_dense_gemm_current_main.json`](../benchmarks/results/qwen35_dense_gemm_current_main.json).

Reproduce the sweep with:

```bash
CUDA_VISIBLE_DEVICES=0 MXFP6_AUTOTUNE=0 \
python3 benchmarks/benchmark.py \
  --shapes 1x8192x5120,2x8192x5120 \
  --library build/mxfp6_torch.so \
  --activation-input bf16 --output-dtype fp16 \
  --compare-fp8 --check-all \
  --warmup 20 --iterations 100 --flush-l2-mb 0
```

Pass the full `(M,N,K)` list from the artifact for the 70-shape matrix. The
abbreviated command above shows the input format without duplicating a long
generated argument in the documentation.

## Complete Qwen3.5 MoE layer

The TP2 benchmark captures routing, routed and shared experts, top-k weighting,
reduction and vLLM custom all-reduce in one CUDA Graph. It uses real layer-0
weights. Each result is the median of nine paired repeats after 40 warmups and
1000 graph replays; each distributed sample is the slower rank.

| Token batch | FP8 layer | MXFP6 layer | Speedup | Relative RMS | Cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.3 us | 16.4 us | 1.787x | 0.1117 | 0.9937 |
| 2 | 32.8 us | 20.5 us | 1.601x | 0.1152 | 0.9936 |
| 4 | 32.8 us | 25.0 us | 1.313x | 0.0912 | 0.9958 |
| 8 | 54.5 us | 38.6 us | 1.411x | 0.0946 | 0.9955 |
| 16 | 123.7 us | 90.3 us | 1.370x | 0.0903 | 0.9959 |
| 24 | 162.4 us | 127.1 us | 1.278x | 0.0980 | 0.9952 |
| 32 | 188.4 us | 147.1 us | 1.281x | 0.0949 | 0.9955 |
| 64 | 238.9 us | 188.3 us | 1.269x | 0.0980 | 0.9952 |
| 96 | 264.6 us | 207.0 us | 1.278x | 0.0994 | 0.9951 |

The recorded routed expert IDs matched the FP8 checkpoint at every batch. The
run recorded this comparison but did not configure a mismatch as a hard process
failure. The [artifact](../benchmarks/results/qwen35_moe_layer_current_main.json)
retains that setting, every result and all paired timing samples.

```bash
CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=lo \
torchrun --master-addr 127.0.0.1 --nproc-per-node=2 \
  benchmarks/benchmark_qwen35_moe_layer.py \
  --fp8-model /models/Qwen3.5-35B-A3B-FP8 \
  --mx-model /models/Qwen3.5-35B-A3B-MXFP6 \
  --batch-sizes 1 2 4 8 16 24 32 64 96 \
  --mx-mode auto --no-small-separate-shared \
  --b4-packed-vector-loads --route-stats \
  --warmup 40 --iterations 1000 --repeats 9
```

The command selects the combined-weight small-batch execution form used by the
measured internal Champion. The checked-in public vLLM adapter retains its
version-locked weight-loading contract; its public serving numbers are reported
separately below.

## Quality evidence

Dense output checks and complete-layer MoE comparisons test numerical behavior
at the operator boundary. A separate 742-case greedy comparison on the frozen
35B-A3B workload measured reference-output fidelity against FP8. Full-character
similarity changed by -0.57 percentage points and answer-section similarity by
-0.32 points. Neither paired 95% interval detected a significant difference.
This is reference fidelity, not task accuracy.

The 27B serving sweep did not run a task-accuracy suite. Performance data should
not be read as a quality claim beyond the operator checks above.

## Public reproduction history

The earlier vLLM v0.25.1 image established that both model paths could run
without the internal Champion environment. Its archived four-lifecycle results
were:

| Model and public workload | FP8 tok/s | MXFP6 tok/s | Gain |
|---|---:|---:|---:|
| Qwen3.5-27B, 3000 input / 1000 output, c32 | 1229.90 | 1341.97 | +9.11% |
| Qwen3.5-35B-A3B, public `random-mm`, c4 | 678.37 | 793.33 | +16.95% |

The current [`examples/vllm`](../examples/vllm/README.md) image is based on
vLLM v0.28.0. A single-lifecycle validation on the same dual-RTX-5090 host used
the current MXFP6 checkpoints, TP2, `max_num_batched_tokens=4096`, prefix
caching disabled and CUDA Graph sizes `[1,2,4,8,16,24,32]`:

| Model | Concurrency | Public v0.28 tok/s | Comparable internal reference | Delta |
|---|---:|---:|---:|---:|
| Qwen3.5-27B | 4 | 355.02 | 355.31 | -0.08% |
| Qwen3.5-27B | 32 | 1342.33 | 1395.28 | -3.80% |
| Qwen3.5-35B-A3B | 4 | 788.11 | 817.15 | -3.55% |
| Qwen3.5-35B-A3B | 32 | 2011.82 | 2043.08 (internal, hybrid lm_head disabled) | -1.53% |

The public image therefore reproduces both native execution paths and tracks
the comparable internal MXFP6 runtime within 3.80% at the reported points. The
full 35B-A3B Champion result of 2335.39 tok/s additionally includes hybrid
NVFP4 lm_head quantization; that separate optimization is excluded from the
public template and is not attributed to MXFP6. These values are an integration
validation, not a replacement for the paired concurrency sweep above. The
[validation artifact](../benchmarks/results/vllm_v028_public_reproducer_validation.json)
records the exact token contracts and whether the SM120 GDN prefill backport
was active for each point.

## Artifact index

| Artifact | Scope |
|---|---|
| [`qwen35_champion_concurrency_tp2.json`](../benchmarks/results/qwen35_champion_concurrency_tp2.json) | Latest Champion full-service c1 through c32 sweep |
| [`vllm_v028_public_reproducer_validation.json`](../benchmarks/results/vllm_v028_public_reproducer_validation.json) | Public vLLM v0.28 two-model c4/c32 integration validation |
| [`qwen35_moe_b4_cluster_tp2.json`](../benchmarks/results/qwen35_moe_b4_cluster_tp2.json) | B4 cluster optimization, correctness and fresh c4 service comparison |
| [`qwen35_dense_gemm_current_main.json`](../benchmarks/results/qwen35_dense_gemm_current_main.json) | Current-main 70-shape Dense GEMM matrix |
| [`qwen35_moe_layer_current_main.json`](../benchmarks/results/qwen35_moe_layer_current_main.json) | Current-main complete TP2 MoE layer |
| [`qwen35_public_vllm_tp2.json`](../benchmarks/results/qwen35_public_vllm_tp2.json) | Historical public vLLM v0.25.1 service comparison |
| [`qwen35_moe_service_tp2.json`](../benchmarks/results/qwen35_moe_service_tp2.json) | Earlier internal 35B-A3B service and fidelity evidence |
| [`qwen35_27b_service_tp2.json`](../benchmarks/results/qwen35_27b_service_tp2.json) | Earlier internal 27B service evidence |
