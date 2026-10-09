# Benchmark artifacts

## Current evidence

| Artifact | Measurement boundary | Main result |
|---|---|---:|
| `mxfp8_dual_actual_inference_bs1_128.json` / `.csv` | Full Qwen3.5-4B inference, real activations and natural caches; all 128 requested batches, 32 gate/up + 24 QKVZ layers | GEMM-only dual latency +27.3% QKVZ / +31.8% gate/up; see `docs/mxfp8-dual-actual-inference.md` |
| `mxfp8_dual_bs1_128.json` / `.csv` | Every M=1–128 for QKVZ and gate/up; native single/dual, GEMM-only and quantization+GEMM, hot/cold weights | Native dual covers all 256 points; see `docs/mxfp8-dual-bs1-128.md` for latency and error |
| `mxfp8_pdl_batch_validation.json` | Real MXFP8 weights, 8-layer dependency chain, and paired full HTTP serving lifecycles | Low-concurrency HTTP +1.6% to +2.1%; M=33–128 chain +2.2% to +3.8%; see `docs/mxfp8-pdl-performance.md` |
| `qwen38_27b_quality_fidelity.json` | Qwen3.8-27B, 256 records, BF16 teacher-forced token-logprob fidelity | MXFP6 MAE 48.6% below NVFP4; FP8 lowest |
| `qwen38_27b_concurrency_tp2.json` | Qwen3.8-27B TP2, fixed ISL 1024 / OSL 256, c1 through c32 | MXFP6 +18.26% to +22.43% vs FP8; NVFP4 fastest |
| `qwen38_27b_serving_tp2.json` | Qwen3.8-27B TP2, three workloads, FP8/MXFP6/NVFP4 | MXFP6 +18.23% to +22.33% vs FP8; NVFP4 fastest |
| `qwen38_27b_capacity_tp1.json` | Qwen3.8-27B TP1 startup at a shared 90% memory budget | MXFP6 and NVFP4 ready; FP8 had no KV block |
| `qwen35_champion_concurrency_tp2.json` | Qwen3.5-27B and 35B-A3B TP2 full service, c1 through c32 | +17.14% to +32.92% output tok/s |
| `qwen35_moe_b4_cluster_tp2.json` | 35B-A3B production B4 layer and c4 full service | +4.41% over the previous MXFP6 Champion; +17.14% vs FP8 |
| `qwen35_dense_gemm_current_main.json` | Five 27B TP2 Dense shapes at 14 values of M | 1.688x geometric mean over 70 shapes |
| `qwen35_moe_layer_current_main.json` | Complete 35B-A3B TP2 MoE layer | 1.241x to 1.787x |

These files are environment-bound integration snapshots, not public
reproducers. The Qwen3.8 environment artifact is
`qwen38_27b_environment.json`; its concurrency artifact retains two fresh
forward/reverse-order lifecycles per format, and its workload-matrix artifact
retains two fresh forward/reverse-order blocks per workload. The Qwen3.5
service artifact
compares official FP8 and MXFP6 on one frozen Champion runtime and contains two
opposite-order samples per point, all absolute latencies, token totals and
repeat spread. The B4 artifact records the new code identity, full-service
samples, complete-layer timing and correctness boundary. The Dense artifact
uses `mxfp6_sm120` commit `53035f7`; the earlier general MoE artifact uses
commit `eb1e582`.

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
| `mxfp8_pdl_review.json` | Synthetic MXFP8 quantization/GEMM PDL review: paired hot-cache Graph and eager samples, environment and correctness scope; no serving claim |
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
