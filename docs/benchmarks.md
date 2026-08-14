# Benchmark methodology and results

This document records the baselines, workloads, metrics and full result tables
behind the headline figures in the project README. Layer results isolate GPU
implementations; serving results measure the complete request path. A layer
speedup is never used as a substitute for an end-to-end result.

## Result summary

| Model | Evidence level | FP8 baseline | Headline MXFP6 result | Artifact |
|---|---|---|---:|---|
| Qwen3.5-27B | public vLLM v0.25.1 TP2 serving | official Qwen FP8 checkpoint | +9.11% output tok/s | [`qwen35_public_vllm_tp2.json`](../benchmarks/results/qwen35_public_vllm_tp2.json) |
| Qwen3.5-35B-A3B | public vLLM v0.25.1 TP2 serving, public random-mm | official Qwen FP8 checkpoint | +16.95% output tok/s | [`qwen35_public_vllm_tp2.json`](../benchmarks/results/qwen35_public_vllm_tp2.json) |
| Qwen3.5-27B | five real linear-layer shapes | vLLM block FP8 | 1.633x geometric mean | [`native_w6a8_dispatch.json`](../benchmarks/results/native_w6a8_dispatch.json) |
| Qwen3.5-35B-A3B | complete TP2 MoE layer | vLLM FP8 MoE path | 1.278x geometric mean, B16-B96 | [`qwen35_moe_tp2.json`](../benchmarks/results/qwen35_moe_tp2.json) |
| Qwen3.5-27B | internal vLLM TP2 serving | official Qwen FP8 checkpoint | +22.44% output tok/s | [`qwen35_27b_service_tp2.json`](../benchmarks/results/qwen35_27b_service_tp2.json) |
| Qwen3.5-35B-A3B | internal vLLM TP2 serving, frozen real multimodal | official Qwen FP8 checkpoint | +13.58% output tok/s | [`qwen35_moe_service_tp2.json`](../benchmarks/results/qwen35_moe_service_tp2.json) |

The first two rows are the primary community-facing results. The next two
isolate kernel and complete-layer behavior. The final two are bounded references
from the internally maintained vLLM 0.25.1 stack. These evidence levels explain
different parts of the system and their speedups must not be combined.

### Experimental public vLLM reproduction

The checked-in example uses the official vLLM v0.25.1 image and four fresh
FP8/MXFP6/MXFP6/FP8 service lifecycles. The complete metrics, including the
35B-A3B P99 TTFT regression, are in the
[benchmark artifact](../benchmarks/results/qwen35_public_vllm_tp2.json).
See [`examples/vllm`](../examples/vllm/README.md) for commands.

## Qwen3.5-27B

### Linear-layer kernel evidence

This benchmark covers the model's five TP2 linear `(N,K)` shapes at each batch
size. It used one RTX 5090, PyTorch `2.11.0+cu130`, CUDA 13.0, BF16 input, FP16
output, warm cache, 20 warmups and 100 measured iterations. Runtime autotuning
was disabled. The vLLM block-FP8 baseline came from the pinned locally patched
development runtime; the Humming reference used revision `694298e9`.

All ratios are standalone GEMM latency ratios. Activation quantization, weight
preparation, Humming JIT compilation and host gaps are excluded.

| M | vs Humming W6A8 | vs vLLM block-FP8 W8A8 |
|---:|---:|---:|
| 1 | 1.449x | 1.973x |
| 16 | 1.137x | 1.754x |
| 32 | 1.254x | 1.530x |
| 64 | 10.612x† | 1.222x |
| 96 | 21.220x† | 1.572x |
| 512 | 1.197x | 1.814x |
| 1024 | 1.330x | 1.737x |
| 2048 | 1.230x | 1.671x |
| 4096 | 1.078x | 1.588x |
| 8192 | 1.067x | 1.583x |
| Overall | 1.212x‡ | 1.633x |

† The pinned Humming configuration enters a correct but pathological slow path
at M=64 and M=96. These rows are shown but excluded from the comparable Humming
overall value. Including them gives a raw 50-shape overall of 2.005x.

‡ Geometric mean over the remaining 40 Humming comparisons. A dedicated BS32
static-dispatch rerun measured 1.278x with BF16 input and 1.277x with FP16.

The vLLM baseline is W8A8 while MXFP6 stores six-bit weights, so this is a
performance comparison rather than a claim that the formats or footprints are
identical. Full dispatch, correctness and timing metadata are in
[`native_w6a8_dispatch.json`](../benchmarks/results/native_w6a8_dispatch.json).

### End-to-end serving evidence

The baseline is the official
[`Qwen/Qwen3.5-27B-FP8`](https://huggingface.co/Qwen/Qwen3.5-27B-FP8)
revision `97f5941bf617e31c5e237364a8602ce3f03a551a`; all 11 local weight shards
matched its Hugging Face LFS SHA-256 values. The candidate is an MXFP6 Quark
export of the same architecture and execution dimensions.

Two fresh service lifecycles per format were run in FP8/MXFP6/MXFP6/FP8 order
on two PIX-connected RTX 5090 GPUs. The fixed-token workload contained 320
requests of 3000 input and 1000 output tokens at 20 requests/s and concurrency
32. Every block completed all requests with exactly 960,000 prompt and 320,000
completion tokens; the primary numerator came from final server-reported usage.

| Format | Output tok/s | Mean TPOT | P99 TPOT | P99 TTFT |
|---|---:|---:|---:|---:|
| Qwen FP8 | 1157.79 | 25.94 ms | 27.44 ms | 7327.79 ms |
| Native MXFP6 | 1417.63 | 21.31 ms | 22.32 ms | 5435.27 ms |

MXFP6 improved output-token throughput by **22.44%**. The two FP8 throughput
blocks differed by 0.24%, and the two MXFP6 blocks by 0.25%. Both formats used
PyTorch 2.11.0+cu130, CUDA 13.0, FlashInfer, a 4096-token scheduler budget, 64
maximum sequences and no speculative decoding. This experiment did not score
task accuracy. The complete ABBA samples and contract are in
[`qwen35_27b_service_tp2.json`](../benchmarks/results/qwen35_27b_service_tp2.json).

## Qwen3.5-35B-A3B

### Whole MoE-layer evidence

The TP2 benchmark captures the complete MoE layer and vLLM custom all-reduce
in one CUDA Graph using real layer-0 checkpoint weights. Its FP8 baseline is
vLLM's runtime Triton block-FP8 expert path, auxiliary-stream shared expert and
graph-registered custom all-reduce. The MXFP6 path includes the same collective.

Small batches use an allocation-free router, split-K W1 with fused SiLU-and-mul
and MXFP8 quantization, and fused W2 routed/shared reduction. Larger batches use
indirect routing, a TMA W1 path over the checkpoint's interleaved gate/up
projection and a sparse grouped W2 path. Dependent small-batch grids synchronize
before their first producer-dependent global-memory access.

The table reports the median of 9 repeats after 40 warmups and 1000 CUDA Graph
replays per repeat.

| Batch | vLLM FP8 | Native MXFP6 | Speedup |
|---:|---:|---:|---:|
| 1 | 29.2 us | 16.4 us | 1.782x |
| 2 | 33.8 us | 20.5 us | 1.652x |
| 4 | 32.8 us | 26.4 us | 1.240x |
| 8 | 54.5 us | 38.8 us | 1.404x |
| 10 | 67.7 us | 40.4 us | 1.676x |
| 12 | 105.1 us | 59.6 us | 1.763x |
| 14 | 112.4 us | 73.6 us | 1.527x |
| 16 | 123.7 us | 91.2 us | 1.356x |
| 24 | 162.4 us | 128.2 us | 1.267x |
| 32 | 184.4 us | 144.7 us | 1.274x |
| 40 | 203.2 us | 159.9 us | 1.271x |
| 48 | 226.4 us | 177.5 us | 1.275x |
| 56 | 227.7 us | 182.9 us | 1.245x |
| 64 | 239.0 us | 188.4 us | 1.268x |
| 80 | 254.9 us | 198.9 us | 1.282x |
| 96 | 265.5 us | 209.9 us | 1.265x |

The geometric-mean speedup over B16-B96 is **1.2778x**. At B1 the FP8/MXFP6
outputs had relative RMS error 0.1117 and cosine similarity 0.9937. Per-batch
numerical and timing results are in
[`qwen35_moe_tp2.json`](../benchmarks/results/qwen35_moe_tp2.json).

As a separate implementation refinement, aligned packed FP6 loads on the B4
separate-shared path improved the same-process complete-layer result from 26.62
us to 24.65 us (1.080x) with exact output parity. This compares two MXFP6
implementations, not MXFP6 against FP8. Samples are in
[`qwen35_moe_b4_vector.json`](../benchmarks/results/qwen35_moe_b4_vector.json).

Reproduce that paired refinement with:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  benchmarks/benchmark_qwen35_moe_layer.py \
  --batch-sizes 4 --mx-mode auto --compare-b4-packed-vector-loads \
  --input-seed 20260815 --warmup 40 --iterations 1000 --repeats 9 \
  --json-out b4_packed_vector.json
```

### End-to-end serving evidence

The baseline is the official
[`Qwen/Qwen3.5-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)
revision `9d1823d2dee688a6b25e77009dc727688c44936e`; all 14 local weight shards
matched its Hugging Face LFS SHA-256 values. The MXFP6 Quark export used the
same architecture and tokenizer assets and was 20.14% smaller on disk.

The FP8/MXFP6/MXFP6/FP8 experiment used two PIX-connected RTX 5090 GPUs, TP2,
the same runtime and 64 frozen real multimodal requests at concurrency 4. Every
block completed 64/64 requests and reported 203,631 prompt plus 59,193
completion tokens. Output lengths, sampling parameters and per-request seeds
were frozen; final server usage supplied the throughput numerator.

| Format | Output tok/s | Mean TPOT | P99 TPOT | P99 TTFT |
|---|---:|---:|---:|---:|
| Qwen FP8 | 589.41 | 6.16 ms | 8.31 ms | 1541.82 ms |
| Native MXFP6 | 669.45 | 5.54 ms | 7.65 ms | 993.54 ms |

MXFP6 improved output-token throughput by **13.58%**, reduced mean TPOT by
10.08% and reduced P99 TPOT by 7.96%.

A separate 742-case greedy comparison measured reference-output fidelity. Full
character similarity changed by -0.57 percentage points and answer-section
similarity by -0.32 points. Neither paired 95% interval detected a significant
loss, although the intervals do not prove a strict one-percentage-point
non-inferiority bound. This is reference fidelity, not audited business-task
accuracy. The serving contract, ABBA samples and fidelity intervals are in
[`qwen35_moe_service_tp2.json`](../benchmarks/results/qwen35_moe_service_tp2.json).

## Reproduction commands

### Dense GEMM

```bash
CUDA_VISIBLE_DEVICES=0 MXFP6_AUTOTUNE=0 \
python3 benchmarks/benchmark.py \
  --library build/mxfp6_torch.so \
  --activation-input bf16 \
  --output-dtype fp16 \
  --compare-fp8 \
  --warmup 20 --iterations 100 --flush-l2-mb 0
```

The benchmark independently reports GEMM and activation quantization. Every
printed speedup uses GEMM alone. `--flush-l2-mb=0` selects warm-cache runs; use
`--flush-l2-mb=256` for an explicit cold-weight regime. Add `--check-all` to
validate every baseline output rather than the first representative shape.

The default matrix contains five Qwen3.5-27B TP2 layers per batch:

| Layer | N | K |
|---|---:|---:|
| GDN input projection | 8192 | 5120 |
| GDN output projection | 5120 | 3072 |
| Full-attention QKV/gate projection | 7168 | 5120 |
| MLP gate/up projection | 17408 | 5120 |
| MLP down projection | 5120 | 8704 |

Small-batch and quantizer microbenchmarks are available separately:

```bash
python3 benchmarks/compare_small_batch.py --flush-l2-mb=0
python3 benchmarks/quantization.py
python3 benchmarks/compare_persistent_workspace.py \
  --library build/mxfp6_torch.so \
  --json benchmarks/results/persistent_workspace.json
python3 tests/stress_persistent_workspace.py \
  --library build/mxfp6_torch.so --iterations 10000
```

### Qwen3.5 MoE layer

The real Qwen3.5 MoE layer benchmark accepts either a single process with an
emulated TP2 shard or two `torchrun` workers:

```bash
torchrun --nproc-per-node=2 \
  benchmarks/benchmark_qwen35_moe_layer.py \
  --batch-sizes 1 --mx-mode array \
  --warmup 20 --iterations 1000 --repeats 9
```

Serving artifacts include their full launch and request contracts. Reproduce
them only in a runtime with the same model, tokenizer, adapter, sampling and
scheduler configuration; the standalone wheel is not a community-vLLM loader.
