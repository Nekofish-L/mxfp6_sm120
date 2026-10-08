# SM120 MXFP8 GEMM

`mxfp6.mxfp8.gemm` computes `A[M,K] @ B[N,K].T` with E4M3 operands,
E8M0 scales per 32 values, FP32 accumulation and BF16 output. All default
paths use CUTLASS/CuTe. The MXFP8 Triton implementation and tactic IDs
100–105 have been removed.

## Shape scheduling

C++ selects **M/N/K intervals**, without an exact-shape lookup or
runtime autotuning. The build generates the native decision tree from
[mxfp8_dispatch.json](../python/mxfp6/mxfp8_dispatch.json).
Rebuild the extension after changing this file; Python does not read the
policy or route individual kernel families at runtime.

The first matching N/K region is selected:

| Region | N | K |
|---|---:|---:|
| Narrow, short K | ≤4096 | ≤3072 |
| Narrow, medium K | ≤4096 | 3072–7168, lower bound excluded |
| Narrow, long K | ≤4096 | >7168 |
| Wide, short K | 4096–9216, lower bound excluded | ≤3072 |
| Wider, short K | 9216–14336, lower bound excluded | ≤3072 |
| Widest, short K | >14336 | ≤3072 |

Each region has nine M intervals with inclusive upper bounds:
**8, 16, 32, 64, 128, 256, 512, 1024, 2048**. Thus an unseen batch such as
M=17 uses the same interval as M=20–32.

Additional splits limit regressions in the broad intervals:

- Narrow/medium K: split at K=5120 for M≤8, 64<M≤128 and 128<M≤256.
- Narrow/medium K, 64<M≤128: also split at M=112. For M≤112 and
  K>5120, three K splits improve work balance.
- Wide/short K, 64<M≤256: split at N=6656.
- Wider/short K, 64<M≤256: split at N=11264; for 64<M≤128 and
  N>11264, also split at M=112.
- Widest/short K, 128<M≤256: split at M=224.

There are **65 configuration intervals**. Each selects
`(tactic, splits, swizzle, sms)`. All use `sms=0`, so the scheduler uses the
device's available SM count. Configurations specialized to an exact K
are excluded from the interval policy.

Small-M, short-K regions mostly use direct-output CUTLASS or CuTe tiles;
long-K, narrow-N regions use Stream-K to supply enough parallel work.
Larger M uses larger tiles and fewer splits. Selection accounts for the
combined measurements of the shapes in a region, rather than copying a
single shape's winner.

Outside the calibrated regions:

- M>2048 uses the general tactic 79.
- N>4096 and K>3072 uses Stream-K tactics 12–16 according to M, with 2/4
  splits for small M and long K, otherwise one split.

The rules accept unseen dimensions within the input contract. Performance
outside the measured range remains untuned; interval coverage alone does
not establish optimal performance on every shape or other GPU models.

## Input contract and use

Requires SM120, CUDA and PyTorch. M must be positive; N and K must be
positive multiples of 128. Inputs and scales must be contiguous on one
CUDA device with 16-byte-aligned storage. Scale storage uses the padded
CUTLASS 128×4 swizzle. Weights are quantized once; activations may be FP16/BF16 or a shared
`MXFP8Tensor`. The public entrypoints follow MXFP6:

```python
import torch
import mxfp6
import mxfp6.mxfp8 as mxfp8

# x: [M, K], weight: [N, K], both CUDA FP16/BF16
w6 = mxfp6.quantize_mxfp6(weight)
w8 = mxfp8.quantize_mxfp8(weight)
y6 = mxfp6.gemm(x, w6, out_dtype=torch.bfloat16)
y8 = mxfp8.gemm(x, w8, out_dtype=torch.bfloat16)

# Shared activation type and quantizer; skip activation conversion in GEMM.
qa = mxfp8.quantize_mxfp8(x)
y8 = mxfp8.gemm(qa, w8)
```

Both modules expose `gemm`, `gemm_from_float`, `gemm_packed`, `warmup`,
`load_library` and the same workspace planning functions. The quantized
entrypoints are `gemm_w6a8` and `gemm_w8a8`. MXFP8 currently supports
`alpha=1` and BF16 output only; other values raise an error. MXFP6 retains
its FP16 default and supports FP16/BF16 output. Pass `out_dtype` explicitly
when sharing calling code.

Both follow Python wrapper → native C++ dispatch → kernel. Floating
activations use the same native MXFP8 quantizer. MXFP6 retains its PDL
quantization/GEMM launch; MXFP8 launches quantization and GEMM in stream
order by default. All workspace layout selection and allocation happen in C++.

Native callers can opt into PDL with
`torch.ops.mxfp8_sm120.gemm_from_float_pdl(input, b, sb)`, or chain
`torch.ops.mxfp6.quantize_mxfp8_pdl(input)` with
`torch.ops.mxfp8_sm120.gemm_pdl(a, b, sa, sb)`. Load both libraries first;
the GEMM operands are E4M3 matrices with packed E8M0 scale tensors.
For M ≤ 32, the quantizer signals an early launch and the selected GEMM
waits before reading its output. The CUTLASS routes and custom cache-hint,
occupancy, irregular, TMA256 and eightwarp routes cover all current
small-batch default tactics. M > 32 uses ordinary launches. The Python
wrappers retain their existing defaults. PDL latency gains depend on shape
and available GPU resources; these operators do not imply a serving speedup.

The backends share the workspace implementation and Python helper, with
independent pools. Plan all expected shapes, freeze capacity, then warm
each stream before graph capture:

```python
backend = mxfp8  # mxfp6 uses the same workflow with w6
backend.begin_workspace_planning()
backend.warmup(qa, w8, out_dtype=torch.bfloat16)
backend.finalize_workspace_planning()
backend.warmup(qa, w8, out_dtype=torch.bfloat16)
```

For multiple shapes, warm each before finalizing. Each stream gets its own
workspace lane; frozen capacity cannot be resized. A larger unplanned
layout falls back to temporary workspace in eager execution and raises
during capture. `workspace_stats()` and `workspace_barriers_zero()` expose
pool state for diagnostics.

Alternatively, `mxfp8.prepare(qa, w8)` returns a callable with private
output/workspace allocated by C++. Warm it on a side stream before graph
capture, and retain it for the graph's lifetime. Inputs and scales may
change in place between replays. Concurrent prepared calls need separate
prepared objects. `W8A8Config` supports explicit diagnostic overrides;
`select_config(m, n, k)` inspects the native policy. Invalid IDs raise an
error. The old `gemm_mxfp8`, `prepare_mxfp8` and raw `mm` APIs were removed.

## Build and configuration limits

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
cmake -S . -B build -DMXFP6_BUILD_STANDALONE=OFF -DBUILD_TESTING=OFF
cmake --build build --target mxfp6_torch mxfp8_torch -j3
```

Build both extensions to use the shared quantizer. Extensions load
automatically on first use from the installed package or build directory;
`MXFP6_LIBRARY_PATH` and `MXFP8_LIBRARY_PATH` select explicit binaries.

See the [CUTLASS patch notes](../patches/cutlass/README.md). The patches
address software restrictions: narrow-N warp layouts, single-stage
mainloops, static-shape TMA descriptors and reusable Stream-K barriers.
A tile exceeding available shared memory remains a hardware resource
limit. N/K alignment above is the current implementation contract.

MXFP6 tile/stage/scheduler choices can supply candidates for MXFP8, but
must be re-instantiated for E4M3 and remeasured. Packed E3M2 weights,
MXFP6 binaries and tactic IDs cannot be reused directly. Rankings from
warm-cache MXFP6 measurements also need cold-cache validation here.

## Benchmark and validation

The benchmark compares current MXFP8, FlashInfer MXFP8 CUTLASS and vLLM
block FP8 on identical source tensors. It times one GEMM per CUDA graph,
with an 8 GB L2 eviction before replay. Eviction, quantization and host
launch overhead are excluded. Default sampling is three interleaved
trials of 30 measurements, reported as the median of trial means.

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --out benchmarks/results/mxfp8_current --shard 0 --shards 2 --check-graphs
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --out benchmarks/results/mxfp8_current --shard 1 --shards 2 --check-graphs
python3 benchmarks/summarize_mxfp8.py benchmarks/results/mxfp8_current
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=python python3 -m pytest \
  tests/test_mxfp8.py tests/test_mxfp8_api.py tests/test_mxfp8_benchmark.py tests/test_model_gemm_totals.py -q
```

The default workload covers Qwen3.5-2B and Qwen3.5-4B, TP=1, ten N/K
pairs and 24 batches. `--shapes` accepts a JSON list of `[M,N,K]` for other
workloads. `--check-graphs` changes both operands and their scales, then
checks a zero-input replay. Model totals weight the measured projection
latencies by actual layer/call counts; they are estimates of quantized
GEMM cost, not full-model inference latency.

## Approximate performance

RTX 5090 / SM120, physical GPUs 4, 5; 240 shapes. Single-GEMM CUDA graph, 8 GB L2 eviction before replay; only GPU kernel time is counted. Quantization, host overhead and eviction are excluded.

FlashInfer uses MXFP8 `mm_mxfp8(..., backend="cutlass")` and exact-batch autotuning. vLLM uses block FP8 `cutlass_scaled_mm`. MXFP8 inputs share E4M3 values and E8M0/32 scales; vLLM quantizes the same source tensors with 1×128 activation / 128×128 weight scales.

Speedup = baseline latency / current latency. Figures are approximate and describe this GPU and cache policy.

| Baseline | Geometric mean speedup | Shapes faster |
|---|---:|---:|
| FlashInfer MXFP8 | 1.14× | 206/240 |
| vLLM block FP8 | 1.26× | 207/240 |

## Model-weighted GEMM estimates

TP=1, all 24 batches from 1 to 2048. The six main quantized projections are weighted by their actual layer/call counts: 96 GEMMs for Qwen3.5-2B and 128 for Qwen3.5-4B. These are sums of microbenchmarks, not full-model inference latency. BF16 projections, lm_head, attention, quantization and communication are excluded.

The oracle chooses the fastest of current MXFP8, FlashInfer MXFP8 and vLLM block FP8 per shape before weighting. A mixed-backend model has not been implemented.

| Model | vs FlashInfer across batches | vs vLLM across batches | Largest gap to per-shape oracle |
|---|---:|---:|---:|
| Qwen3.5-2B | 1.01–1.21× | 1.15–1.66× | 0.9% (12 µs, batch 64) |
| Qwen3.5-4B | 1.02–1.22× | 1.20–1.59× | 1.0% (29 µs, batch 4) |

## Range scheduling versus the former point table

Both configurations are measured on the same GPU for each shape. Positive changes mean more time.

| Model | Weighted latency change across batches | Worst absolute increase |
|---|---:|---:|
| Qwen3.5-2B | -3.2% to +1.7% | 41 µs |
| Qwen3.5-4B | -3.2% to +1.4% | 64 µs |

Changing-input/scale and zero-input graph checks passed for 240/240 measured shapes.

The final policy has 65 M/N/K configuration intervals and no MXFP8 Triton path. All 72 additional unseen shapes passed changing-input/scale and zero-input graph checks. Detailed search and per-shape records were removed as requested.

## Native dispatch and shared workspace validation

The C++ refactor preserves all 65 intervals and all 27 existing MXFP8
GPU kernel binaries. On GPUs 4/5, cold-cache comparison with the preceding
implementation measured about **+0.25%** aggregate MXFP8 latency across
240 shapes and **−0.2%** across nine representative MXFP6 shapes. Retesting
15 outliers found no repeatable regression; changes were within run noise.
These figures cover GPU GEMM time, excluding Python/host overhead.

Validation passed 232 API/kernel/benchmark tests, all 240 changing-input,
scale and zero-input graph checks, and the FlashInfer/vLLM benchmark smoke
checks. MXFP6 also passed 14 conversion/GEMM/workspace regression groups,
10,000 random replays and 1,000 dual-stream iterations across 28 boundary
batch sizes, with zero workspace fallbacks. Both backends' native pools
remain independent.
