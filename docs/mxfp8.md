# SM120 MXFP8 GEMM

`mxfp6.mxfp8.gemm` computes `A[M,K] @ B[N,K].T` with E4M3 operands,
E8M0 scales per 32 values, FP32 accumulation and BF16 output. All default
paths use CUTLASS/CuTe. The MXFP8 Triton implementation and tactic IDs
100–105 have been removed.

The separate opt-in [dual activation API](mxfp8-dual.md) supplies high and
BF16-rounded residual limbs for M=1–128 at N12288/N18432, K2560. It uses a Triton activation
quantizer and native GEMM, without changing this module's default dispatch.

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

- Narrow/short K, 32<M≤64: split at M=48.
- Narrow/medium K, 16<M≤32: split at K=5120; 32<M≤64: split at M=48.
- Wide/short K, 32<M≤64: split at N=6656, then M=48 for the lower N region.
- Narrow/medium K: split at K=5120 for 64<M≤128 and 128<M≤256.
- Narrow/medium K, 64<M≤128: also split at M=112. For M≤112 and
  K>5120, three K splits improve work balance.
- Wide/short K, 64<M≤256: split at N=6656.
- Wider/short K, 16<M≤32 and 64<M≤256: split at N=11264; for 64<M≤128 and
  N>11264, also split at M=112. For 16<M≤32 and N>11264, split at K=2048.
- Widest/short K, 128<M≤256: split at M=224.

There are **71 configuration intervals**. Each selects
`(tactic, splits, swizzle, sms)`. All use `sms=0`, so the scheduler uses the
device's available SM count. Configurations specialized to an exact K
are excluded from the interval policy.

M≤32 in these six regions uses one CuTe TMA kernel template, including
long-K, narrow-N projections. The selected instantiations vary tile dimensions,
pipeline depth and warp layout; no exact-K specialization is required.
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

## Unified small-batch kernel

`small_batch.cuh` implements tactics 1240–1250 using a single value/scale
pipeline and output layout. It covers all ten Qwen3.5-2B/4B projection shapes
at every integer M1–32 through the interval policy, plus selected M33–64 regions. TMA moves the E4M3
values, `cp.async` moves packed E8M0 scales, and CuTe's SM120 block-scaled
`16x8x32` MMA accumulates in FP32. Invalid rows and K tails are masked.
The pipeline does not issue TMA reads past the last K slab and drains
pending scale copies before reusing shared memory for output.

For narrow projections, tactic 1240 partitions each K slab among four
warps in the same CTA. The epilogue sums their FP32 partial results
and rounds once to BF16. This adds parallel work without a second kernel,
global partial-result buffer or inter-CTA semaphore. Other instantiations
use the same template with one K partition. Tactic 1249 uses two K partitions with a 32×16 weight/batch tile for
long-K M17–32 projections. Tactic 1250 uses a 64×32 tile and three
pipeline stages to reuse weights across more batch rows at M49–64.
The M33–48 narrow and smaller wide projections reuse tactics 1241/1243.
All support PDL and changing
inputs/scales during CUDA Graph replay. FP32 reduction order can differ
from the previous Stream-K implementation.

The limiting distinction is between instruction support and CUTLASS's
default thread organization: SM120 exposes a warp-level MMA atom, while
the generic warp-specialized builders choose larger producer/consumer
layouts. The unified kernel uses the existing CuTe MMA/TMA primitives;
this change needs no additional CUTLASS patch. Shared-memory capacity and
TMA layout/alignment requirements still apply to every instantiation.
See the pinned CUTLASS [SM120 builder's AtomLayoutMNK selection](https://github.com/NVIDIA/cutlass/blob/e6233cbac5d7c7a865c19c91cd684ceece19513c/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl)
and [warp-level MMA primitive](https://github.com/NVIDIA/cutlass/blob/e6233cbac5d7c7a865c19c91cd684ceece19513c/include/cute/arch/mma_sm120.hpp).

`tests/test_mxfp8_small_batch.py` covers all 640 model shape/batch pairs
against independently decoded FP32 operands, plus unseen partial tiles
and changing-input/scale/zero-input graphs with PDL off/on. The existing
all-tactic suites also exercise all eleven small-batch tactics explicitly.

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

All Python GEMM entrypoints (`gemm`, `gemm_from_float`, `gemm_w8a8`,
`gemm_packed`) and `prepare`/`warmup` accept `use_pdl=True`. This also applies
to explicit `W8A8Config` choices and private workspace launches. Defaults
remain `False`. Native `gemm` and `gemm_from_float` accept the same optional
flag; their existing positional arguments remain valid.

Native callers can also opt into PDL with
`torch.ops.mxfp8_sm120.gemm_from_float_pdl(input, b, sb)`, or chain
`torch.ops.mxfp6.quantize_mxfp8_pdl(input)` with
`torch.ops.mxfp8_sm120.gemm_pdl(a, b, sa, sb)`. Load both libraries first;
the GEMM operands are E4M3 matrices with packed E8M0 scale tensors.
The explicit PDL entrypoints have no M ≤ 32 restriction. The quantizer
signals an early launch and the selected GEMM
waits before reading its output. The CUTLASS routes and custom cache-hint,
occupancy, irregular, TMA256, eightwarp, vector-store and TMA eightwarp routes
cover default dispatch across decode and prefill batches. Every registered
tactic now supports the flag, including GEMV, cute, specialized, TMA4,
TMA fewwarp/oddwarp/oddrotate/scales, register pipeline and pair-LDSM paths.
Both dual kernel families accept `mxfp8_dual.gemm_out(..., use_pdl=True)`
for every M1–128 and both projection widths.
The quantizer waits before padding stores as well as input reads: a scale
allocation can reuse the preceding GEMM's temporary Stream-K workspace.
PDL latency gains depend on shape
and available GPU resources; these operators do not imply a serving speedup.

Run `tests/test_mxfp8_pdl_all_paths.py` to scan every registered tactic at
dynamic K512 and fixed K2048/K2560, test Python/prepared entrypoints and
dual→quantization→single dependencies. `tests/test_mxfp8_dual.py` compares
PDL Graph replay with ordinary eager results for all 256 dual shapes.
On CUDA 13, the path suite also inspects captured graph edge metadata for
every tactic and both dual families: PDL off has no programmatic edges,
and PDL on creates them. This catches a flag silently falling back to an
ordinary launch even when numerical comparisons would still pass.

`benchmarks/benchmark_mxfp8_pdl.py` measures paired Graph timings using
checkpoint weights, both for repeated projections and for a rotating
multi-layer MLP chain whose weight footprint exceeds L2. It requires
`safetensors`, and the chain additionally requires a matching vLLM Mach
source tree. `benchmark_mxfp8_pdl_serving.py` compares identical serving
configurations with PDL off, M ≤ 32, and an expanded deployment limit.
Its token/concurrency workload is explicit and differs from the public
six-point benchmark. Native API support and a profitable deployment limit
are separate questions; measure both on the target workload.
The [PDL measurements](#pdl-measurements) below cover service, dependency-chain and isolated-projection timing.

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
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --out benchmarks/results/mxfp8_current --shard 0 --shards 2 --check-graphs
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --out benchmarks/results/mxfp8_current --shard 1 --shards 2 --check-graphs
python3 benchmarks/summarize_mxfp8.py benchmarks/results/mxfp8_current
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=python python3 -m pytest \
  tests/test_mxfp8.py tests/test_mxfp8_api.py tests/test_mxfp8_benchmark.py tests/test_model_gemm_totals.py -q
```

The default workload covers Qwen3.5-2B and Qwen3.5-4B, TP=1, ten N/K
pairs and 24 batches. `--shapes` accepts a JSON list of `[M,N,K]` for other
workloads. `--check-graphs` changes both operands and their scales, then
checks a zero-input replay. Model totals weight the measured projection
latencies by actual layer/call counts; they are estimates of quantized
GEMM cost, not full-model inference latency.

## Decode results (October 14, 2026)

The RTX 5090 sweep covers Qwen3.5-2B/4B TP1, ten N/K pairs and batches
**1, 2, 4, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64**.
Each measurement times one GEMM in a CUDA Graph after an 8 GB L2 eviction.
Quantization, eviction and host overhead are excluded. Four balanced
cyclic backend orders each collect 30 samples; results are medians of
trial means. These are kernel measurements, not serving throughput.

| Baseline | Geometric mean speedup | Points faster | Largest remaining slowdown |
|---|---:|---:|---:|
| FlashInfer MXFP8 | 1.228× | 160/170 | 0.91% |
| vLLM block FP8 | 1.247× | 134/170 | 3.09% |

Two tuning stages contribute to the current policy:

- [October 13](../benchmarks/results/mxfp8_decode_20261013.json): K partition,
  tile and pipeline changes improve 24 points by 1.084× on average, with
  a largest latency increase of 0.85%. Only two new instantiations are
  retained: tactics 1249 (32×16×512, two K partitions, three stages) and
  1250 (64×32×256, three stages).
- [October 14](../benchmarks/results/mxfp8_decode_20261014.json): N>11264,
  K>2048, M17–32 in the wider/short-K region switches to existing tactic
  1247 (64×16×256, one stage). More independent CTAs and fewer K-loop
  iterations improve the four measured points by 1.010×, without a
  measured regression. Independent randomized confirmation wins 27/28
  trial comparisons (1.018× average).

| Changed batch, N12288/K2560 | Previous policy (µs) | Current policy (µs) |
|---|---:|---:|
| 20 | 24.670 | 24.621 |
| 24 | 25.195 | 24.751 |
| 28 | 24.986 | 24.720 |
| 32 | 24.917 | 24.685 |

M24/M28 now beat FlashInfer; the changed-point vLLM win count remains
1/4. The frozen policy measures 161/170 FlashInfer wins and 135/170 vLLM
wins in the final sweep. The other 166 configurations are identical, but
independent timing produces flips near parity and an apparent maximum
3.72% increase at an unchanged point. The all-point paired ratio is
0.9995×; neither that ratio nor the total win-count difference establishes
an aggregate regression or improvement. The formerly worst point,
M32/N12288/K2048, retains its configuration and measures 20.571 µs versus
vLLM's 20.435 µs (0.66% slower); its earlier 5.02% gap did not reproduce.

Artifacts retain raw trial means, configurations, environment/source/library
hashes and model-weighted totals. They describe the measured builds before
cleanup. `runtime_tactic_map` maps the two retained experimental IDs to
their current IDs; the underlying kernel templates and intervals are
unchanged. Discarded candidates and duplicate search logs are omitted.
The [initial 120-point study](../benchmarks/results/mxfp8_small_batch_20261013.json)
records the original unified-template comparison.

Reproduce the final workload and frozen-policy comparison:

```bash
python3 - <<'PYTHON'
import json
from pathlib import Path
d = json.loads(Path('benchmarks/results/mxfp8_decode_20261014.json').read_text())
aliases = d['runtime_tactic_map']
def runtime_config(config):
    return [aliases.get(str(config[0]), config[0]), *config[1:]]
Path('/tmp/decode_shapes.json').write_text(json.dumps([[r['m'], r['n'], r['k']] for r in d['points']]))
Path('/tmp/decode_before.json').write_text(json.dumps({'configs': {
    f"{r['m']},{r['n']},{r['k']}": runtime_config(r['reference_config']) for r in d['points']}}))
PYTHON
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --out benchmarks/results/mxfp8_decode_recheck --shapes /tmp/decode_shapes.json \
  --reference-dispatch /tmp/decode_before.json --reference-bench /data/lxy/gemm_bench.py \
  --tests 30 --trials 4 --check-graphs
```

Before cleanup, 401 kernel/API/Graph/PDL/model-weighting tests passed,
including 640 independently decoded FP32 comparisons, and all 170
changing-input/scale/zero-input benchmark graph checks passed. Twelve
changed-policy and K-tail graph cases with PDL off/on passed memcheck
(zero errors) and racecheck (zero hazards). Cleanup removes ten unused
instantiations and their twenty diagnostic tests. Rebuilding after cleanup
and rerunning the retained suites passed all 381 tests; both retained
additions remain covered.

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

## Fixed BF16 BA projection

`mxfp8.bf16_ba_m16(x, weight)` computes BF16 `X @ W.T` with FP32 MMA accumulation and one BF16 output rounding. It requires contiguous, 16-byte-aligned CUDA BF16 `X[16,2048]` and `W[32,2048]` on SM120 and returns `Y[16,32]`. The call uses `mxfp8_torch` without quantization. Warm it before CUDA Graph capture and update captured inputs in place. Its fixed reduction order is not guaranteed bitwise equal to an arbitrary BF16 GEMM backend.

## PDL measurements

October 8, 2026; RTX 5090, CUDA 13.0, PyTorch 2.13, vLLM 0.29. The Qwen3.5-4B single-activation MXFP8 service comparison changed only PDL off/on and its physical-row limit. Each setting used two fresh lifecycles in forward/reverse order; the table reports output tokens/s.

| Concurrency | PDL off tok/s | M≤32 tok/s | M≤128 tok/s | M≤32 vs off | M≤128 vs off | M≤128 vs M≤32 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 852.31 (852.21, 852.41) | 870.01 (870.75, 869.27) | 869.82 (870.08, 869.55) | +2.08% | +2.05% | -0.02% |
| 16 | 2769.84 (2769.67, 2770.01) | 2813.56 (2816.81, 2810.31) | 2814.94 (2813.67, 2816.20) | +1.58% | +1.63% | +0.05% |
| 32 | 4203.76 (4195.13, 4212.39) | 4227.66 (4217.61, 4237.70) | 4239.30 (4230.01, 4248.58) | +0.57% | +0.85% | +0.28% |
| 64 | 5178.57 (5171.38, 5185.75) | 5149.25 (5155.80, 5142.70) | 5195.95 (5188.78, 5203.13) | -0.57% | +0.34% | +0.91% |
| 128 | 6075.12 (6042.47, 6107.77) | 6048.58 (6038.18, 6058.98) | 6071.19 (6050.45, 6091.93) | -0.44% | -0.06% | +0.37% |

The workload used 1024 input and 512 output tokens, greedy sampling, TP1, FP8 KV, Triton attention and an NVFP4 candidate head with BF16 refinement. Limits were maxseq128, maxlen4096, maxbatch2048 and memory utilization 0.80. All 2,928 scored requests completed with matching token contracts and response hashes. The M≤128 result was +2.05%/+1.63% at c4/c16 and +0.34%/−0.06% at c64/c128. Differences near the two-run spread are inconclusive. This is a PDL increment on the recorded service configuration.

An eight-layer rotating MLP chain (557 MiB of real weights, synthetic BF16 inputs) used GemmaNorm → gate/up → SwiGLU/quantization → down with planned workspaces. Timings are medians of nine randomized paired Graph trials; GPU gain is off/on−1.

| M | PDL off µs | PDL on µs | GPU speedup |
|---:|---:|---:|---:|
| 1 | 426.50 | 411.70 | +3.59% |
| 16 | 440.12 | 426.29 | +3.24% |
| 32 | 437.25 | 424.04 | +3.12% |
| 33 | 450.66 | 436.43 | +3.26% |
| 48 | 453.32 | 443.60 | +2.19% |
| 64 | 456.70 | 442.16 | +3.29% |
| 65 | 491.11 | 475.44 | +3.30% |
| 96 | 504.37 | 486.09 | +3.76% |
| 128 | 511.95 | 496.95 | +3.02% |
| 192 | 624.84 | 619.93 | +0.79% |
| 256 | 691.35 | 680.50 | +1.60% |
| 512 | 1297.20 | 1295.56 | +0.13% |
| 1024 | 2255.31 | 2233.80 | +0.96% |
| 2048 | 4417.13 | 4421.84 | -0.11% |

Repeated isolated projections can regress: at M64, down was 20.6% slower with PDL while the rotating chain was 3.3% faster. The [raw artifact](../benchmarks/results/mxfp8_pdl_batch_validation.json) retains all 56 projection points, 14 chain points, source/library hashes and service samples. Changed-input Graph replay produced bitwise-equal PDL on/off outputs; sanitizer checks recorded zero errors.

Use [the GPU benchmark](../benchmarks/benchmark_mxfp8_pdl.py) and [the HTTP benchmark](../benchmarks/benchmark_mxfp8_pdl_serving.py) to measure the target shape and deployment limit. Native PDL APIs and buffer contracts are specified above.
