# MXFP8 W8A8 on SM120

`mxfp6.gemm_mxfp8` computes `A[M,K] @ B[N,K].T` from E4M3 operands and
one E8M0 scale per 32 values, with FP32 accumulation and BF16 output.
Scales use the padded 128x4 swizzled layout accepted by this repository's
W6A8 path and produced by FlashInfer's MXFP8 quantizer.

The kernel portfolio reuses the existing CUTLASS SM120 configuration classes
and runtime patches. It includes transposed small-batch GEMMs, different tile
sizes and pipeline stages, persistent scheduling, Stream-K with private
workspace, and direct output stores. Small-batch CUDA GEMV candidates decode
FP8 values directly. Runtime execution does not call Humming, FlashInfer GEMM,
or vLLM. Quantization is outside the GEMM.

The measured shape table targets RTX 5090 (170 SMs). It chooses a configuration;
it does not guarantee every shape passes the requested speedup. The acceptance
report retains failures. Other SM120 devices may require retuning, especially
configurations with an explicit SM count.

## Wide-N / short-K update

The default dispatch now replaces **47 of the 144** shapes in the six requested
N/K groups. On GPUs 4/5, the promoted shapes have a **1.082x** geometric mean
speedup over the old configuration; across all 144 shapes the mean is **1.026x**.
Using freshly measured same-GPU baselines, the original acceptance count changes
from **80/144 to 90/144**. The full performance requirement remains unmet.
No configuration was promoted for `(N,K)=(18432,2560)`.

The [optimization report](../benchmarks/results/mxfp8_short_k/REPORT.md) lists
all shapes, before/after times, and fresh FlashInfer/block-FP8/Humming baselines.
[CSV](../benchmarks/results/mxfp8_short_k/comparison.csv) and raw interleaved
samples are included. Promotion uses both search and validation gates; the
report describes those gates explicitly. Do not combine these counts with the
older GPU-6/7 measurements below.

The [rolling FlashInfer/block-FP8 comparison](../benchmarks/results/mxfp8_decode_validation/BASELINES.md)
always merges the latest deployed measurements per shape and labels their
source/GPU. `summarize_mxfp8_short_k.py --write-dispatch` refreshes this file
automatically; `summarize_mxfp8.py` retains registered updates through
`baseline_updates.json`, so regenerating it does not restore stale measurements.

The selected approach uses independent 4-warp CTAs, K tiles of 128/256, native
block-scaled MMA, and direct BF16 output. Larger K tiles reduce the number of
pipeline iterations; small batches may use transposed MMA tiles. The public
input layout and precision are unchanged. The experiments also tested smaller
CUTLASS tiles and higher CTA grid limits; only measured winners are dispatched.

Reproduce the search and frozen-candidate validation with
`benchmarks/optimize_mxfp8_short_k.py --phase search|validate`, using `--shard 0`
on GPU 4 and `--shard 1` on GPU 5. Validation takes `--plan` pointing to
`benchmarks/results/mxfp8_short_k/plan.json`. Use a new `--out` directory to
preserve the recorded results. The original dispatch snapshot is retained in
`benchmarks/results/mxfp8_decode_validation/dispatch.json`.

## Initial validation, before the short-K update

The initial frozen dispatch table passed **93/240 shapes**: **80/188 normal** and
**13/52 outliers**. **147 shapes fail**, so the requested performance goal
has not been achieved. These are independent validation measurements after
configuration selection, not the best observed tuning timings.

See the [passing/failing shape lists and all timings](../benchmarks/results/mxfp8_decode_validation/REPORT.md)
and [CSV](../benchmarks/results/mxfp8_decode_validation/comparison.csv).
The validation directory includes raw samples, a dispatch snapshot, source
hashes, and baseline metadata. Baselines were measured in the preceding
same-GPU run and reused for this validation. The CMake-built extension passed
all 19 correctness tests, including changed inputs during graph replay and
concurrent execution with separate workspaces.

## Build

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
# Source-tree use compiles a separate extension on first load.
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=python python3 -c \
  'from mxfp6.mxfp8 import load_library; load_library()'

# Alternatively, build the CMake target into build/mxfp8_torch.so.
cmake -S . -B build -DPython3_EXECUTABLE="$(command -v python3)" \
  -DMXFP6_BUILD_STANDALONE=OFF -DBUILD_TESTING=OFF
cmake --build build --target mxfp8_torch -j3
```

The wheel build also includes `mxfp8_torch.so`. Kernel execution requires
PyTorch, Triton (for tactics 100–105), and SM120; benchmark-only dependencies include FlashInfer,
Humming kernels, and vLLM.

## Use

```python
import flashinfer
from mxfp6 import gemm_mxfp8, prepare_mxfp8

a, sa = flashinfer.mxfp8_quantize(activation, is_sf_swizzled_layout=True)
b, sb = flashinfer.mxfp8_quantize(weight, is_sf_swizzled_layout=True)
y = gemm_mxfp8(a, b, sa, sb)

# Repeated calls / graph capture: retain output and private workspace.
run = prepare_mxfp8(a, b, sa, sb)
y = run()
```

Like the MXFP6 API, both functions load their extension automatically on first
use. No explicit `load_library()` call is required. That function remains an
optional preloading hook. For CUDA graphs, call `prepare_mxfp8` before capture;
it also compiles a selected Triton kernel for the exact shape. Warm the prepared
call on a side stream before capture, following normal PyTorch graph usage.

The existing repository quantizer can also supply operands:

```python
from mxfp6 import quantize_mxfp8
qa = quantize_mxfp8(activation)
qb = quantize_mxfp8(weight)
run = prepare_mxfp8(qa.dequantized_values(), qb.dequantized_values(),
                   qa.scales, qb.scales)
```

Despite its name, `dequantized_values()` only views the FP8 payload bytes;
it does not apply scales or convert to BF16.

Inputs must be contiguous, 16-byte aligned, and on one SM120 device.
M is positive; N and K are positive multiples of 128. Weight storage is
`[N,K]`. Optional `out` is contiguous BF16 `[M,N]` and must not alias inputs.
The operation uses the current PyTorch CUDA stream.

`gemm_mxfp8` allocates/initializes Stream-K workspace if none is supplied.
`prepare_mxfp8` allocates private workspace once and uses self-resetting
barriers. Each concurrent invocation needs separate output/workspace.
Input contents may change between graph replays; no output values are cached.
Explicit configuration options are `tactic`, `splits`, `swizzle`, and `sms`
(0 means all physical SMs). `splits` applies to Stream-K.

## Acceptance method: hot launch + cold weight cache

The benchmark directly imports `gemm_bench.py::bench_kineto`.
It passes a **pre-captured single-GEMM graph's `replay`** as the test function:

1. Quantize inputs and prepare output/workspace before capture.
2. Warm the kernel and capture one invocation.
3. Verify the graph contains exactly one GPU GEMM kernel.
4. Before each replay, use the reference script's **8 GB L2 eviction**.
5. Record only the GEMM kernel's profiler duration. Eviction and host launch
   gaps are excluded. Default sampling is 30 active measurements per trial,
   with three trials; report the median of trial means.

This reproduces the requested hot-launch/cold-weight condition for a GEMM.
It does not measure a full LLM's end-to-end decode latency.

The original manifest supplies 240 shapes and 52 fixed outliers. For each
shape, Humming (default heuristics), FlashInfer (exact-M autotune), and vLLM
block FP8 are **remeasured on the same GPU using the same procedure**.
Normal points require speedup **>1.05x Humming**; outliers require
**>1.1x the faster of FlashInfer/vLLM**. Original hot-weight timings are not
used as the acceptance baseline.

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --tune --tune-sms --shards 2 --shard 0 --out benchmarks/results/mxfp8_decode
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --tune --tune-sms --shards 2 --shard 1 --out benchmarks/results/mxfp8_decode
python3 benchmarks/summarize_mxfp8.py benchmarks/results/mxfp8_decode \
  --write-dispatch python/mxfp6/mxfp8_dispatch.json
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=python python3 -m pytest tests/test_mxfp8.py -q
```

Run the two shards concurrently on their separate GPUs. `--candidate-results`
can reuse a previous search; `--config` validates a frozen dispatch table
without searching. `--reuse-baselines` can reuse the measured baselines from
a previous run with the same GPU, input sequence, and reference script;
metadata records the reused files and hashes. `--reference-bench` and `--baseline` override external
source paths. Metadata includes source hashes, GPU identity, and sampling
parameters. Raw data retain candidate timings, all baseline samples,
correctness errors, and selected configurations.

To independently validate the shipped table and freshly measure all baselines,
run both shards with a new output directory (without `--tune`):

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --config python/mxfp6/mxfp8_dispatch.json --shards 2 --shard 0 \
  --out benchmarks/results/mxfp8_recheck
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=python python3 benchmarks/benchmark_mxfp8.py \
  --config python/mxfp6/mxfp8_dispatch.json --shards 2 --shard 1 \
  --out benchmarks/results/mxfp8_recheck
python3 benchmarks/summarize_mxfp8.py benchmarks/results/mxfp8_recheck
```

## Candidate families

- 0–11, 17–29, 31–32: static CUTLASS kernels with different orientations,
  tiles, and pipeline stages.
- 12–16, 39–40, 42, 51, 57: Stream-K with private reusable workspace.
- 33–36: CUDA GEMV with 1/2/4/8 activation rows per weight load.
- 37–38, 41: additional wide/narrow tiles.
- 43–50: SM120 mainloops with a register-to-global BF16 epilogue.
- 52–56: additional transposed 64/128-column tiles.
- 58–65: two-stage direct-output kernels, including static grids with up to
  two CTA slots per physical SM.
- 66–69: experimental 32-row weight tiles; slower in the short-K checks.
- 100–105: independent 4-warp Triton CTAs with block-scaled MMA and direct
  BF16 stores. Selected by the measured shape table; `prepare_mxfp8` compiles
  the exact shape before graph capture.

Tactic 30 was removed because its forced stage count exceeded shared memory.
Unsupported shapes/configurations raise an error.
