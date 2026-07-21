# SM120 MXFP6 GEMM

Native mixed-precision GEMM for NVIDIA Blackwell GeForce SM120:

```text
D[M, N] = A[M, K] @ B[N, K].T
```

Persistent weights are bit-packed E3M2 with one UE8M0 scale per 32 values.
FP16 or BF16 activations are dynamically mapped 16-to-8 into E4M3/UE8M0 and
consumed by a native CUTLASS W6A8 kernel. Accumulation is FP32 and output is
FP16. Production dispatch is implemented entirely in this repository and does
not call Humming.

The 16-to-8 activation mapping is intentional. It removes repeated
activation-side six-bit unpacking from the MMA mainloop, while persistent
weights retain their six-bit storage and bandwidth advantage. The repository
also keeps a packed-MXFP6 activation compatibility API for existing callers.

## Performance

The following results were measured on one NVIDIA GeForce RTX 5090 using:

- PyTorch `2.11.0+cu130`, CUDA 13.0;
- vLLM `0.20.3.dev4+ge38d84f55.d20260715`;
- pinned Humming revision `694298e9`;
- Qwen3.5-27B TP2 linear shapes, five `(N,K)` pairs per batch;
- BF16 activation input, warm cache, 20 warmups and 100 measured iterations;
- checked-in static dispatch with runtime autotuning disabled.

All ratios use standalone GEMM kernel time only:

```text
speedup = reference GEMM latency / native MXFP6 GEMM latency
```

Activation quantization, weight preparation, Humming JIT compilation and host
gaps are excluded from every ratio.

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
at M=64 and M=96. Repeated `--check-all` runs reproduce 10.8x and 21.1x batch
geometric means. These two batches are shown but excluded from the comparable
Humming overall value. Including them gives a raw 50-shape overall of 2.005x.

‡ Geometric mean over the remaining 40 Humming comparisons. A dedicated BS32
static-dispatch rerun measured 1.278x with BF16 input and 1.277x with FP16.

The vLLM baseline is block-scaled W8A8, while this project keeps weights at six
bits. It is a kernel-performance comparison, not a claim that the numerical
formats or weight footprints are identical. Full dispatch and measurement
metadata are recorded in
[`benchmarks/results/native_w6a8_dispatch.json`](benchmarks/results/native_w6a8_dispatch.json).

## Supported shapes

The public operator accepts:

- `M > 0`;
- `N > 0` and `N % 8 == 0`;
- `K > 0` and `K % 128 == 0`.

Checked-in exact overrides cover the five Qwen3.5-27B TP2 linear layers at
`M=1,16,32,64,96,512,1024,2048,4096,8192`. Swapped x8, x16 and x32 tiles keep
the small activation batch in the tensor-core N dimension. Larger batches use
normal 64x64, 64x128 and 128x128 tiles. The portfolio includes ping-pong,
cooperative, static-persistent and selective Stream-K scheduling.

## Requirements

- NVIDIA compute capability 12.0 GPU;
- CUDA Toolkit 12.8 or newer;
- CUDA-enabled PyTorch;
- CMake 3.24 or newer and a C++17 compiler;
- Ninja is recommended.

CUTLASS is pinned at `e6233cbac5d7c7a865c19c91cd684ceece19513c`.
The optional Humming reference is pinned at
`694298e9eb25ffdfc088353b49ba537ebf304479`.

## Installation

Clone with the pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
```

For an existing clone:

```bash
git submodule update --init --depth 1 third_party/cutlass third_party/humming
```

Build a wheel against the active PyTorch ABI:

```bash
./scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

Install the optional Humming comparison dependencies when needed:

```bash
python3 -m pip install --no-build-isolation '.[humming]'
```

The wheel build applies the required runtime CUTLASS patch idempotently. Use
`MAX_JOBS=1` on memory-constrained systems.

For a development build:

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Python API

Quantize a persistent weight once, then pass FP16 or BF16 activations directly
to `gemm`:

```python
import torch
import mxfp6

m, n, k = 32, 8192, 5120
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

packed_weight = mxfp6.quantize_mxfp6(weight)
output = mxfp6.gemm(a, packed_weight)
```

The activation quantizer and W6A8 GEMM can be invoked separately for explicit
prewarming, activation reuse or CUDA Graph capture:

```python
quantized_a = mxfp6.quantize_activation(a)
mxfp6.autotune_w6a8(quantized_a, packed_weight)
output = mxfp6.gemm_w6a8(quantized_a, packed_weight)
```

Logical E3M2 code and UE8M0 scale utilities remain available for serialization
and tests:

```python
codes = torch.randint(0, 64, (n, k), device="cuda", dtype=torch.uint8)
scales = torch.full((n, k // 32), 0x7f, device="cuda", dtype=torch.uint8)
packed_weight = mxfp6.pack_operand(codes, scales)
restored_codes, restored_scales = mxfp6.unpack_operand(packed_weight)
```

Each code uses its low six bits. UE8M0 byte `0x7f` represents scale 1.0.

Humming is an explicit reference backend only:

```python
packed_a6 = mxfp6.quantize_mxfp6(a)
humming_weight = mxfp6.prepare_humming_weight(packed_weight)
reference = mxfp6.gemm(packed_a6, humming_weight)
```

## Runtime autotuning

Unknown W6A8 shapes use first-use selection over 29 precompiled native CUTLASS
families. This is AOT kernel selection, not runtime NVRTC compilation:

1. a coarse pass ranks kernel and swizzle choices;
2. the three best families refine raster and swizzle;
3. the winner is checked numerically against the deterministic fallback;
4. the full launch config is installed in the C++ dispatcher and cached by
   build, GPU, shape and measurement policy.

Later calls use the in-process override; later processes load the JSON cache
without profiling. Tuning and file I/O are disabled during CUDA Graph capture
and `torch.compile` tracing.

Useful controls:

```bash
MXFP6_AUTOTUNE=0                    # checked-in/static fallback only
MXFP6_AUTOTUNE_VERBOSE=1            # print tune and cache-hit decisions
MXFP6_AUTOTUNE_CACHE_DIR=/local/dir # override the persistent cache directory
MXFP6_AUTOTUNE_FLUSH_L2_MB=256      # tune with explicit cache flushing
MXFP6_AUTOTUNE_EXACT=1              # retune checked-in exact shapes
MXFP6_AUTOTUNE_EXHAUSTIVE=1         # offline: refine every eligible family
```

The default cache is `$XDG_CACHE_HOME/mxfp6/autotune`, or
`~/.cache/mxfp6/autotune` when `XDG_CACHE_HOME` is unset.

## Benchmarking

Reproduce the table above:

```bash
CUDA_VISIBLE_DEVICES=0 MXFP6_AUTOTUNE=0 \
python3 benchmarks/benchmark.py \
  --library build/mxfp6_torch.so \
  --activation-input bf16 \
  --compare-humming --compare-fp8 \
  --warmup 20 --iterations 100 --flush-l2-mb 0
```

The benchmark independently reports GEMM and activation quantization. Every
printed speedup uses GEMM alone. `--flush-l2-mb=0` selects warm-cache runs;
use `--flush-l2-mb=256` for an explicit cold-weight regime. Add `--check-all`
to validate every baseline output rather than the first representative shape.

The default matrix contains five Qwen3.5-27B TP2 layers per batch:

| Layer | N | K |
|---|---:|---:|
| GDN input projection | 8192 | 5120 |
| GDN output projection | 5120 | 3072 |
| Full-attention QKV/gate projection | 7168 | 5120 |
| MLP gate/up projection | 17408 | 5120 |
| MLP down projection | 5120 | 8704 |

Small-batch W6A8/W6A6 and quantizer microbenchmarks are available separately:

```bash
python3 benchmarks/compare_small_batch.py --flush-l2-mb=0
python3 benchmarks/quantization.py
```

## CUTLASS development

The runtime build requires the small-tile SM120 patch. The profiler patch adds
the wider candidate grid used for offline kernel search:

- `patches/cutlass/0001-sm120-mxfp6-small-tile-runtime.patch`;
- `patches/cutlass/0002-sm120-mxfp6-profiler-search.patch`.

Apply or inspect the queue with:

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
bash scripts/apply_cutlass_patches.sh --check --runtime-only
bash scripts/apply_cutlass_patches.sh --reverse
```

Build generated profiler libraries and run fixed-shape search with:

```bash
./scripts/build_profiler.sh w6a8
python3 benchmarks/autotune.py \
  --mma=w6a8 --devices=0 --orientations=both \
  --top-k=10 --output-dir=benchmarks/results/native_search
```

The CUTLASS profiler is a candidate generator rather than the production
dispatcher. Ordered profiling, clock drift, warm weights and minimum-selection
bias can change a winner. Promote a fixed configuration only after independent
randomized validation. General atomic work stealing is not used for uniform
interior tiles; Stream-K is retained only for underfilled tail waves.

## Repository layout

```text
csrc/                  CUDA/C++ kernels and quantizers
python/mxfp6/          Python API and persistent autotuner
benchmarks/            GEMM, baseline and search tools
benchmarks/results/    Reviewed dispatch and measurement metadata
tests/                 CUDA correctness and stream tests
patches/cutlass/       Versioned SM120 CUTLASS fixes
scripts/               Build and patch helpers
third_party/           Pinned CUTLASS and Humming submodules
```

## License

The project is released under the BSD 3-Clause License. CUTLASS and Humming
retain their upstream licenses.
