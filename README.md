# SM120 MXFP6 GEMM

Native OCP MXFP6 matrix multiplication for NVIDIA Blackwell GeForce SM120:

```text
D[M, N] = A[M, K] @ B[N, K].T
```

Both operands use bit-packed E3M2 storage with one UE8M0 scale per 32 values.
The dispatcher selects one of two SM120 block-scaled tensor-core paths:

- native E3M2-by-E3M2 compute for general and latency-oriented shapes;
- E4M3-by-E3M2 compute for profiler-tuned large-M shapes, after losslessly
  expanding only the activation.

Both paths accumulate in FP32 and write FP16 output. Persistent weights always
remain packed at six bits per value. CUDA operators provide FP6 packing,
lossless activation expansion, and CUTLASS scale-layout conversion.

The implementation uses the CUTLASS C++ CollectiveBuilder API. CUTLASS is a
pinned upstream submodule; the small-tile SM120 fixes required by this project
are maintained as an explicit patch queue.

The large-M design is based on the same storage/compute separation used by
Humming, while the shipped path uses CUTLASS kernels selected by exhaustive
profiler search. A pinned Humming backend is retained as an optional correctness
and performance reference.

The public operator supports every shape satisfying:

- `M > 0`
- `N > 0` and `N % 8 == 0`
- `K > 0` and `K % 128 == 0`

The checked-in exact profiler overrides are tuned specifically for
Qwen3.5-27B with tensor parallel size 2. They cover its five linear-layer
shapes at `M=1,16,32,64,96,2048`. Native W6A6 is used whenever it wins the
complete pipeline; selected shapes at `M=32,64,96,2048` losslessly expand the
activation and use mixed W6A8. All other valid problems use an 18-kernel
native-W6A6 portfolio derived from a representative exhaustive search. The
general dispatcher swaps operands for `M <= 96`, then selects tile width,
pipeline depth, schedule, and Stream-K from the problem geometry.

## Requirements

- NVIDIA compute capability 12.0 GPU
- CUDA Toolkit 12.8 or newer
- CUDA-enabled PyTorch
- CMake 3.24 or newer and a C++17 compiler
- Ninja (recommended)

The optional Humming reference backend requires the dependencies installed by
the `humming` Python extra. It JIT-compiles on first use and reuses its cache in
later processes.

The current CUTLASS revision is
`e6233cbac5d7c7a865c19c91cd684ceece19513c`.
The bundled Humming revision is
`694298e9eb25ffdfc088353b49ba537ebf304479`.

## Installation

Clone the repository and initialize both pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
```

For an existing clone without submodules:

```bash
git submodule update --init --depth 1 third_party/cutlass third_party/humming
```

Build and install a wheel:

```bash
./scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

To include the large-M Humming backend and its runtime dependencies, install
directly from the checkout:

```bash
python3 -m pip install --no-build-isolation '.[humming]'
```

Built wheels contain the pinned Humming source snapshot, so deployment does
not require a separate Humming checkout. The optional dependencies still need
to be installed through the extra.

The build script applies the required runtime patch idempotently. It does not
download `nvidia-cutlass-dsl`; CUTLASS source is obtained only through the Git
submodule. Build isolation is disabled intentionally so the extension links
against the active PyTorch ABI. Use `MAX_JOBS=1` on memory-constrained hosts.

To build the development targets directly:

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The shared library contains `sm_120a` machine code and must be built against a
PyTorch/CUDA ABI compatible with the deployment environment.

## Python API

`pack_operand` and `gemm` form the recommended interface. Weights should be
packed once and reused.

```python
import torch
import mxfp6

m, n, k = 16, 8192, 5120
a_codes = torch.randint(0, 64, (m, k), device="cuda", dtype=torch.uint8)
b_codes = torch.randint(0, 64, (n, k), device="cuda", dtype=torch.uint8)
sfa = torch.full((m, k // 32), 0x7f, device="cuda", dtype=torch.uint8)
sfb = torch.full((n, k // 32), 0x7f, device="cuda", dtype=torch.uint8)

packed_b = mxfp6.pack_operand(b_codes, sfb)
packed_a = mxfp6.pack_operand(a_codes, sfa)
output = mxfp6.gemm(packed_a, packed_b)
```

Each `uint8` code stores one E3M2 pattern in its low six bits. UE8M0 byte
`0x7f` represents a scale of 1.0. The logical scale tensor shape is always
`[rows, K / 32]`.

Lower-level and conversion APIs are also available:

```python
output = mxfp6.gemm_from_codes(a_codes, b_codes, sfa, sfb)
packed_values = mxfp6.pack_fp6(a_codes)
restored_codes = mxfp6.unpack_fp6(packed_values, m, k)
packed_scales = mxfp6.pack_scales(sfa)
restored_scales = mxfp6.unpack_scales(packed_scales, m, k)
```

`gemm_from_codes` includes conversion overhead and is intended for convenience,
not latency-critical inference.

### Hybrid dispatch and Humming reference

For tuned mixed entries at `M=32,64,96,2048`, the normal packed call
automatically expands A and selects the profiler-ranked E4M3-by-E3M2 CUTLASS
kernel. Programmatic dependent launch reduces the handoff cost between the
conversion and GEMM kernels. The temporary is exactly `M*K` bytes; both input
objects retain their compact `6/8 * rows * K`-byte value storage.

The optional Humming backend can be selected explicitly for comparison. Repack
a persistent weight once, then pass it to the same `gemm` entry point:

```python
humming_b = mxfp6.prepare_humming_weight(packed_b)
output = mxfp6.gemm(packed_a, humming_b)
```

The prepared Humming object also stores values at six bits each and does not
retain a duplicate native weight. Its current wrapper requires `N % 256 == 0`
and `alpha == 1.0`.

## Benchmarking

Build the development library, then run:

```bash
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/benchmark.py
```

The benchmark reports both complete GPU-pipeline latency and isolated GEMM
kernel latency. Pipeline timing includes activation expansion when the mixed
path is selected. It uses an 8 GB cache flush by default; pass
`--flush-l2-mb=0` for warm-cache measurements.

The default problems are the five Qwen3.5-27B TP2 linear layers under the
operator's `A[M,K] @ B[N,K].T` convention, at
`M=1,16,32,64,96,2048`:

| Layer | N | K |
|---|---:|---:|
| GDN input projection | 8192 | 5120 |
| GDN output projection | 5120 | 3072 |
| Full-attention QKV/gate projection | 7168 | 5120 |
| MLP gate/up projection | 17408 | 5120 |
| MLP down projection | 5120 | 8704 |

Compare against PyTorch's FP8 `torch._scaled_mm`:

```bash
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/benchmark.py \
  --compare-torch-scaled-mm
```

This baseline losslessly expands the same E3M2 values to E4M3, uses scalar
unit scales and FP16 output, and leaves FP8 preparation outside the timed
region. `use_fast_accum=False` matches the PyTorch default and is faster for
these SM120 shapes in the tested PyTorch build.

An optional FP8 comparison uses vLLM's installed block-scaled CUTLASS operator
without importing external source files:

```bash
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/benchmark.py --compare-fp8
```

Compare the optimized CUTLASS path with the bundled Humming reference:

```bash
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/benchmark.py \
  --compare-humming --flush-l2-mb=0
```

Persistent weight conversion and first-use JIT are outside the timed region;
activation E3M2-to-E4M3 conversion is inside both pipeline measurements.

One representative RTX 5090 cold-cache run reports the following
geometric-mean pipeline speedups. These are full operator times and include
activation expansion on mixed paths:

| M | Shapes | vLLM FP8 / MXFP6 | `torch._scaled_mm` / MXFP6 |
|---:|---:|---:|---:|
| 1 | 5 | 1.266x | 1.651x |
| 16 | 5 | 1.153x | 1.443x |
| 32 | 5 | 1.110x | 1.336x |
| 64 | 5 | 1.079x | 1.803x |
| 96 | 5 | 1.274x | 1.530x |
| 2048 | 5 | 1.518x | 0.999x |
| Overall | 30 | 1.225x | 1.436x |

The shallow-K GDN output projection at `M=32` and `M=64` is approximately at
parity and can vary by about five percent across runs. The grouped results are
stable across physical GPUs 4-7. Absolute values vary with clocks, power
limits, thermals, and software versions.

## CUTLASS profiler autotuning

Build the generated profiler libraries for native W6A6 and mixed W6A8 search:

```bash
./scripts/build_profiler.sh w6a6
./scripts/build_profiler.sh w6a8
```

Run native exhaustive fixed-shape search for the 30 default targets on
physical GPUs 4-7:

```bash
python3 benchmarks/autotune.py \
  --devices=4,5,6,7 \
  --orientations=both \
  --workspace-count=0 \
  --llc-capacity-kib=524288 \
  --split-k-slices=1 \
  --top-k=10 \
  --output-dir=benchmarks/results/native_search
```

Reproduce the mixed-MMA candidate search used by the hybrid dispatcher:

```bash
python3 benchmarks/autotune.py \
  --mma=w6a8 \
  --devices=4,5,6,7 \
  --orientations=normal \
  --workspace-count=0 \
  --llc-capacity-kib=524288 \
  --split-k-slices=1 \
  --top-k=10 \
  --output-dir=benchmarks/results/mixed_search
```

The script defaults to GPUs `4,5,6,7`, serializes work assigned to each GPU,
and uses CUTLASS exhaustive fixed-shape search with top-k ranking. The profiler
is a candidate generator: final selection must use `benchmark.py` with its
default 8 GB flush because profiler ordering alone did not reproduce every
small-batch PyTorch pipeline result. Raw CSV files and exploratory outputs
under `benchmarks/results/` are ignored. The curated target-shape and
general-policy manifests are retained under `benchmarks/results/`.

Analyze one or more profiler result directories and reproduce the orientation
summary, rule quality, and greedy portfolio selection with:

```bash
python3 benchmarks/analyze.py benchmarks/results/latest \
  --swap-max-m=96 --portfolio-size=18
```

## CUTLASS patches

- `0001-sm120-mxfp6-small-tile-runtime.patch` is required for runtime builds.
- `0002-sm120-mxfp6-profiler-search.patch` adds profiler candidates and minimal
  library generation; it is required only for autotuning.

Useful commands:

```bash
bash scripts/apply_cutlass_patches.sh --check --runtime-only
bash scripts/apply_cutlass_patches.sh --check
bash scripts/apply_cutlass_patches.sh --reverse
```

## Repository layout

```text
csrc/                  CUDA/C++ kernels and headers
python/mxfp6/           Python package and operator wrappers
benchmarks/             Performance and autotuning tools
tests/                  End-to-end CUDA tests
scripts/                Build and patch entry points
patches/cutlass/        Versioned CUTLASS patch queue
third_party/cutlass/    Pinned upstream submodule
third_party/humming/    Pinned optional large-M backend
```

## License

The project is released under the BSD 3-Clause License. CUTLASS and Humming
retain their upstream licenses; the Humming license is included in built
wheels.
