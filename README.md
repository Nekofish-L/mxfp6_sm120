# SM120 MXFP6 GEMM

Native OCP MXFP6 matrix multiplication for NVIDIA Blackwell GeForce SM120:

```text
D[M, N] = A[M, K] @ B[N, K].T
```

Inputs use E3M2 values with one UE8M0 scale per 32 elements. The kernel uses
SM120 block-scaled tensor cores (`QMMA.SF.16832.F32.E3M2.E3M2.E8`), accumulates
in FP32, and writes FP16 output. CUDA kernels are also provided for FP6 packing
and CUTLASS scale-layout conversion.

The implementation uses the CUTLASS C++ CollectiveBuilder API. CUTLASS is a
pinned upstream submodule; the small-tile SM120 fixes required by this project
are maintained as an explicit patch queue.

## Requirements

- NVIDIA compute capability 12.0 GPU
- CUDA Toolkit 12.8 or newer
- CUDA-enabled PyTorch
- CMake 3.24 or newer and a C++17 compiler
- Ninja (recommended)

The current CUTLASS revision is
`e6233cbac5d7c7a865c19c91cd684ceece19513c`.

## Installation

Clone the repository and initialize CUTLASS:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
```

For an existing clone without submodules:

```bash
git submodule update --init --depth 1 third_party/cutlass
```

Build and install a wheel:

```bash
./scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

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

m, n, k = 16, 5120, 8192
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

## Benchmarking

Build the development library, then run:

```bash
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/benchmark.py
```

The benchmark reports kernel latency, TFLOP/s, estimated bandwidth, and
correctness. It uses an 8 GB cache flush by default to match the target
inference benchmark. Pass `--flush-l2-mb=0` for warm-cache measurements.

An optional FP8 comparison uses vLLM's installed block-scaled CUTLASS operator
without importing external source files:

```bash
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/benchmark.py --compare-fp8
```

## CUTLASS profiler autotuning

Apply the complete patch queue and build the generated profiler library:

```bash
bash scripts/apply_cutlass_patches.sh
cmake -S third_party/cutlass -B build_profiler -G Ninja \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  -DCUTLASS_LIBRARY_KERNELS='cutlass3x_sm120_bstensorop_gemm_ue8m0xe3m2_ue8m0xe3m2_f32_void_f16_*' \
  -DCUTLASS_LIBRARY_MINIMAL_MXFP6_PROFILER=ON \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL=max \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_profiler --parallel
```

Run exhaustive fixed-shape search on physical GPUs 4-7:

```bash
python3 benchmarks/autotune.py \
  --devices=4,5,6,7 \
  --m-values=1,16,32,64,128,256,512,1024,2048,4096 \
  --n-values=128,512,2048,8192,16384 \
  --k-values=128,512,2048,8192,16384 \
  --orientations=both
```

The script defaults to GPUs `4,5,6,7`, serializes work assigned to each GPU,
and uses CUTLASS exhaustive fixed-shape search with top-k ranking. Raw CSV files
and exploratory outputs under `benchmarks/results/` are ignored. The curated
target-shape manifest is retained as
`benchmarks/results/target_shapes.json`.

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
```

## License

The project is released under the BSD 3-Clause License. CUTLASS retains its
upstream license; the wheel includes the corresponding notice.
