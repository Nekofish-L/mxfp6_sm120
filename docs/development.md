# Development

## Source build

Clone the repository and pinned CUTLASS submodule:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
```

Toolchain, ABI and platform requirements are listed below.

For an existing clone:

```bash
git submodule update --init third_party/cutlass
```

Build a wheel against the active PyTorch ABI:

```bash
./scripts/build_wheel.sh
python3 -m pip install --no-deps dist/mxfp6_sm120-*.whl
```

The wheel build applies the required runtime CUTLASS patch idempotently. Use
`MAX_JOBS=1` on memory-constrained systems.

### Wheel build identity

The wheel command uses the installed Torch runtime with `--no-build-isolation` and `--no-deps`; install the target Torch and Triton first. It writes `build_info.json` next to both DSOs in the build output and packages it in the wheel, keeping generated identities and DSOs in the build directory.

The identity records the complete Torch version, Torch's CUDA version, CXX11 ABI, Python, detected C++/CUDA compilers, CMake arguments/cache and both DSO hashes. For packaged DSOs accompanied by this metadata, the loader checks the exact Torch/CUDA/CXX ABI identity and SHA-256 before loading. Python/compiler/CMake details provide build provenance. Inspect an installed wheel without loading a DSO:

```bash
python3 -c 'from importlib.metadata import distribution; print(distribution("mxfp6-sm120").locate_file("mxfp6/build_info.json").read_text())'
```

Explicit `MXFP6_LIBRARY_PATH` / `MXFP8_LIBRARY_PATH` overrides retain their existing precedence and bypass the packaged identity check, including paths inside the package. Manual/source builds without `build_info.json` retain the existing loader behavior and path selection.

For a manual CMake build and CTest run:

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Dense host validation, overrides and operator registration live in
`csrc/torch_extension.cpp`; template instantiation lives in
`csrc/dense_kernels.cu`. W6A8 exact-policy edits affect the host selector, while
candidate/kernel changes rebuild the CUDA unit. Keep CUDA templates out of the
host binding file so API and policy edits retain the shorter incremental build.

Validate the optional W6A8 JIT path and compare it against AOT on identical
parameter/workspace addresses:

```bash
PYTHONPATH=python MXFP6_LIBRARY_PATH="$PWD/build/mxfp6_torch.so" \
  MXFP6_TEST_JIT=1 pytest -q tests/test_w6a8_jit.py
python3 benchmarks/compare_w6a8_jit.py --library build/mxfp6_torch.so \
  --output build/w6a8-jit-comparison.json
```

`benchmarks/benchmark_w6a8_prepared.py` separately measures host enqueue cost
and Kineto GPU duration for ordinary versus prepared calls. Its default
baseline uses persistent workspace planning; `--workspace ephemeral` includes
ordinary per-call workspace initialization in the execution path.

## CUTLASS patch queue

CUTLASS is pinned at `e6233cbac5d7c7a865c19c91cd684ceece19513c`.
The runtime patch set includes small tiles, persistent Stream-K workspaces, the single-stage mainloop, static problem shapes and PDL release. `scripts/apply_cutlass_patches.sh --runtime-only` applies patches 0001/0003/0004/0005/0006; patch 0002 adds the wider offline profiler search grid. See the [patch queue](../patches/cutlass/README.md) for each change.

Apply, validate or reverse the queue with:

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

The CUTLASS profiler is a candidate generator, not the production dispatcher.
Ordered profiling, clock drift, cache state and minimum-selection bias can
change a winner. Promote a fixed configuration only after independent
randomized validation under the target execution mode.

## Validation

Run the source checks before submitting a change:

```bash
python3 -m compileall -q python benchmarks tests tools
git diff --check
```

CUDA changes require the focused correctness tests for their public operator,
then the relevant layer or end-to-end benchmark. Follow the evidence and
review requirements in [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Repository layout

```text
csrc/                  CUDA/C++ kernels and quantizers
python/mxfp6/          Python API and persistent autotuner
benchmarks/            GEMM, baseline and search tools
benchmarks/results/    Reviewed dispatch and measurement metadata
docs/                  User guides and integration contracts
examples/vllm/         Version-locked public vLLM reproducer
tests/                 CUDA correctness and stream tests
tools/                 Build and Qwen3.5 workspace validation helpers
patches/cutlass/       Versioned SM120 CUTLASS fixes
scripts/               Build and patch helpers
third_party/           Pinned CUTLASS submodule
```

## Compatibility

### Dispatch and build iteration

W6A8 candidate IDs, launch metadata and ordered exact-shape rules live in
`python/mxfp6/w6a8_dispatch.json`. CMake generates the native registration and
direct launch branches. After editing the manifest, regenerate the checked-in
Python registry before building:

```bash
python3 scripts/generate_w6a8_dispatch.py \
  --python-output python/mxfp6/_w6a8_registry.py
```

The generator preserves direct calls and branch order. Candidate IDs and the
ABI remain stable; the existing library fingerprint still invalidates tuning
cache entries when a rebuilt library changes. Search ordering lives in
`_autotune_policy.py`, CUDA Graph measurement in `_autotune_benchmark.py`, and
cache/installation orchestration in `autotune.py`. The refactor preserves the
candidate shortlists, event boundaries, validation, fallback and selection
threshold. Tests freeze the original native dispatch tokens; an intentional
policy change needs fresh correctness/performance evidence before updating
that snapshot.

Wheel builds select up to eight parallel compiler jobs, bounded by CPU affinity,
container CPU/memory limits and available memory. The budget allows 24 GiB per
job and reserves a quarter of available memory. `MAX_JOBS` takes precedence.
Use `python3 scripts/build_resources.py` to inspect the detected budget. The
helper does not use `nproc`, whose output can be reduced by `OMP_NUM_THREADS`.
For incremental development, retain a CMake build directory and build only the
affected target:

```bash
cmake --build build --target mxfp6_torch --parallel 8
cmake --build build --target mxfp8_torch --parallel 8
```

The full build remains the default. For serving with the builtin W8A8 policy,
`-DMXFP8_PRUNE_UNUSED_FAMILIES=ON` omits entire family translation units that
have no policy references, including generic fallbacks. This preserves builtin
selection for all shapes while reducing explicit tactic availability. Such
builds intentionally exclude some experimental winners and are unsuitable for
full-portfolio tuning. Core translation units, dual activation and W6A8 remain
included. Policy/registry changes trigger regeneration of the source list.
For wheels, pass the option through `CMAKE_ARGS`:

```bash
MAX_JOBS=8 CMAKE_ARGS=-DMXFP8_PRUNE_UNUSED_FAMILIES=ON bash scripts/build_wheel.sh
```

For a hybrid portfolio that retains every existing tactic ID across AOT/JIT,
use `-DMXFP8_JIT_UNUSED_FAMILIES=ON` instead. This omits only default-unreferenced
families whose complete native ID range is covered by the JIT manifest (currently
`cute` and `specialized`, 52 tactics). Other families remain AOT. Query the union
with `available_configs(backend="auto")` and execute through
`prepare(..., backend="auto")`; migrated JIT tactics require scheduler parameters
1/1/0, the matching source checkout and compiler. Default `gemm` and `prepare`
still use AOT, so an omitted explicit tactic requires opting into auto/JIT.
The two pruning options are mutually exclusive. The measured 22.3% build-time
reduction below applies to broader builtin-only pruning, not this hybrid mode.

Inspect builtin reachability with `benchmarks/audit_dispatch.py`; screen all
available configurations on public model projection shapes with
`benchmarks/screen_configs.py --backend w8a8 --output build/screen.json`.
The offline tools load `bench_kineto` from a sibling DeepGEMM checkout by default;
set `MXFP_BENCH_KINETO` or `--kineto-source` to choose another implementation.
Add `--flush-l2-mb 256` for the reference helper's 256 MB L2 eviction. These are bounded
operator measurements, not serving throughput or evidence that a configuration
is globally useless. Warm-cache and flushed results must remain separate.

For behavior-preserving changes, `scripts/compare_cuda_code.py` compares every
GPU function's SASS and resource metadata. `benchmarks/compare_refactor.py`
records full-output comparisons and alternating same-GPU `bench_kineto`
measurements of Graph replay and eager GPU kernel duration, in warm and cold
cache regimes. These exclude host launch time and workspace memsets. Set `CUDA_VISIBLE_DEVICES` to an idle GPU,
and use `MXFP6_LIBRARY_PATH`/`MXFP8_LIBRARY_PATH` to select source-built libraries.
Retain raw results and inspect per-shape outliers before accepting a change.

`benchmarks/evaluate_shortlist.py` evaluates wave/tile ranking against the
complete measured W6A8 coarse portfolio without installing a policy. A geometric
shortlist alone does not model SM120 register pressure, pipeline latency or
memory traffic; keep exhaustive/offline coverage when evaluating a tighter list.

### Optional DeepJIT backend

`mxfp8.precompile_jit` and `mxfp8.prepare_jit` provide explicit JIT entrypoints
for the 88 migrated cp.async tactics. Default AOT dispatch remains unchanged.
They require a matching source checkout, initialized CUTLASS/DeepJIT submodules,
a C++20 host compiler and NVCC 12.9+. Installed wheels can supply `source_root`;
the source checkout is not embedded into the wheel. The first opt-in call builds
a small PyTorch host bridge using the normal Torch extension cache. Default
AOT imports/builds do not load or compile this bridge.

The backend uses one owned runtime/kernel handle per device and specialization,
with thread locks and cross-process file locks preventing duplicate compilation.
Each prepared callable owns its tensors, output and compiled handle. The
process cache also retains compiled handles until process exit, so releasing a
prepared callable cannot invalidate a captured Graph. There is no handle eviction;
applications should precompile a bounded serving portfolio.

Call precompile/prepare before CUDA Graph capture or `torch.compile` tracing.
Prepared launches perform no source hashing, compilation, cache I/O or output
allocation. They validate tensor contracts, guard the input device, use its
current CUDA stream, and support PDL. Each concurrent invocation needs its own
prepared object/output. Dynamic-K kernels reuse compiled handles; the 2048/2560
specializations have separate handles. The default AOT candidate set is retained.

Set `MXFP8_JIT_CACHE_DIR` before first use to choose the CUBIN cache (otherwise
DeepJIT uses `DJ_JIT_CACHE_DIR` or `~/.dj`). Source roots, environment/compiler
settings and cache paths are fixed for the lifetime of a process: restart after
editing source or changing settings. Dependency signatures include the complete
CUTLASS, DeepJIT and local include trees, plus the host bridge source; DeepJIT
also hashes the compiler, compiler options and generated kernel source. This
conservatively invalidates unrelated kernels when a header changes.

`experiments/deepjit_probe.py` now benchmarks these same public APIs rather than
maintaining a second C++ runtime implementation. Initialize the dependency with
`git submodule update --init third_party/deepjit`, then run
`python3 experiments/deepjit_probe.py --output build/jit-runtime/probe.json`
with the native AOT library paths configured. `--source-root` selects the matching
checkout; `--cache` allows a separate cold-cache or disk-hit experiment. The
probe records both eager and Graph GPU durations using `bench_kineto`.

The DeepGEMM-pinned DeepJIT revision
`2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43` uses explicit `120a` architecture and
PTXAS register-usage level 5 to match AOT, overriding its `120f`/level-10 defaults.
Its parser rejects quoted includes, so local device helpers are inlined while
external dependency trees are hashed separately. Each CUBIN contains one kernel.
Tuning decision persistence remains separate in `_autotune_cache.py`; compiling
a candidate does not select it or install an autotune result.

### Device kernels and launch adapters

The `cute_4warp`, `specialized`, `vector_store`, `cache_hint`, `eightwarp` and
`occupancy` families keep their device implementation under
`csrc/include/mxfp8_gemm/kernels/`. Their host adapters are generated under the build directory from
`python/mxfp6/mxfp8_cp_async.json`; the handwritten host headers are removed. Both AOT and the optional
JIT experiment consume the same device source.

`kernels/async_pipeline.cuh` owns their shared asynchronous copies and prefetch
loops. Compile-time thread count and cache policy preserve the 256-thread
`eightwarp` path and the `.ca` 16-byte copies in `cache_hint`; the other families
retain 128 threads and `.cg` copies. Scale copies remain `.ca` in all families.
The six previous copies of this pipeline are removed. `cp_async_body.inc`
also shares their MMA mainloop and scalar/vector epilogues; only the existing
kernel entries and compile-time policies remain family-specific. It expands
inside each entry to preserve the original parameter/restrict scope: extracting
a separate force-inlined device function changed register allocation. Keep this
include free of `#pragma once`, since multiple entries can use it in one TU.
Tactic IDs, launch geometry and default dispatch remain unchanged.

Five TMA families (`fewwarp`, `oddwarp`, `oddrotate`, `tma256` and
`eightwarp`) share `kernels/tma_warp_body.inc`. The first three also share
`tma_warp_launch.inc`. These expand in the original kernel entry and family
namespace, respectively, preserving kernel signatures and fixed-K specializations.
Each family retains its prefetch implementation, configuration switch and launch
resource policy. `tma256` fixes the warp count to four; `eightwarp` reserves the
larger of pipeline and output storage, while the others require the output to
fit within the pipeline storage. The includes deliberately have no include
guards; shared source does not merge compiled tactics.

### Native tactic routing

`mxfp8_gemm/tactic.hpp` decodes a stable tactic ID into its backend, launch
argument and Stream-K capability. Availability enumeration, explicit-ID validation
and native execution consume this same description. Family-local IDs still come
from `registry.def`; replicated native tactics retain their full encoded ID,
and GEMV receives its row count. Pruned builds use the generated registry, so
omitted families remain unavailable. Shape policy selection is separate and
unchanged. The CPU routing snapshot covers every ID and gap in both full and
pruned builds, including signed integer extremes.

### Shared launch and configuration descriptions

The first six migrated families cover 88 stable tactics. Their manifest records
local IDs, tiles, stages, swap/warp/scale layout, thread count, fixed-K policy
and shared-memory policy. `_mxfp8_config.py` resolves the descriptors for build
and JIT use; it does not change default dispatch or runtime availability. Tactics
outside this manifest still use their existing implementation, and a pruned
build may omit some of the described families.

`generate_mxfp8_configs.py` checks the manifest ranges against `registry.def`
and emits the family adapters. `mxfp8_gemm/launch.cuh` owns shared-memory opt-in,
stream selection, PDL launch, swapped operand mapping and fixed-K dispatch.
Validation and the device guard remain at each operator entry, before any
launch. The generated templates preserve dynamic K and the 2048/2560
specializations, including the original 256-thread and output-storage rules.
Generated headers are build artifacts; edit the manifest rather than keeping
another copy of configuration switches. Unchanged outputs retain their mtime,
and a generation stamp avoids repeated builds after metadata-only edits.

The optional probe now obtains both its kernel template expression and launch
resources from that same manifest. For example:

```bash
python3 experiments/deepjit_probe.py --tactic 520 --shape 32 512 2048 \
  --output build/host-refactor/jit-520.json
```

The probe accepts the 88 described tactics, positive M, and N/K divisible
by 128. The explicit prepared JIT API uses the same contract, with BF16 output
and splits/swizzle/sms fixed to 1/1/0. Opt-in `prepare(..., backend="auto")` uses JIT only when the same selected
tactic is absent from AOT; the default backend remains `aot`.

### Refactor validation (2026-11-16)

The [validation receipt](../benchmarks/results/refactor_validation_20261116.json)
records the unchanged default dispatch, device-code comparisons, real-shape
coverage and `bench_kineto` measurements on RTX 5090 / CUDA 13.0. The full build
preserves all 312 W6A8 and 861 W8A8 device functions and their resource usage.
All 32 W6A8 and 23 W8A8 comparison cases produce bitwise identical outputs.
Graph warm-cache median changes are -0.01% and -0.02%; maximum increases are
0.58% and 0.28%. Cold-cache Graph maximum increases are 1.16% and 1.25%.
These are measured samples, not a guarantee for every possible workload.

The default W8A8 policy reaches 50 of 726 tactics across 36,400 boundary cells,
and selects 33 on 260 model shapes. However, bounded Kineto sweeps find 25
warm-cache and 30 cold-cache winning tactic IDs outside that reachable set.
Do not equate default-unreachable tactics with useless tuning candidates.
The geometric W6A8 shortlist also misses measured winners (up to 65.16%
regret for a 12-ID shortlist), so it is not enabled in production.

Optional family pruning reduces the W8A8 DSO from 44,669,688 to 24,471,496
bytes and retains 329 identical device functions. It remains disabled by
default because explicit tuning configurations disappear. A later same-source clean build comparison (Release, eight jobs, W8A8
target only) took 454.5 seconds for full AOT and 353.4 seconds for family pruning,
a 22.3% reduction. This is one sequential sample per mode with uncontrolled
filesystem caches, excluding configure time; it is not a wheel build benchmark.
Opt-in hybrid preparation restores omitted migrated tactics only; other omitted
families still require the full build. The isolated DeepJIT probe produces
identical SASS/resources and outputs for one tactic: cold compile/load takes
2.22 seconds, a new-process disk hit about 0.65 ms, and cached Graph kernel
medians are 8.698 us (AOT) versus 8.690 us (JIT). This establishes feasibility
for a hybrid path. The subsequent explicit prepared JIT API is validated
separately in the same receipt; automatic fallback/default AOT pruning remains
disabled.

### Remaining refactor scope

The hybrid portfolio preserves tactic IDs by retaining unmigrated families in
AOT. It does not finish JIT migration of the TMA/CUTLASS families or provide
per-tactic AOT pruning. Those changes need separate compiler/cache, Graph and
resource validation. The SM120 shape-policy cost model is also unfinished:
the geometric Top-12 shortlist still misses measured winners, so default
selection remains unchanged. Build speedups for the conservative JIT-covered
mode require a separate clean-build comparison; do not reuse the builtin-only
pruning measurement for it.

### Project scope

`mxfp6-sm120` is a source-built PyTorch CUDA extension for NVIDIA compute
capability 12.0. The public compatibility boundary is the Python and
`torch.ops.mxfp6` API in this repository. vLLM and SGLang are optional
benchmark/integration consumers, not package dependencies.

### Supported boundary

| Component | Current policy |
|---|---|
| GPU | Exactly SM120. Other compute capabilities fail closed at runtime. |
| OS | Linux. Other operating systems are not supported. |
| CUDA Toolkit | 12.8 or newer; the release is compiled for `sm_120a`. |
| PyTorch | CUDA-enabled PyTorch with FP8 dtype support. Build the extension against the exact PyTorch ABI used at runtime. |
| Python | 3.10 or newer. |
| Build tools | CMake 3.24+ and a host compiler with C++20 support; both C++ and CUDA sources compile in C++20 mode. |
| CUTLASS | Pinned submodule commit `e6233cbac5d7c7a865c19c91cd684ceece19513c` plus the checked-in runtime patches. |
| Distribution | Source wheel only. No cross-PyTorch or cross-CUDA binary compatibility is promised. |

The build intentionally rejects an uninitialized or mismatched CUTLASS
checkout. The Python API checks CUDA placement, dtype, contiguity, shape and
SM120 capability before dispatch.


### Qualified 0.3 build

On November 10, 2026, a Linux x86_64 / CPython 3.12.3 wheel was built with Torch 2.14.0+cu130, Torch CUDA 13.0, CXX11 ABI enabled, GCC 13.3.0, NVCC 13.0.88 and CMake 4.1.0. Its wheel RECORD and build identity match both packaged DSOs. Device-code inspection confirmed SM120 targets and unchanged executable device math relative to the previously tested target build. Source builds use the active Torch runtime. See [serving performance](performance.md) for the serving measurements and their runtime versions.

### Upgrade policy

Until a 1.0 release, CUDA operator schemas, workspace layouts and Qwen-specific
APIs may change between minor releases. Patch releases must remain compatible
within their minor line. Applications should pin both the package version and
the source commit.

## Runtime consumers

The standalone package has no vLLM or SGLang dependency and does not provide a general model checkpoint loader. [vLLM Mach](https://github.com/troycheng/vllm-mach) is the serving integration for current model profiles. [`examples/vllm`](../examples/vllm/README.md) retains a separate vLLM v0.28.0 TP2 reproducer for the validated Qwen3.5-27B and 35B-A3B Quark layouts.

| Public reproducer path | vLLM v0.28 interface | Package responsibility |
|---|---|---|
| Dense | `MxFp6LinearKernel` | Weight processing, capability checks and native W6A8 GEMM |
| Routed MoE | `FusedMoEExpertsModular` | Routing, W1, activation, W2, shared expert and reduction |

The example registers native Dense execution ahead of the emulation fallback and selects routed MoE only for its validated checkpoint geometry, SM120 and TP2 configuration. CUDA implementations and GPU tests belong to this package; registration, checkpoint admission and unsupported-case fallback belong to the runtime adapter. SGLang has no adapter in this repository.

See [checkpoint conversion and format](quantize-checkpoints.md) for the model-scoped loader boundary and [v0.28 integration receipt](../benchmarks/results/vllm_v028_public_reproducer_validation.json) for its measured execution checks.
