# Development

## Source build

Clone the repository and pinned CUTLASS submodule:

```bash
git clone --recurse-submodules https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
```

Toolchain and platform requirements are listed in
[Compatibility](compatibility.md).

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

For an editable source build and CTest run:

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## CUTLASS patch queue

CUTLASS is pinned at `e6233cbac5d7c7a865c19c91cd684ceece19513c`.
The runtime build requires the small-tile SM120 patch; the profiler patch adds
the wider candidate grid used for offline search:

- `patches/cutlass/0001-sm120-mxfp6-small-tile-runtime.patch`;
- `patches/cutlass/0002-sm120-mxfp6-profiler-search.patch`;
- `patches/cutlass/0003-sm120-streamk-persistent-workspace.patch`.

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
