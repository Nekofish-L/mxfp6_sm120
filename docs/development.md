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

For an editable source build and CTest run:

```bash
bash scripts/apply_cutlass_patches.sh --runtime-only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

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
| CUTLASS | Pinned submodule commit `e6233cbac5d7c7a865c19c91cd684ceece19513c` plus the checked-in runtime patches. |
| Distribution | Source wheel only. No cross-PyTorch or cross-CUDA binary compatibility is promised. |

The build intentionally rejects an uninitialized or mismatched CUTLASS
checkout. The Python API checks CUDA placement, dtype, contiguity, shape and
SM120 capability before dispatch.


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

See [checkpoint conversion and format](quantize-checkpoints.md) for the model-scoped loader boundary and [benchmark evidence](benchmarks.md) for the frozen reproducer measurements.
