# Contributing

## Scope

Contributions should improve the SM120 kernel package, its public operator
contracts, or reproducible integration evidence. Runtime-specific adapters
must remain narrowly versioned and must not make this package import private
vLLM or SGLang modules.

## Development checks

Initialize the pinned CUTLASS submodule, apply the runtime patches and build
against the PyTorch installation that will run the tests:

```bash
git submodule update --init --depth 1 third_party/cutlass
bash scripts/apply_cutlass_patches.sh --runtime-only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest -q tests/test_ops.py
```

Before submitting a documentation or packaging-only change, run:

```bash
python -m compileall -q python benchmarks tests tools
git diff --check
python -m build --sdist
```

## Kernel changes

A performance change must include:

- a correctness oracle and the shapes, dtypes and routes it covers;
- eager and CUDA Graph replay where the production path captures graphs;
- paired measurements against the current implementation on the same GPU;
- raw samples, environment identity and an explanation of the physical
  mechanism;
- a clearly bounded claim: kernel, layer or end-to-end service.

Do not promote a microbenchmark winner as a serving result. Regressions or
long-tail-only improvements should be reported explicitly rather than hidden
inside an aggregate.

## Public artifacts

Do not commit private checkpoints, datasets, hostnames, credentials or internal
runtime patches. Public artifacts may contain either:

- reproducible benchmark results, which must identify a public model and a
  public conversion recipe or checkpoint; or
- environment-bound integration snapshots, which must say that they are not
  publicly reproducible and may contain only aggregate results, contract and
  content hashes, versioned environment identity and other redistributable
  metadata.

An integration snapshot must not imply public checkpoint availability, runtime
support or general workload performance. It must still identify the public
baseline, measurement scope and any quality evidence that was or was not run.

Changes to operator schemas, packed layouts or checkpoint plans must update the
relevant document under `docs/` and the changelog.
