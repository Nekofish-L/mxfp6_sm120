# CUTLASS patch queue

The CUTLASS submodule is pinned to upstream commit
`e6233cbac5d7c7a865c19c91cd684ceece19513c`.

- `0001-sm120-mxfp6-small-tile-runtime.patch` is required to build and run
  the selected TileM=64 and TileN=8/16 production kernels correctly.
- `0002-sm120-mxfp6-profiler-search.patch` adds the candidate generation,
  static scheduler, and minimal-library options used to reproduce the
  exhaustive profiler search. It is not required by an installed wheel.

Run `scripts/apply_cutlass_patches.sh` after initializing the submodule. The
script is idempotent, verifies the pinned upstream commit, and supports
`--check`, `--reverse`, and `--runtime-only`.
