# CUTLASS patch queue

The CUTLASS submodule is pinned to upstream commit
`e6233cbac5d7c7a865c19c91cd684ceece19513c`.

- `0001-sm120-mxfp6-small-tile-runtime.patch` fixes the scale-factor layout
  for TileM=32/64 and the shared-memory copy atoms for narrow TileN=8/16
  production kernels. It is required at build and runtime.
- `0002-sm120-mxfp6-profiler-search.patch` adds the candidate generation,
  static scheduler, and minimal-library options used to reproduce the
  exhaustive profiler search. It is not required by an installed wheel.
- `0003-sm120-streamk-persistent-workspace.patch` makes the SM120 Stream-K
  barrier self-reset after the final accumulator consumer and exposes the
  reduction/barrier workspace sizes needed by the persistent arena. It is
  required at runtime.

Run `scripts/apply_cutlass_patches.sh` after initializing the submodule. The
script is idempotent, verifies the pinned upstream commit, and supports
`--check`, `--reverse`, and `--runtime-only`.
