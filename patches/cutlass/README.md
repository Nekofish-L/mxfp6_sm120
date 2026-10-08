# CUTLASS patch queue

The CUTLASS submodule is pinned to upstream commit
`e6233cbac5d7c7a865c19c91cd684ceece19513c`.

- `0001-sm120-mxfp6-small-tile-runtime.patch` fixes the scale-factor layout
  for TileM=32/64 and the shared-memory copy atoms for narrow TileN=8/16
  production kernels. For narrow-N Pingpong tiles, the MMA atom layout is
  limited to four warps; cooperative tiles retain eight. This also enables
  Pingpong 128x8 tiles without violating the kernel thread-count contract.
  It is required at build and runtime.
- `0002-sm120-mxfp6-profiler-search.patch` adds the candidate generation,
  static scheduler, and minimal-library options used to reproduce the
  exhaustive profiler search. It is not required by an installed wheel.
- `0003-sm120-streamk-persistent-workspace.patch` makes the SM120 Stream-K
  barrier self-reset after the final accumulator consumer and exposes the
  reduction/barrier workspace sizes needed by the persistent arena. It is
  required at runtime.

- `0004-sm120-single-stage-mainloop.patch` permits one shared-memory stage
  in the SM120 block-scaled register-sourced mainloop. The consumer copies
  the last fragment to registers before releasing the buffer and waits for
  the next producer phase before reusing it. This enables tiles such as
  256x128x128 whose double-buffered storage exceeds the device limit.
  Correctness includes one/multiple K tiles, partial M tiles, and changing
  input CUDA graphs; covered by `tests/test_mxfp8.py`.

- `0005-sm120-static-problem-shape.patch` materializes static problem extents
  as runtime integers when constructing TMA descriptors. This removes the
  descriptor type mismatch for compile-time M/K specialization in the SM120
  block-scaled mainloop and its TMA epilogue. The experiment passes 120
  shape/config checks, six changing-input graphs, and synccheck (zero errors);
  production coverage is in `tests/test_mxfp8.py`.

- `0006-sm120-pdl-release.patch` enables early dependent-grid release after
  MMA in the SM120 cooperative and pingpong kernels. These schedulers do not
  expose the SM90 `is_last_tile` query; dependent consumers still wait for
  the complete grid, including remaining tiles, reductions and output stores.
  It is required for the opt-in MXFP8 PDL path and is covered by
  `tests/test_mxfp8_pdl.py` and the ordinary GEMM regressions.

Run `scripts/apply_cutlass_patches.sh` after initializing the submodule. The
script is idempotent, verifies the pinned upstream commit, and supports
`--check`, `--reverse`, and `--runtime-only`.
