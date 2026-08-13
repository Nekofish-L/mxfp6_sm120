# Runtime autotuning

Unknown W6A8 shapes use first-use selection from precompiled native CUTLASS
families. This is AOT kernel selection, not runtime NVRTC compilation.

The default searches all 30 families through `M=512`, where the wider search
recovers small and medium-prefill swapped-tile winners. Above `M=512`, it uses
the stable normal-family shortlist: measured family gaps are small there, and
ranking a larger set from finite samples can select a lucky but slower
candidate.

## Selection flow

`autotune_w6a8` accepts FP16/BF16 or prequantized MXFP8 activation input. Its
cache and in-process overrides are isolated by output dtype:

1. A coarse pass ranks kernel and swizzle choices.
2. The three best families refine raster and swizzle.
3. The winner is checked numerically against the deterministic fallback.
4. The full launch config is installed in the C++ dispatcher and cached by
   build, GPU, shape, output dtype and measurement policy.

Later calls use the in-process override. Later processes load the JSON cache
without profiling. Tuning and file I/O are disabled during CUDA Graph capture
and `torch.compile` tracing.

For Qwen3.5-27B TP2, the static eager-prefill path also includes two bounded
post-graph rules for `1024 < M < 2048`: the shallow-K `5120x3072` projection
selects the 64x128 ping-pong tile when its final wave is fuller than the
128x128 alternative; the deep-K `7168x5120` and `8192x5120` projections retain
Stream-K for an exceptionally short final wave. Other shapes remain on the
generic dispatcher until an offline cache is supplied.

## Measurement policy

The default cadence follows FlashInfer's three warmups and ten measured
launches per timing sample, then takes the median of three samples. It uses
warm-cache ranking by default (`MXFP6_AUTOTUNE_FLUSH_L2_MB=0`).

```bash
MXFP6_AUTOTUNE=0                    # checked-in/static dispatch only
MXFP6_AUTOTUNE=cache_only           # cache hits; static dispatch on miss
MXFP6_AUTOTUNE_VERBOSE=1            # print tuning and cache-hit decisions
MXFP6_AUTOTUNE_CACHE_DIR=/local/dir # persistent cache directory
MXFP6_AUTOTUNE_WARMUP=3             # warmup count
MXFP6_AUTOTUNE_ITERATIONS=10        # launches per timing sample
MXFP6_AUTOTUNE_REPEATS=3            # median sample count
MXFP6_AUTOTUNE_FLUSH_L2_MB=0        # warm-cache ranking; 256 flushes L2
MXFP6_AUTOTUNE_ALL_FAMILIES_MAX_M=512
MXFP6_AUTOTUNE_EXACT=1              # retune checked-in exact shapes
MXFP6_AUTOTUNE_EXHAUSTIVE=1         # refine every eligible family
```

The default cache is `$XDG_CACHE_HOME/mxfp6/autotune`, or
`~/.cache/mxfp6/autotune` when `XDG_CACHE_HOME` is unset.

## Serving policy

Generate or distribute the cache during a separate warmup job, then launch
workers with `MXFP6_AUTOTUNE=cache_only`. In this mode, a cache miss falls back
immediately to the native static dispatcher instead of starting a profiler in
a live request. Even `force=True` cannot start profiling in cache-only mode;
use `MXFP6_AUTOTUNE=1` only for the separate cache-generation job.

An example offline retuning command is:

```bash
MXFP6_AUTOTUNE_WARMUP=5 MXFP6_AUTOTUNE_ITERATIONS=20 \
MXFP6_AUTOTUNE_REPEATS=5 python3 benchmarks/retune_runtime.py \
  --devices=0,1,2,3 --library build/mxfp6_torch.so
```

The profiler and timing method can influence the selected configuration. A
cached winner should therefore be independently validated under the target
CUDA Graph and workload before deployment. See
[Benchmark methodology](benchmarks.md) and
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
