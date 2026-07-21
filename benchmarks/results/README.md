# Reviewed benchmark results

`native_w6a8_dispatch.json` is the reviewed production manifest. It records:

- the FP16/BF16 16-to-8 activation and packed-W6 weight formats;
- exact small-batch tile, scheduler, raster and swizzle selections;
- the large-batch native CUTLASS policy;
- the current RTX 5090 GEMM-only comparison against Humming W6A8 and vLLM
  block-FP8 W8A8.

Production dispatch never selects Humming. Unknown W6A8 shapes are selected
from the compiled native portfolio on first use and cached per build, GPU,
shape and measurement policy. Machine-local autotune caches and raw profiler
CSV files are generated artifacts and are intentionally ignored.

Reproduce the checked-in warm-cache comparison with:

```bash
CUDA_VISIBLE_DEVICES=0 MXFP6_AUTOTUNE=0 \
python3 benchmarks/benchmark.py \
  --library build/mxfp6_torch.so \
  --activation-input bf16 \
  --compare-humming --compare-fp8 \
  --warmup 20 --iterations 100 --flush-l2-mb 0
```

Every speedup uses standalone GEMM kernel time. Activation quantization,
persistent weight preparation, Humming first-use JIT and host gaps are not part
of the ratio.
