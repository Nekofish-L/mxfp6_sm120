# Reviewed benchmark results

`native_w6a8_dispatch.json` is the reviewed production manifest. It records:

- the FP16/BF16 16-to-8 activation and packed-W6 weight formats;
- native FP16 and BF16 GEMM epilogues;
- exact small-batch tile, scheduler, raster and swizzle selections;
- the large-batch native CUTLASS policy;
- the historical RTX 5090 GEMM-only comparison against Humming W6A8 and the
  current comparison against vLLM block-FP8 W8A8.

`persistent_workspace.json` compares Stream-K launches with per-call CUTLASS
initialization against the self-resetting persistent arena.
`runtime_autotune_v4.json` records the post-change FP16/BF16 retune across the
50 Qwen3.5-27B TP2 shapes, including the measurement policy and timings.
`qwen35_moe_tp2.json` records CUDA-Graph whole-layer Qwen3.5 MoE latency,
numerics, and the measured batch-dependent schedule on two RTX 5090 GPUs.

Humming is not a submodule, dependency or backend of the current project. Its
checked-in measurements are retained as an external historical reference from
[`inclusionAI/humming@694298e9`](https://github.com/inclusionAI/humming/tree/694298e9eb25ffdfc088353b49ba537ebf304479).
Unknown W6A8 shapes are selected from the compiled native portfolio on first
use and cached per build, GPU, shape and measurement policy. Machine-local
autotune caches and raw profiler CSV files are generated artifacts and are
intentionally ignored.

Reproduce the current native and vLLM warm-cache comparison with:

```bash
CUDA_VISIBLE_DEVICES=0 MXFP6_AUTOTUNE=0 \
python3 benchmarks/benchmark.py \
  --library build/mxfp6_torch.so \
  --activation-input bf16 \
  --output-dtype fp16 \
  --compare-fp8 \
  --warmup 20 --iterations 100 --flush-l2-mb 0
```

Every speedup uses standalone GEMM kernel time. Activation quantization,
persistent weight preparation and host gaps are not part of the ratio. The
archived Humming ratios additionally exclude its first-use JIT.
