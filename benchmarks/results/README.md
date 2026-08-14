# Reviewed benchmark results

The checked-in results intentionally separate isolated implementation evidence
from full-serving evidence:

| Model | Evidence level | Artifact |
|---|---|---|
| Qwen3.5-27B | five real linear-layer GEMM shapes | `native_w6a8_dispatch.json` |
| Qwen3.5-27B | full TP2 serving | `qwen35_27b_service_tp2.json` |
| Qwen3.5-35B-A3B | complete TP2 MoE layer | `qwen35_moe_tp2.json` |
| Qwen3.5-35B-A3B | full TP2 serving and reference fidelity | `qwen35_moe_service_tp2.json` |
| Qwen3.5-35B-A3B | B4 MXFP6 implementation refinement | `qwen35_moe_b4_vector.json` |
| Qwen3.5-27B and 35B-A3B | experimental public vLLM TP2 serving | `qwen35_public_vllm_tp2.json` |
| Qwen3.5-27B and 35B-A3B | public conversion, load and CUDA Graph validation | `qwen35_public_conversion_validation.json` |

`native_w6a8_dispatch.json` is the reviewed dense production manifest. It
records:

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
`qwen35_27b_service_tp2.json` records the full-model Qwen3.5-27B TP2 serving
comparison against the official community FP8 checkpoint, including the
fixed-token workload, server-usage metric contract and four fresh service
lifecycle samples.
`qwen35_moe_tp2.json` records CUDA-Graph whole-layer Qwen3.5 MoE latency,
numerics, and the measured batch-dependent schedule on two RTX 5090 GPUs.
`qwen35_moe_service_tp2.json` records the full-model Qwen3.5-35B-A3B TP2
comparison against the official community FP8 checkpoint, plus the separately
measured 742-case reference-output fidelity summary.
`qwen35_moe_b4_vector.json` records the paired scalar/vector B=4 TP2 layer
comparison with programmatic router-to-W1 dependency ordering.
`qwen35_public_vllm_tp2.json` records the public-image FP8/MXFP6/MXFP6/FP8
reproduction. Its 27B workload matches the published fixed-token contract. Its
35B-A3B workload is a deterministic public `random-mm` replacement and is not
the private real-multimodal workload used by the headline service result.
`qwen35_public_conversion_validation.json` records a full conversion of local
mirrors of both public source models, byte-exact shard comparison, and TP2
CUDA Graph generation smokes. Checkpoints are not distributed here.

The serving manifests are environment-bound integration snapshots from a
pinned internally maintained vLLM integration. They are not publicly
reproducible MXFP6 checkpoint recipes or general workload claims, and they do
not promise that the wheel alone can load the checkpoint in an unmodified vLLM
or SGLang release. The 27B workload did not score task quality; the 35B quality
section measures reference-output fidelity rather than audited business-task
accuracy.

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
