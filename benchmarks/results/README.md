# Benchmark artifacts

| Scope | Key result | Artifact |
|---|---:|---|
| Public Qwen3.5-27B TP2 serving | +9.11% output tok/s | `qwen35_public_vllm_tp2.json` |
| Public Qwen3.5-35B-A3B TP2 serving | +16.95% output tok/s | `qwen35_public_vllm_tp2.json` |
| Public conversion and TP2 smoke | both models pass | `qwen35_public_conversion_validation.json` |
| Dense Qwen3.5 GEMM shapes | 1.633x vs vLLM block-FP8 | `native_w6a8_dispatch.json` |
| Complete Qwen3.5 MoE layer | 1.278x vs vLLM FP8 MoE | `qwen35_moe_tp2.json` |
| Internal Qwen3.5-27B serving | +22.44% output tok/s | `qwen35_27b_service_tp2.json` |
| Internal Qwen3.5-35B-A3B serving | +13.58% output tok/s | `qwen35_moe_service_tp2.json` |
| B4 MoE implementation refinement | paired scalar/vector result | `qwen35_moe_b4_vector.json` |

The public serving artifact contains all four fresh
FP8/MXFP6/MXFP6/FP8 blocks, workload parameters and latency guardrails. Its
35B-A3B workload is public `random-mm`, not the private real-multimodal
workload used by the internal serving result.

The internal serving artifacts are retained as bounded research evidence.
They do not imply support in unmodified vLLM or SGLang. The 27B workload did
not score task quality; the 35B quality section measures reference-output
fidelity rather than audited business accuracy.

Other JSON files record autotuning, persistent workspace and historical kernel
experiments. Machine-local caches and raw profiler exports are intentionally
not checked in.

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
