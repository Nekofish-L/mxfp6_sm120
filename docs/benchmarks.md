# Native kernel benchmarks

RTX 5090 measurements for Dense W6A8 and the complete TP2 MoE layer. Full-service results are in [model performance](performance.md).

## Dense GEMM

The Dense benchmark covers the five Qwen3.5-27B TP2 projection shapes:

| Layer | N | K |
|---|---:|---:|
| GDN input projection | 8192 | 5120 |
| GDN output projection | 5120 | 3072 |
| Full-attention QKV/gate projection | 7168 | 5120 |
| MLP gate/up projection | 17408 | 5120 |
| MLP down projection | 5120 | 8704 |

It used BF16 input, FP16 output, 20 warmups, 100 measured iterations, warm L2
state and disabled runtime autotuning. The ratio compares vLLM block-scaled FP8
GEMM latency with MXFP6 GEMM latency; dynamic activation quantization is reported
separately and excluded from the ratio.

| M | Geometric mean speedup over five shapes |
|---:|---:|
| 1 | 1.977x |
| 2 | 1.983x |
| 4 | 1.981x |
| 8 | 1.998x |
| 16 | 1.603x |
| 24 | 1.514x |
| 32 | 1.532x |
| 64 | 1.232x |
| 96 | 1.555x |
| 512 | 1.829x |
| 1024 | 1.761x |
| 2048 | 1.697x |
| 4096 | 1.590x |
| 8192 | 1.588x |
| Overall | 1.688x |

All 70 native outputs matched the decoded FP32 matmul reference exactly after
FP16 output conversion. Exact static dispatch now covers the five measured
projection shapes at `M=24`, improving that batch's geometric-mean ratio from
0.717x to 1.514x. The full rows are in
[`qwen35_dense_gemm_current_main.json`](../benchmarks/results/qwen35_dense_gemm_current_main.json).

Reproduce the sweep with:

```bash
CUDA_VISIBLE_DEVICES=0 MXFP6_AUTOTUNE=0 \
python3 benchmarks/benchmark.py \
  --shapes 1x8192x5120,2x8192x5120 \
  --library build/mxfp6_torch.so \
  --activation-input bf16 --output-dtype fp16 \
  --compare-fp8 --check-all \
  --warmup 20 --iterations 100 --flush-l2-mb 0
```

Pass the full `(M,N,K)` list from the artifact for the 70-shape matrix. The
abbreviated command above shows the input format without duplicating a long
generated argument in the documentation.

## Complete Qwen3.5 MoE layer

The TP2 benchmark captures routing, routed and shared experts, top-k weighting,
reduction and vLLM custom all-reduce in one CUDA Graph. It uses real layer-0
weights. Each result is the median of nine paired repeats after 40 warmups and
1000 graph replays; each distributed sample is the slower rank.

| Token batch | FP8 layer | MXFP6 layer | Speedup | Relative RMS | Cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.3 us | 16.4 us | 1.787x | 0.1117 | 0.9937 |
| 2 | 32.8 us | 20.5 us | 1.601x | 0.1152 | 0.9936 |
| 4 | 32.8 us | 25.0 us | 1.313x | 0.0912 | 0.9958 |
| 8 | 54.5 us | 38.6 us | 1.411x | 0.0946 | 0.9955 |
| 16 | 123.7 us | 90.3 us | 1.370x | 0.0903 | 0.9959 |
| 24 | 162.4 us | 127.1 us | 1.278x | 0.0980 | 0.9952 |
| 32 | 188.4 us | 147.1 us | 1.281x | 0.0949 | 0.9955 |
| 64 | 238.9 us | 188.3 us | 1.269x | 0.0980 | 0.9952 |
| 96 | 264.6 us | 207.0 us | 1.278x | 0.0994 | 0.9951 |

The table uses the original nine-batch artifact (1.241–1.787×) except M4, which is the later [B4 cluster update](../benchmarks/results/qwen35_moe_b4_cluster_tp2.json) at 1.313×. The combined table therefore spans 1.269–1.787×; these two implementation snapshots remain separate in the artifacts.

The recorded routed expert IDs matched the FP8 checkpoint at every batch. The
run recorded this comparison but did not configure a mismatch as a hard process
failure. The [artifact](../benchmarks/results/qwen35_moe_layer_current_main.json)
retains that setting, every result and all paired timing samples.

```bash
CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=lo \
torchrun --master-addr 127.0.0.1 --nproc-per-node=2 \
  benchmarks/benchmark_qwen35_moe_layer.py \
  --fp8-model /models/Qwen3.5-35B-A3B-FP8 \
  --mx-model /models/Qwen3.5-35B-A3B-MXFP6 \
  --batch-sizes 1 2 4 8 16 24 32 64 96 \
  --mx-mode auto --no-small-separate-shared \
  --b4-packed-vector-loads --route-stats \
  --warmup 40 --iterations 1000 --repeats 9
```

The command selects the measured combined-weight small-batch execution form.


## Evidence

The [artifact index](../benchmarks/results/README.md) links the recorded code revisions, timing samples and correctness checks. Earlier service receipts remain as raw JSON; they are not the current model-performance guide.
