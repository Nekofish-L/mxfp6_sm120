# Qwen3.5 MoE kernels

The repository includes specialized SM120 execution paths for the Qwen3.5
Mixture-of-Experts topology. The wheel exposes standalone PyTorch operators and
layer benchmarks. A separate, version-locked vLLM v0.25.1 reproducer supports
the tested Qwen3.5-35B-A3B profile.

## Execution paths

Small batches use an allocation-free router, split-K W1 with fused
SiLU-and-mul and MXFP8 activation quantization, followed by fused W2 routed and
shared-expert reduction. Larger batches use indirect routing, a TMA W1 path
over the checkpoint's interleaved gate/up projection and a sparse grouped W2
path.

The implementation preserves CUDA Graph capture requirements by using
caller-owned persistent workspaces. Dependent small-batch grids synchronize
before their first producer-dependent global-memory access.

## Batch-one API

The independent package exposes an allocation-free Qwen3.5 TP2 batch-one
schedule:

```python
import torch
import mxfp6

workspace = mxfp6.Qwen35MoeB1Workspace.allocate("cuda")

# combined_gate: [257, 2048] BF16, routed gate followed by shared gate
# w1: [257, 512, 1536] packed FP6, routed experts followed by shared expert
# w2: [257, 2048, 192] packed FP6, routed experts followed by shared expert
def moe_layer():
    return mxfp6.qwen35_moe_b1_out(
        workspace,
        hidden_states,
        combined_gate,
        w1,
        w1_scales,
        w2,
        w2_scales,
    )

moe_layer()
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = moe_layer()
```

The exact tensor layouts and accepted batch paths are validated by the tests
and real-checkpoint layer benchmark. They should not be treated as a stable
cross-model MoE ABI.

## Runtime boundary

`mxfp6` itself imports only PyTorch. The real-checkpoint comparison in
[`benchmarks/benchmark_qwen35_moe_layer.py`](../benchmarks/benchmark_qwen35_moe_layer.py)
optionally imports vLLM for the FP8 runtime baseline and its custom TP
all-reduce.

Serving evidence includes a public two-model vLLM v0.25.1 reproducer and an
independently optimized internal vLLM reference. The wheel alone does not load
an MXFP6 Hugging Face checkpoint, and neither adapter is a general runtime
backend. See [Runtime integration](runtime-integration.md).

## Benchmarking

Run the real Qwen3.5 MoE layer benchmark with two workers:

```bash
torchrun --nproc-per-node=2 \
  benchmarks/benchmark_qwen35_moe_layer.py \
  --fp8-model "$PWD/models/Qwen3.5-35B-A3B-FP8" \
  --mx-model "$PWD/models/Qwen3.5-35B-A3B-MXFP6" \
  --batch-sizes 1 2 4 8 10 12 14 16 24 32 40 48 56 64 80 96 \
  --mx-mode auto --route-stats \
  --warmup 40 --iterations 1000 --repeats 9
```

The benchmark can also emulate a TP2 shard in one process. For checked-in
results, the TP2 benchmark captures the complete MoE layer and vLLM custom
all-reduce in one CUDA Graph using real layer-0 checkpoint weights.

The current full batch table, numerical comparisons, refinement measurements,
serving metrics and exact artifacts are in
[Benchmark methodology](benchmarks.md). Performance PRs must retain the
distinction between isolated kernel, complete layer and end-to-end serving
evidence.
