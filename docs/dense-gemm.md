# Dense GEMM

MXFP6 SM120 implements a native block-scaled mixed-precision operation:

```text
D[M, N] = A[M, K] @ B[N, K].T
```

Persistent weights are bit-packed E3M2 values with one UE8M0 scale per 32
values. FP16 or BF16 activations are dynamically mapped to E4M3/UE8M0 and
consumed by a native CUTLASS W6A8 kernel. Accumulation is FP32; FP16 and BF16
outputs are written directly by the epilogue.

The 16-to-8 activation mapping removes repeated activation-side six-bit
unpacking from the MMA mainloop while preserving the six-bit storage and
bandwidth advantage of persistent weights. A packed-MXFP6 activation
compatibility API remains available for existing callers.

## Supported shapes

The public dense operator accepts:

- `M > 0`;
- `N > 0` and `N % 8 == 0`;
- `K > 0` and `K % 128 == 0`.

Checked-in exact overrides cover the five Qwen3.5-27B TP2 linear layers at
`M=1,16,32,64,96,512,1024,2048,4096,8192`. Swapped x8, x16 and x32 tiles keep
the small activation batch in the tensor-core N dimension. Larger batches use
normal 64x64, 64x128 and 128x128 tiles. The portfolio includes ping-pong,
cooperative, static-persistent and selective Stream-K scheduling.

## Basic API

Quantize a weight once and reuse it across calls:

```python
import torch
import mxfp6

m, n, k = 32, 8192, 5120
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

packed_weight = mxfp6.quantize_mxfp6(weight)
output = mxfp6.gemm(a, packed_weight, out_dtype=torch.bfloat16)
```

`out_dtype` accepts `torch.float16` (the default) or `torch.bfloat16`.

## Explicit activation quantization

The activation quantizer and W6A8 GEMM can be invoked separately for explicit
prewarming, activation reuse or CUDA Graph capture:

```python
quantized_a = mxfp6.quantize_activation(a)
mxfp6.autotune_w6a8(
    quantized_a, packed_weight, out_dtype=torch.bfloat16
)
output = mxfp6.gemm_w6a8(
    quantized_a, packed_weight, out_dtype=torch.bfloat16
)
```

The public warmup entry accepts either the original FP16/BF16 activation or an
`MXFP8Tensor`, optionally runs dtype-specific autotuning, executes the
production launch path and synchronizes before returning:

```python
config = mxfp6.warmup_w6a8(
    a,
    packed_weight,
    out_dtype=torch.bfloat16,
    iterations=3,
    autotune=True,
)
```

See [Autotuning](autotuning.md) before enabling first-use profiling in a
serving worker.

## Persistent workspace planning

Stream-K kernels can reuse a self-resetting workspace after all production
shapes and selected launch configurations have been warmed:

```python
mxfp6.begin_workspace_planning(device)
for activation, weight in production_problems:
    mxfp6.warmup_w6a8(activation, weight, autotune=True)
stats = mxfp6.finalize_workspace_planning(device)
```

Planning launches use CUTLASS's compatibility initialization path. After
finalization, each CUDA stream receives a fixed lane whose barrier tail is
cleared once; subsequent launches use `update()` and do not enqueue a workspace
memset. Warm every stream eagerly before capturing its CUDA Graph.

`mxfp6.workspace_stats()` reports arena capacity and lane count, while
`mxfp6.workspace_barriers_zero()` provides a synchronizing validation hook.

Stream-K is enabled by default. Set `MXFP6_STREAM_K=0` or
`MXFP6_STREAM_K=false` before the first MXFP6 operation to disable every
Stream-K dispatch and remove its workspace-planning requirement:

```bash
MXFP6_STREAM_K=0 python3 your_application.py
```

When disabled, built-in dispatch and runtime autotuning exclude Stream-K
kernels, and cached decisions are isolated from the default policy. The
workspace APIs remain callable but collect zero layouts.

## Serialization utilities

Logical E3M2 code and UE8M0 scale utilities are available for serialization
experiments and tests:

```python
codes = torch.randint(0, 64, (n, k), device="cuda", dtype=torch.uint8)
scales = torch.full((n, k // 32), 0x7f, device="cuda", dtype=torch.uint8)
packed_weight = mxfp6.pack_operand(codes, scales)
restored_codes, restored_scales = mxfp6.unpack_operand(packed_weight)
```

Each code uses its low six bits. UE8M0 byte `0x7f` represents scale 1.0.
These utilities expose the current in-memory layout; they are not yet a stable
portable checkpoint format. See [Checkpoint format](checkpoint-format.md).

## Performance and validation

Dense kernel and Qwen3.5-27B serving results, baseline identities, environment
provenance and reproduction commands are recorded in
[Benchmark methodology](benchmarks.md). Correctness and performance changes
should follow the evidence requirements in
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
