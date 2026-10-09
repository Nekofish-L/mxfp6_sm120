# Dual MXFP8 activations at batches 1–128

The opt-in `mxfp6.mxfp8_dual` module uses a high E4M3 activation limb and
an E4M3 residual limb with separate E8M0 scales. The residual is rounded to
BF16 before quantization. GEMM maintains two independent FP32 accumulators,
adds them after the reduction, and rounds the sum once to BF16.

Every integer M from **1 to 128** is supported for both `(N,K)=(18432,2560)`
and `(12288,2560)`. The actual M is passed to a single native GEMM; there
is no input-padding copy or batch splitting. TMA handles partial value tiles,
scale reads mask invalid rows, and output stores respect the logical M.

| Shape interval | Kernel tile (weight rows × activation rows × K) |
| --- | --- |
| M1–32, N18432, K2560 | MLP, swapped operands, 128x16x256, one stage |
| M1–32, N12288, K2560 | QKVZ, swapped operands, 128x16x256, one stage |
| M33–128, either N, K2560 | cfg2, 128x32x128, two value stages, register scales |

Each launch uses 128 threads. The original M32 MLP/QKVZ and M64 QKVZ
arithmetic is preserved. Other N/K or M outside 1–128 raise an error.
Existing MXFP6 and single-activation MXFP8
dispatch remain unchanged. This module does not select model layers or provide
a complete serving profile.

```python
import torch
from mxfp6 import mxfp8_dual

# x: finite contiguous CUDA BF16 [M,2560], M=1–128
# weight: contiguous CUDA E4M3FN [N,2560]
# s_weight: flat CUDA uint8 [N*80], CUTLASS 128x4 swizzled E8M0 scales
hi, s_hi, res, s_res = mxfp8_dual.quantize(x)
out = torch.empty(x.shape[0], weight.shape[0], device=x.device, dtype=torch.bfloat16)
mxfp8_dual.gemm_out(hi, weight, s_hi, s_weight, res, s_res, out)
```

Pass `use_pdl=True` to `gemm_out` to enable programmatic dependent launch
for either tile family and every supported batch. The kernel waits before
reading both activation limbs/scales and signals following PDL kernels.
The flag defaults to `False`. The current Triton dual quantizer uses ordinary
stream ordering; a fused activation/quantization producer may signal readiness
for the PDL GEMM, which still waits for that producer's stores to complete.

For CUDA Graph capture, allocate buffers and warm both operations first; capture
`quantize_out(x,hi,s_hi,res,s_res)` and `gemm_out(...)`. Inputs may change in
place between replays. Concurrent calls need independent output and quantizer
buffers. Scale buffers have exactly 10240 uint8 elements, including padding
written by the quantizer. GEMM requires contiguous, 16-byte-aligned operands
on the same SM120 device, distinct output storage, and exact weight/scale sizes.
Nonfinite quantizer input is unsupported; input finiteness is a caller contract
and is not checked with a device synchronization on each launch.

Build from source using the installation workflow in `CONTRIBUTING.md`. All
kernel families and `mxfp8_sm120::dual_gemm_out` are linked into `mxfp8_torch`,
which the normal package loader loads. No separate extension or runtime adapter
is required. The quantizer lazily imports the package's Triton dependency.
The native implementation shares one templated small-batch kernel across both
projection widths in `mxfp8_dual_small.cu`; `mxfp8_dual_pipereg.cu` handles
M33–128. Tensor validation lives in `mxfp8_dual_extension.cu`, and both kernels
reuse the scale/layout helpers and TMA descriptor builder in
`include/mxfp8_gemm/dual.cuh`. The public variant names remain unchanged.
These APIs are unreleased source additions; the original 0.2.1 wheel does not
provide them.

On SM120, run `pytest -q tests/test_mxfp8_dual.py`. Tests use random inputs,
an independent exponent/scale oracle, eager execution and changing-input,
poisoned-buffer CUDA Graph replay for every supported shape. The GEMM oracle
checks mathematical accuracy within BF16 rounding; bitwise comparison against
a previous implementation is a separate check.

Original fixed-shape component validation on RTX 5090 (SM120), CUDA 13.0.88,
PyTorch 2.13.0+cu130 and Triton 3.7.1 passed all six tests. A paired check against the three original
prototype binaries also found zero differing output bytes for every shape on
zero, small, random, large and negative inputs, including changing-input graph
replay. Both activation limbs and scale buffers matched exactly. This establishes
the tested component boundary; it does not establish whole-model fidelity or
end-to-end throughput for a serving integration.

The batch extension tests every M for both projections, with an independent
scale oracle, poisoned-buffer changing-input Graph replay, and a mathematical
GEMM reference. The [single/dual comparison](mxfp8-dual-bs1-128.md) records
all 256 batch/projection points, GEMM-only and quantization+GEMM timings,
hot/cold weight-cache measurements, and activation approximation error.
