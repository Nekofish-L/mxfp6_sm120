# Fixed-shape dual MXFP8 activations

The opt-in `mxfp6.mxfp8_dual` module uses a high E4M3 activation limb and
an E4M3 residual limb with separate E8M0 scales. The residual is rounded to
BF16 before quantization. GEMM maintains two independent FP32 accumulators,
adds them after the reduction, and rounds the sum once to BF16.

Only these `(M,N,K)` shapes are implemented:

| Shape | Fixed kernel |
| --- | --- |
| `(32,18432,2560)` | MLP, swapped operands, 128x16x256, one stage |
| `(32,12288,2560)` | QKVZ, swapped operands, 128x16x256, one stage |
| `(64,12288,2560)` | QKVZ, cfg2, 128x32x128, two value stages, register scales |

Each launch uses 128 threads and preserves its own kernel implementation.
Other shapes raise an error. Existing MXFP6 and single-activation MXFP8
dispatch remain unchanged. This module does not select model layers or provide
a complete serving profile.

```python
import torch
from mxfp6 import mxfp8_dual

# x: finite contiguous CUDA BF16 [32,2560] or [64,2560]
# weight: contiguous CUDA E4M3FN [N,2560]
# s_weight: flat CUDA uint8 [N*80], CUTLASS 128x4 swizzled E8M0 scales
hi, s_hi, res, s_res = mxfp8_dual.quantize(x)
out = torch.empty(x.shape[0], weight.shape[0], device=x.device, dtype=torch.bfloat16)
mxfp8_dual.gemm_out(hi, weight, s_hi, s_weight, res, s_res, out)
```

For CUDA Graph capture, allocate buffers and warm both operations first; capture
`quantize_out(x,hi,s_hi,res,s_res)` and `gemm_out(...)`. Inputs may change in
place between replays. Concurrent calls need independent output and quantizer
buffers. Scale buffers have exactly 10240 uint8 elements, including padding
written by the quantizer. GEMM requires contiguous, 16-byte-aligned operands
on the same SM120 device, distinct output storage, and exact weight/scale sizes.
Nonfinite quantizer input is unsupported; input finiteness is a caller contract
and is not checked with a device synchronization on each launch.

Build from source using the installation workflow in `CONTRIBUTING.md`. All
three kernels and `mxfp8_sm120::dual_gemm_out` are linked into `mxfp8_torch`,
which the normal package loader loads. No separate extension or runtime adapter
is required. The quantizer lazily imports the package's Triton dependency.
These APIs are unreleased source additions; the original 0.2.1 wheel does not
provide them.

On SM120, run `pytest -q tests/test_mxfp8_dual.py`. Tests use random inputs,
an independent exponent/scale oracle, eager execution and changing-input,
poisoned-buffer CUDA Graph replay for every supported shape. The GEMM oracle
checks mathematical accuracy within BF16 rounding; bitwise comparison against
a previous implementation is a separate check.

Component validation on RTX 5090 (SM120), CUDA 13.0.88, PyTorch 2.13.0+cu130
and Triton 3.7.1 passed all six tests. A paired check against the three original
prototype binaries also found zero differing output bytes for every shape on
zero, small, random, large and negative inputs, including changing-input graph
replay. Both activation limbs and scale buffers matched exactly. This establishes
the tested component boundary; it does not establish whole-model fidelity or
end-to-end throughput for a serving integration.
