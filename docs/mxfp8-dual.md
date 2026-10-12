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
Build from `main` for the current dual-activation APIs.

On SM120, run `pytest -q tests/test_mxfp8_dual.py`. Tests use random inputs,
an independent exponent/scale oracle, eager execution and changing-input,
poisoned-buffer CUDA Graph replay for every supported shape. The GEMM oracle checks mathematical accuracy within BF16 rounding.

## Performance and activation fidelity

On RTX 5090, CUDA 13.0, PyTorch 2.13 and Triton 3.7.1, the synthetic sweep measures every integer M1–128 for both projections. D/S is dual latency divided by single latency; a ratio above 1 means slower execution. Hot weights use 16 calls per Graph; cold weights use one call with a 256 MiB L2 eviction outside timing. Results aggregate nine randomized paired trials.

| 投影 | 纯GEMM hot D/S | 纯GEMM cold D/S | 量化+GEMM hot D/S | 量化+GEMM cold D/S |
|---|---:|---:|---:|---:|
| QKVZ | 1.507× | 1.276× | 1.462× | 1.250× |
| gate/up | 1.684× | 1.326× | 1.643× | 1.326× |

| 投影 | single relative L2 | dual relative L2 | 误差改善倍数 |
|---|---:|---:|---:|
| QKVZ | 2.670% | 0.166% | 16.10× |
| gate/up | 2.669% | 0.166% | 16.10× |

误差为128个bs的中位数。参考是BF16输入对应的FP32 GEMM，使用同一份反量化MXFP8权重，TF32关闭；衡量激活近似和BF16输出舍入，不包含原始BF16权重转MXFP8的误差。

Activation error is measured against FP32 GEMM with BF16 activations and the same decoded MXFP8 weights; it excludes weight-conversion error. The 256-point correctness sweep includes independent scale oracles and changing-input poisoned-buffer Graph replay.

![Dual GEMM hot/cold comparison](mxfp8-dual-bs1-128-gemm.svg)

### Real-model activations

The October 9 Qwen3.5-4B study uses real BF16 activations and natural caches across 32 gate/up and 24 QKVZ projections, with PDL off. ISL256/OSL32 covers all request batches1–128 in both orders; ISL3000/OSL384 checks seven representative batches. Graph padding maps requests to physical M (for example BS3→M4, BS33→M40).

GEMM timings exclude quantization and use external events inside the Graph. Steady-state timings discard the first three and final two decode steps. Backbone timings remove internal events and exclude LM head, sampling, CPU scheduling and HTTP. Across the 128 request batches, steady-state dual GEMM latency is 27.3% higher for QKVZ and 31.8% higher for gate/up; backbone latency is 8.2% higher.

| 请求 BS 范围 | 稳态 QKVZ D/S | 稳态 gate/up D/S | 无内部事件 backbone D/S |
|---|---:|---:|---:|
| 1-32 | 1.060 | 1.013 | 1.007 |
| 33-64 | 1.180 | 1.064 | 1.034 |
| 65-128 | 1.448 | 1.673 | 1.148 |

![Dual GEMM with real activations and natural caches](mxfp8-dual-actual-inference.svg)

### Data and tools

- Synthetic sweep: [JSON](../benchmarks/results/mxfp8_dual_bs1_128.json), [CSV](../benchmarks/results/mxfp8_dual_bs1_128.csv), [benchmark](../benchmarks/benchmark_mxfp8_dual.py).
- Natural-cache study: [JSON](../benchmarks/results/mxfp8_dual_actual_inference_bs1_128.json), [CSV](../benchmarks/results/mxfp8_dual_actual_inference_bs1_128.csv), [inference harness](../benchmarks/benchmark_mxfp8_dual_inference.py), [report tool](../benchmarks/report_mxfp8_dual_inference.py).

The JSON retains per-batch absolute latencies, physical-row mappings, repeat samples, environment and numerical checks. Use the native test suite for the current source revision.
