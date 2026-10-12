# Activation producer contracts

These explicit APIs prepare the packed activation layout consumed by native W6A8 GEMM. Model-level eligibility, serving gains and runtime configuration belong to [Mach's producer integration guide](https://github.com/troycheng/vllm-mach/blob/main/docs/dense-producer-fusion.md). Operator rounding and graph contracts remain defined here.

## SwiGLU

`gemm_from_swiglu` consumes contiguous `[M,2K]` gate/up activations, rounds
SiLU and the product to the source dtype, produces the existing packed
MXFP8 layout, and calls the existing PDL-enabled W6A8 dispatch. The producer
initializes all padding scales on every invocation. It preserves the
separate vLLM activation's `up + positive-zero` signed-zero boundary.
The standalone `silu_and_mul_mxfp8` API exposes the same producer.

No weight format, collective, residual operation, or output precision changes.
Workspace planning and warmup use the existing down-projection `[M,K]`
contract. The new entry uses installed dispatch overrides; it does not run
a separate autotuner during graph capture.

Validation covers FP16/BF16, M1/2/4/8/16/24/32, poisoned scales and changing-input graph replay. Mach's seven TP2 down-shape comparisons match separate vLLM activation, FP8 codes/scales and final BF16 GEMM output exactly. Use the PDL-enabled fused entry for integration; a separate ordinary GEMM call has a different launch dependency contract.

## GDN norm

`gemm_from_gdn` fuses gated RMSNorm/MXFP8 preparation for BF16 `[M,24,128]` at M1/2/4/8/16/24/32 using the pinned vLLM 0.29 norm reduction layout. Workspace planning follows the resulting projection shape; warm the configured call before capture. The explicit entry is model scoped; it does not imply eligibility for every GDN implementation.

## Attention sigmoid gate

`mxfp6.attention.sigmoid_gate_mxfp8(x, gate)` returns uint8 FP8 codes and
packed UE8M0 scales for native SM120 TP2 attention output projection.
Supported inputs are contiguous BF16 `[M,3072]`, with
`M ∈ {1,2,4,8,16,24,32}`. The BF16 gate may have a larger row stride.
The result feeds the existing `gemm_w6a8_pdl` entry point, without another
activation quantizer or a collective.

The producer preserves eager PyTorch's FP32 exp/division, sigmoid→BF16,
product→BF16, and block32 MXFP8 quantization boundaries. It initializes all
unused packed-scale rows to 127 within the producer. `rounded_out` is an
optional diagnostic buffer; production does not materialize the product.
It does not modify QKV, QK norm, RoPE, attention or KV-cache management.

`tests/test_attention_gate_quant.py` checks every finite BF16 gate value and every finite BF16 attention
value (including subnormals) against five gate values,
14 contiguous/strided cases with 120 changing-input poisoned graph replays,
and seven projection shapes with 120 changing-input PDL graph replays.
Rounded BF16 values and quantization codes/scales must be bitwise equal;
the independent FP64 analytic gate check uses rtol 0.008 and atol 1e-34
(the absolute allowance covers FP32 sigmoid underflow on saturated inputs).
Projection outputs must exactly match the public float-input GEMM route.

This is an opt-in producer. Model-level acceptance belongs to the Mach
TP2 optimization measurements, with default BF16 activations/head and
FP32 recurrent state held fixed.

## Packed-scale initialization

The allocating packed quantizer initializes only padding in the quantization
kernel; the owning quantization subgroup writes each logical scale. These
sets are disjoint and cover the complete physical scale allocation. No barrier,
workspace reset, quantization arithmetic or packed layout is changed. Both
FP6/FP8 allocation paths retain initialized UE8M0 padding bytes (127).
The dual/out quantizers keep their existing initialization contracts.

Validation on RTX 5090: 120 FP16/BF16 × row/K cases with an independent
FP64/frexp exponent and scale-layout oracle; changing-input captured replay
poisons output/scales before every replay. The original library passes the same
oracle. FP6 codec, scales, dynamic quantization, float GEMM and nondefault-stream
checks pass. The existing stress harness passes 28 boundary row sizes,
10,000 randomized launches and 1,000 dual-stream launches, with zero fallback
launches and reset barriers. The rebuilt wheel contains the tested shared library.

[Raw preparation timings](tp2-scale-initialization.json) record the component measurement. Full-model experiment contracts and fidelity checks remain in [Mach's development results](https://github.com/troycheng/vllm-mach/blob/7e841db8dc07b1d8329d10ef14ea0cbe45a3f074/docs/tp2-optimization-results.md); component timings are not HTTP service gains.
