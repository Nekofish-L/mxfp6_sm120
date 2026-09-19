# TP2 attention sigmoid gate / MXFP8 producer

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
