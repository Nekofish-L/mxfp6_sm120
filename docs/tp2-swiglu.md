# TP2 SwiGLU producer

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

Validation on SM120 with PyTorch 2.13/CUDA 13:

- 176 producer tests pass (120 quantization padding + 56 SwiGLU cases),
  including poisoned scales and changing-input graph replay, FP16/BF16,
  M1/2/4/8/16/24/32, zero/tiny/large finite activation values.
- Mach's seven real TP2 down-shape tests match separate vLLM activation,
  FP8 codes/scales and final BF16 GEMM output exactly under graph replay.
- Fresh full-checkpoint M4/M32 teacher-forced scores (256 queries, 10,479
  target tokens each) match the preceding implementation exactly.
- Five full-checkpoint trials improve B4/B16 throughput by 1.25%/0.82%
  over the scale-initialization-only version. B32 is unchanged within
  trial variation. These development runs include prefill and are not
  HTTP serving results.

Detailed evidence and figures are maintained in the adjacent Mach repository,
`docs/tp2-optimization-results.md` and `docs/data/tp2-p1b.json`.
An earlier separate `gemm_w6a8` integration disabled PDL and regressed B32;
that integration was rejected. The accepted entry retains PDL.
