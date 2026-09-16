# TP2 activation scale initialization

The allocating packed quantizer initializes only padding in the quantization
kernel; the owning quantization subgroup writes each logical scale. These
sets are disjoint and cover the complete physical scale allocation. No barrier,
workspace reset, quantization arithmetic or packed layout is changed. Both
FP6/FP8 allocation paths retain initialized UE8M0 padding bytes (127).
The dual/out quantizers keep their existing initialization contracts.

The grid also covers the scale allocation with at least one thread per scale
byte. Reusing only the old small-M grid lost performance because too few CTAs
serialized padding writes. That prototype is not retained.

Validation on RTX 5090: 120 FP16/BF16 × row/K cases with an independent
FP64/frexp exponent and scale-layout oracle; changing-input captured replay
poisons output/scales before every replay. The original library passes the same
oracle. FP6 codec, scales, dynamic quantization, float GEMM and nondefault-stream
checks pass. The existing stress harness passes 28 boundary row sizes,
10,000 randomized launches and 1,000 dual-stream launches, with zero fallback
launches and reset barriers. The rebuilt wheel contains the tested shared library.

[Raw preparation microbenchmarks](tp2-scale-initialization.json) use the same GPU.
Typical 5120-wide activation preparation falls from 2.1 to 1.6 us. This alone
is not a serving result. Mach's full-model TP2 test (five repeats, 2048 input,
1025 output at B4/16/32) improves throughput 2.32%/3.31%/3.48%; B1's 1.13%
change is inconclusive relative to its variance. Physical M4/M32 teacher-forced
fidelity remains exactly equal on 256 queries / 10,479 target tokens.
See the adjacent Mach repository's `docs/tp2-optimization-results.md` for the
full workload contract and per-rank traces. HTTP serving validation is separate.
