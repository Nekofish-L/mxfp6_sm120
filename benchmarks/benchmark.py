#!/usr/bin/env python3
"""Correctness and performance benchmark for torch.ops.mxfp6.gemm."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "build" / "mxfp6_torch.so"
BENCHMARK_NK = (
    (5120, 8192),
    (3072, 5120),
    (5120, 7168),
    (5120, 17408),
    (8704, 5120),
)
DEFAULT_SHAPES = [
    (m, n, k) for m in (1, 16, 32, 2048) for n, k in BENCHMARK_NK
]


def parse_shapes(value: str) -> list[tuple[int, int, int]]:
    shapes = []
    for item in value.split(","):
        try:
            m, n, k = (int(x) for x in item.lower().split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid shape {item!r}; expected MxNxK"
            ) from exc
        shapes.append((m, n, k))
    return shapes


def decode_e3m2(codes: torch.Tensor) -> torch.Tensor:
    raw = codes.to(torch.int32)
    exponent = (raw >> 2) & 0x07
    mantissa = raw & 0x03
    subnormal = mantissa.float() * (2.0**-4)
    normal = torch.ldexp(1.0 + mantissa.float() / 4.0, exponent - 3)
    value = torch.where(exponent == 0, subnormal, normal)
    return torch.where((raw & 0x20) != 0, -value, value)


def make_inputs(
    m: int, n: int, k: int, seed: int
) -> tuple[torch.Tensor, ...]:
    if not ((1 <= m <= 32) or m == 2048) or n % 128 or k % 128:
        raise ValueError(
            "M must be in [1,32] or 2048; N and K must be multiples of 128"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Restrict magnitudes to <= 1.75 so large-K reference results stay finite.
    a_codes = torch.randint(0, 16, (m, k), generator=generator, dtype=torch.uint8)
    b_codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.uint8)
    a_codes |= torch.randint(0, 2, (m, k), generator=generator, dtype=torch.uint8) << 5
    b_codes |= torch.randint(0, 2, (n, k), generator=generator, dtype=torch.uint8) << 5

    # Pack both logical [M,K] / [N,K] arrays in their displayed order. The
    # CUTLASS B stride makes the second buffer the transposed GEMM operand.
    # Exercise the production CUDA conversion path. Conversion remains outside
    # the timed GEMM region, as it would for prepacked model weights.
    a = torch.ops.mxfp6.pack_fp6(a_codes.cuda())
    b = torch.ops.mxfp6.pack_fp6(b_codes.cuda())
    sfa_logical = torch.full(
        (m, k // 32),
        0x7F,
        dtype=torch.uint8,
        device="cuda",
    )
    sfb_logical = torch.full(
        (n, k // 32),
        0x7F,
        dtype=torch.uint8,
        device="cuda",
    )
    sfa = torch.ops.mxfp6.pack_scales(sfa_logical, m, k)
    sfb = torch.ops.mxfp6.pack_scales(sfb_logical, n, k)
    return a, b, sfa, sfb, a_codes, b_codes


def check_correctness(
    tensors: tuple[torch.Tensor, ...], m: int, n: int, k: int
) -> tuple[float, float]:
    a, b, sfa, sfb, a_codes, b_codes = tensors
    actual = torch.ops.mxfp6.gemm(a, b, sfa, sfb, m, n, k)
    a_ref = decode_e3m2(a_codes).cuda()
    b_ref = decode_e3m2(b_codes).cuda()
    reference = (a_ref @ b_ref.t()).half()
    torch.cuda.synchronize()

    error = (actual.float() - reference.float()).abs()
    max_abs = error.max().item()
    denominator = actual.double().square().sum() + reference.double().square().sum()
    relative_diff = (1.0 - 2.0 * (actual.double() * reference.double()).sum() / denominator).item()
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.25)
    return max_abs, relative_diff


def benchmark(
    args: Iterable[torch.Tensor],
    m: int,
    n: int,
    k: int,
    warmup: int,
    iterations: int,
    flush_l2_mb: int,
) -> float:
    a, b, sfa, sfb = tuple(args)[:4]
    run = lambda: torch.ops.mxfp6.gemm(a, b, sfa, sfb, m, n, k)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    flush = None
    if flush_l2_mb > 0:
        flush = torch.empty(
            flush_l2_mb * 1_000_000 // 4, dtype=torch.int32, device="cuda"
        )

    # Sum CUDA kernel duration rather than Python/dispatcher submission gaps
    # between consecutive launches.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], acc_events=True
    ) as profiler:
        for _ in range(iterations):
            if flush is not None:
                flush.zero_()
            run()
    torch.cuda.synchronize()
    kernel_times = [
        event.self_device_time_total
        for event in profiler.events()
        if event.device_type.name == "CUDA" and "cutlass" in event.name.lower()
    ]
    if len(kernel_times) != iterations:
        raise RuntimeError(
            f"expected {iterations} CUTLASS kernels, profiler found {len(kernel_times)}"
        )
    return sum(kernel_times) / len(kernel_times)


def quantize_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K-aligned row groups for vLLM's block-scaled FP8 kernel."""
    m, k = x.shape
    padded_k = ((k + 127) // 128) * 128
    padded = torch.zeros((m, padded_k), dtype=x.dtype, device=x.device)
    padded[:, :k] = x
    groups = padded.view(m, -1, 128)
    scale = groups.abs().float().amax(dim=2).clamp(1.0e-4) / 448.0
    values = (groups / scale.unsqueeze(2)).to(torch.float8_e4m3fn)
    return values.view(m, padded_k)[:, :k].contiguous(), scale


def quantize_fp8_per_block(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 128x128 blocks for vLLM's block-scaled FP8 kernel."""
    m, k = x.shape
    padded_m = ((m + 127) // 128) * 128
    padded_k = ((k + 127) // 128) * 128
    padded = torch.zeros((padded_m, padded_k), dtype=x.dtype, device=x.device)
    padded[:m, :k] = x
    blocks = padded.view(-1, 128, padded_k // 128, 128)
    scale = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1.0e-4)
    scale = scale / 448.0
    values = (blocks / scale).to(torch.float8_e4m3fn)
    return values.view_as(padded)[:m, :k].contiguous(), scale.flatten(1, 3)


def benchmark_fp8(
    m: int,
    n: int,
    k: int,
    seed: int,
    iterations: int,
    warmup: int,
    flush_l2_mb: int,
) -> float:
    try:
        from vllm import _custom_ops as vllm_ops
    except ImportError as error:
        raise RuntimeError(
            "--compare-fp8 requires an installed vLLM build with SM120 CUTLASS ops"
        ) from error

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    a_bf16 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    a, scale_a = quantize_fp8_per_token(a_bf16)
    b, scale_b = quantize_fp8_per_block(b_bf16)
    run = lambda: vllm_ops.cutlass_scaled_mm(
        a,
        b.t(),
        scale_a=scale_a,
        scale_b=scale_b.t(),
        out_dtype=torch.bfloat16,
    )
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    flush = None
    if flush_l2_mb > 0:
        flush = torch.empty(
            flush_l2_mb * 1_000_000 // 4, dtype=torch.int32, device="cuda"
        )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], acc_events=True
    ) as profiler:
        for _ in range(iterations):
            if flush is not None:
                flush.zero_()
            run()
    torch.cuda.synchronize()
    kernel_times = [
        event.self_device_time_total
        for event in profiler.events()
        if event.device_type.name == "CUDA" and "cutlass" in event.name.lower()
    ]
    if len(kernel_times) != iterations:
        raise RuntimeError(
            f"expected {iterations} FP8 CUTLASS kernels, found {len(kernel_times)}"
        )
    return sum(kernel_times) / len(kernel_times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=DEFAULT_SHAPES,
        help="comma-separated MxNxK problems",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--flush-l2-mb",
        type=int,
        default=8000,
        help="L2 flush buffer size before every profiled GEMM; use 0 for warm-cache",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--check-all", action="store_true", help="run FP32 reference for every shape"
    )
    parser.add_argument(
        "--compare-fp8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also run the installed vLLM block-scaled FP8 CUTLASS kernel",
    )
    options = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(f"SM120 is required; found SM{major}{minor}")
    if not options.library.is_file():
        raise FileNotFoundError(
            f"{options.library} was not found; run: cmake --build build -j"
        )
    if options.warmup < 0 or options.iterations <= 0 or options.flush_l2_mb < 0:
        raise ValueError(
            "warmup/flush-l2-mb must be nonnegative and iterations must be positive"
        )

    torch.ops.load_library(str(options.library))
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"Device: {torch.cuda.get_device_name()} (SM120)")
    print("Op: torch.ops.mxfp6.gemm (E3M2 x E3M2, UE8M0/32 -> FP16)")
    speedups: list[tuple[int, float]] = []

    for index, (m, n, k) in enumerate(options.shapes):
        tensors = make_inputs(m, n, k, options.seed + index)
        if index == 0 or options.check_all:
            max_abs, diff = check_correctness(tensors, m, n, k)
            correctness = f"max_abs={max_abs:.4g}, diff={diff:.3g}"
        else:
            correctness = "check=skipped"

        time_us = benchmark(
            tensors,
            m,
            n,
            k,
            options.warmup,
            options.iterations,
            options.flush_l2_mb,
        )
        tflops = 2.0 * m * n * k / time_us / 1.0e6
        input_bytes = (m * k + n * k) * 3 // 4
        scale_bytes = (((m + 127) // 128) * 128 + n) * k // 32
        output_bytes = m * n * 2
        bandwidth = (input_bytes + scale_bytes + output_bytes) / time_us / 1.0e3
        comparison = ""
        if options.compare_fp8:
            fp8_time_us = benchmark_fp8(
                m,
                n,
                k,
                options.seed + index,
                options.iterations,
                options.warmup,
                options.flush_l2_mb,
            )
            comparison = (
                f" | FP8={fp8_time_us:8.3f} us | "
                f"MXFP6 speedup={fp8_time_us / time_us:5.3f}x"
            )
            speedups.append((m, fp8_time_us / time_us))
        print(
            f" > Perf (m={m:6}, n={n:6}, k={k:6}, layout=NT, FP16, "
            f"mxfp6): {time_us:8.3f} us | {tflops:7.2f} TFLOPS | "
            f"{bandwidth:7.2f} GB/s | {correctness}{comparison}"
        )

    if speedups:
        print("Speedup summary (geometric mean, FP8 time / MXFP6 time):")
        for batch in sorted({m for m, _ in speedups}):
            values = [value for m, value in speedups if m == batch]
            geometric_mean = math.exp(sum(math.log(value) for value in values) / len(values))
            print(f"  bs{batch:<4}: {geometric_mean:.3f}x over {len(values)} shapes")
        overall = math.exp(
            sum(math.log(value) for _, value in speedups) / len(speedups)
        )
        target = "PASS" if overall >= 1.2 else "MISS"
        print(
            f"  overall: {overall:.3f}x over {len(speedups)} shapes "
            f"(target >=1.200x: {target})"
        )


if __name__ == "__main__":
    main()
