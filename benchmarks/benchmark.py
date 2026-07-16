#!/usr/bin/env python3
"""Correctness and performance benchmark for torch.ops.mxfp6.gemm."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Callable, Iterable

import torch

from model_shapes import DEFAULT_SHAPES, QWEN35_27B_TP2_NK


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "build" / "mxfp6_torch.so"
SHAPE_NAMES = {(n, k): name for name, n, k in QWEN35_27B_TP2_NK}


@dataclass(frozen=True)
class Timing:
    """GPU latency for the complete operator and its compute kernel."""

    pipeline_us: float
    kernel_us: float
    conversion_us: float = 0.0


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
    if m <= 0 or n <= 0 or n % 8 or k <= 0 or k % 128:
        raise ValueError(
            "M must be positive, N a multiple of 8, and K a multiple of 128"
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
    return (
        a,
        b,
        sfa,
        sfb,
        a_codes,
        b_codes,
        sfa_logical,
        sfb_logical,
    )


def check_correctness(
    tensors: tuple[torch.Tensor, ...], m: int, n: int, k: int
) -> tuple[float, float]:
    a, b, sfa, sfb, a_codes, b_codes, *_ = tensors
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


def benchmark_operation(
    run: Callable[[], torch.Tensor],
    kernel_filter: Callable[[str], bool],
    warmup: int,
    iterations: int,
    flush_l2_mb: int,
    conversion_filter: Callable[[str], bool] | None = None,
) -> Timing:
    """Measure device-side pipeline latency and isolate its main kernel."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    flush = None
    if flush_l2_mb > 0:
        flush = torch.empty(
            flush_l2_mb * 1_000_000 // 4, dtype=torch.int32, device="cuda"
        )

    starts: list[torch.cuda.Event] = []
    ends: list[torch.cuda.Event] = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], acc_events=True
    ) as profiler:
        if flush is None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            starts.append(start)
            ends.append(end)
            start.record()
            for _ in range(iterations):
                run()
            end.record()
        else:
            for _ in range(iterations):
                flush.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                starts.append(start)
                ends.append(end)
                start.record()
                run()
                end.record()
    ends[-1].synchronize()

    if flush is None:
        pipeline_us = starts[0].elapsed_time(ends[0]) * 1.0e3 / iterations
    else:
        pipeline_us = (
            sum(start.elapsed_time(end) for start, end in zip(starts, ends))
            * 1.0e3
            / iterations
        )

    kernel_times = [
        event.self_device_time_total
        for event in profiler.events()
        if event.device_type.name == "CUDA" and kernel_filter(event.name.lower())
    ]
    if len(kernel_times) != iterations:
        raise RuntimeError(
            f"expected {iterations} main kernels, profiler found {len(kernel_times)}"
        )
    conversion_us = 0.0
    if conversion_filter is not None:
        conversion_us = sum(
            event.self_device_time_total
            for event in profiler.events()
            if event.device_type.name == "CUDA"
            and conversion_filter(event.name.lower())
        ) / iterations
    return Timing(
        pipeline_us=pipeline_us,
        kernel_us=sum(kernel_times) / len(kernel_times),
        conversion_us=conversion_us,
    )


def benchmark(
    args: Iterable[torch.Tensor],
    m: int,
    n: int,
    k: int,
    warmup: int,
    iterations: int,
    flush_l2_mb: int,
) -> Timing:
    a, b, sfa, sfb = tuple(args)[:4]
    run = lambda: torch.ops.mxfp6.gemm(a, b, sfa, sfb, m, n, k)
    return benchmark_operation(
        run,
        lambda name: "cutlass" in name,
        warmup,
        iterations,
        flush_l2_mb,
        lambda name: "expand_fp6_to_fp8_vector_kernel" in name,
    )


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
) -> Timing:
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
    return benchmark_operation(
        run,
        lambda name: "cutlass" in name,
        warmup,
        iterations,
        flush_l2_mb,
    )


def benchmark_torch_scaled_mm(
    tensors: tuple[torch.Tensor, ...],
    m: int,
    n: int,
    k: int,
    iterations: int,
    warmup: int,
    flush_l2_mb: int,
    check: bool,
) -> Timing:
    """Benchmark PyTorch's dense FP8 GEMM on the same numerical inputs.

    The benchmark inputs use UE8M0 scales equal to one. E3M2 values can
    therefore be losslessly expanded to E4M3 and passed to ``_scaled_mm``
    with scalar unit scales. Expansion is preparation and stays outside the
    timed region for this pre-quantized FP8 baseline.
    """
    a, b, sfa, sfb = tensors[:4]
    a_fp8 = torch.ops.mxfp6.expand_fp6_to_fp8(a, m, k).view(
        torch.float8_e4m3fn
    )
    b_fp8 = torch.ops.mxfp6.expand_fp6_to_fp8(b, n, k).view(
        torch.float8_e4m3fn
    )
    scale_a = torch.ones((), dtype=torch.float32, device="cuda")
    scale_b = torch.ones((), dtype=torch.float32, device="cuda")
    run = lambda: torch._scaled_mm(
        a_fp8,
        b_fp8.t(),
        scale_a,
        scale_b,
        out_dtype=torch.float16,
        use_fast_accum=False,
    )
    if check:
        expected = torch.ops.mxfp6.gemm(a, b, sfa, sfb, m, n, k)
        torch.testing.assert_close(run(), expected, rtol=2e-3, atol=0.25)
        torch.cuda.synchronize()
    return benchmark_operation(
        run,
        lambda name: any(
            marker in name
            for marker in ("nvjet_sm120", "cutlass", "cublas", "scaled_mm")
        ),
        warmup,
        iterations,
        flush_l2_mb,
    )


def benchmark_humming(
    run,
    iterations: int,
    warmup: int,
    flush_l2_mb: int,
) -> Timing:
    """Measure activation expansion plus the Humming W6A8 GEMM."""
    return benchmark_operation(
        run,
        lambda name: "humming<" in name,
        warmup,
        iterations,
        flush_l2_mb,
        lambda name: "expand_fp6_to_fp8_vector_kernel" in name,
    )


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
    parser.add_argument(
        "--compare-torch-scaled-mm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also run torch._scaled_mm with losslessly expanded FP8 inputs",
    )
    parser.add_argument(
        "--compare-humming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also run the bundled Humming W6A8 bridge for eligible large-M shapes",
    )
    parser.add_argument(
        "--humming-min-m",
        type=int,
        default=512,
        help="minimum M for the Humming comparison (default: 512)",
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

    os.environ["MXFP6_LIBRARY_PATH"] = str(options.library.resolve())
    sys.path.insert(0, str(ROOT / "python"))
    import mxfp6

    mxfp6.load_library()
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"Device: {torch.cuda.get_device_name()} (SM120)")
    print("Op: torch.ops.mxfp6.gemm (packed E3M2 storage, UE8M0/32 -> FP16)")
    print("Latency: GPU pipeline (conversion + GEMM) and isolated GEMM kernel")
    vllm_fp8_speedups: list[tuple[int, float]] = []
    torch_fp8_speedups: list[tuple[int, float]] = []
    humming_speedups: list[tuple[int, float]] = []
    humming_checked = False

    for index, (m, n, k) in enumerate(options.shapes):
        layer = SHAPE_NAMES.get((n, k), "custom")
        tensors = make_inputs(m, n, k, options.seed + index)
        if index == 0 or options.check_all:
            max_abs, diff = check_correctness(tensors, m, n, k)
            correctness = f"max_abs={max_abs:.4g}, diff={diff:.3g}"
        else:
            correctness = "check=skipped"

        timing = benchmark(
            tensors,
            m,
            n,
            k,
            options.warmup,
            options.iterations,
            options.flush_l2_mb,
        )
        pipeline_tflops = 2.0 * m * n * k / timing.pipeline_us / 1.0e6
        kernel_tflops = 2.0 * m * n * k / timing.kernel_us / 1.0e6
        input_bytes = (m * k + n * k) * 3 // 4
        scale_bytes = (((m + 127) // 128) * 128 + n) * k // 32
        output_bytes = m * n * 2
        bandwidth = (
            input_bytes + scale_bytes + output_bytes
        ) / timing.pipeline_us / 1.0e3
        comparison = ""
        if options.compare_fp8:
            fp8_timing = benchmark_fp8(
                m,
                n,
                k,
                options.seed + index,
                options.iterations,
                options.warmup,
                options.flush_l2_mb,
            )
            comparison = (
                f" | FP8 e2e={fp8_timing.pipeline_us:8.3f} us "
                f"(gemm={fp8_timing.kernel_us:8.3f}) | "
                f"MXFP6 speedup={fp8_timing.pipeline_us / timing.pipeline_us:5.3f}x"
            )
            vllm_fp8_speedups.append(
                (m, fp8_timing.pipeline_us / timing.pipeline_us)
            )
        if options.compare_torch_scaled_mm:
            torch_fp8_timing = benchmark_torch_scaled_mm(
                tensors,
                m,
                n,
                k,
                options.iterations,
                options.warmup,
                options.flush_l2_mb,
                index == 0 or options.check_all,
            )
            comparison += (
                f" | torch._scaled_mm e2e={torch_fp8_timing.pipeline_us:8.3f} us "
                f"(gemm={torch_fp8_timing.kernel_us:8.3f}) | "
                f"MXFP6 speedup="
                f"{torch_fp8_timing.pipeline_us / timing.pipeline_us:5.3f}x"
            )
            torch_fp8_speedups.append(
                (m, torch_fp8_timing.pipeline_us / timing.pipeline_us)
            )
        if options.compare_humming:
            if m < options.humming_min_m or n % 256:
                comparison += " | Humming=skipped"
            else:
                a, b, sfa, sfb, _, _, sfa_logical, sfb_logical = tensors
                packed_a = mxfp6.PackedMXFP6Tensor(
                    a, sfa, m, k, sfa_logical
                )
                packed_b = mxfp6.PackedMXFP6Tensor(
                    b, sfb, n, k, sfb_logical
                )
                humming_b = mxfp6.prepare_humming_weight(packed_b)
                run_humming = lambda: mxfp6.gemm(packed_a, humming_b)
                if not humming_checked or options.check_all:
                    torch.testing.assert_close(
                        run_humming(),
                        torch.ops.mxfp6.gemm(a, b, sfa, sfb, m, n, k),
                        rtol=2e-3,
                        atol=0.5,
                    )
                    humming_checked = True
                humming_timing = benchmark_humming(
                    run_humming,
                    options.iterations,
                    options.warmup,
                    options.flush_l2_mb,
                )
                comparison += (
                    f" | Humming e2e={humming_timing.pipeline_us:8.3f} us "
                    f"(gemm={humming_timing.kernel_us:8.3f}, "
                    f"convert={humming_timing.conversion_us:6.3f}) | "
                    f"MXFP6 speedup={humming_timing.pipeline_us / timing.pipeline_us:5.3f}x"
                )
                humming_speedups.append(
                    (m, humming_timing.pipeline_us / timing.pipeline_us)
                )
        conversion = (
            f", convert={timing.conversion_us:6.3f}"
            if timing.conversion_us > 0.0
            else ""
        )
        print(
            f" > Perf (layer={layer}, m={m:6}, n={n:6}, k={k:6}, "
            f"layout=NT, FP16, "
            f"mxfp6): e2e={timing.pipeline_us:8.3f} us "
            f"(gemm={timing.kernel_us:8.3f}{conversion}) | "
            f"{pipeline_tflops:7.2f} e2e TFLOPS | "
            f"{kernel_tflops:7.2f} kernel TFLOPS | "
            f"{bandwidth:7.2f} GB/s | {correctness}{comparison}"
        )

    if vllm_fp8_speedups:
        print("Speedup over vLLM block-scaled FP8 (FP8 time / MXFP6 time):")
        for batch in sorted({m for m, _ in vllm_fp8_speedups}):
            values = [value for m, value in vllm_fp8_speedups if m == batch]
            geometric_mean = math.exp(sum(math.log(value) for value in values) / len(values))
            print(f"  bs{batch:<4}: {geometric_mean:.3f}x over {len(values)} shapes")
        overall = math.exp(
            sum(math.log(value) for _, value in vllm_fp8_speedups)
            / len(vllm_fp8_speedups)
        )
        target = "PASS" if overall >= 1.2 else "MISS"
        print(
            f"  overall: {overall:.3f}x over {len(vllm_fp8_speedups)} shapes "
            f"(target >=1.200x: {target})"
        )

    if torch_fp8_speedups:
        print("Speedup over torch._scaled_mm (FP8 time / MXFP6 time):")
        for batch in sorted({m for m, _ in torch_fp8_speedups}):
            values = [value for m, value in torch_fp8_speedups if m == batch]
            geometric_mean = math.exp(
                sum(math.log(value) for value in values) / len(values)
            )
            print(
                f"  bs{batch:<4}: {geometric_mean:.3f}x "
                f"over {len(values)} shapes"
            )
        overall = math.exp(
            sum(math.log(value) for _, value in torch_fp8_speedups)
            / len(torch_fp8_speedups)
        )
        print(
            f"  overall: {overall:.3f}x over "
            f"{len(torch_fp8_speedups)} shapes"
        )

    if humming_speedups:
        overall = math.exp(
            sum(math.log(value) for _, value in humming_speedups)
            / len(humming_speedups)
        )
        print(
            "MXFP6 pipeline speedup over Humming (Humming time / MXFP6 time): "
            f"{overall:.3f}x over {len(humming_speedups)} shapes"
        )


if __name__ == "__main__":
    main()
