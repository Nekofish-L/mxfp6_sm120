#!/usr/bin/env python3
"""Compare initialized and persistent Stream-K workspace launch paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = (
    (1, 8192, 5120),
    (16, 8192, 5120),
    (128, 8192, 5120),
    (512, 8192, 5120),
)


def parse_shapes(value: str) -> tuple[tuple[int, int, int], ...]:
    try:
        shapes = tuple(
            tuple(int(part) for part in item.lower().split("x"))
            for item in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shapes must use comma-separated MxNxK syntax"
        ) from error
    if not shapes or any(len(shape) != 3 or min(shape) <= 0 for shape in shapes):
        raise argparse.ArgumentTypeError("every shape must contain three positive values")
    return shapes  # type: ignore[return-value]


def elapsed_us(run, iterations: int, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def profile_cuda(run, iterations: int) -> tuple[float, int]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        acc_events=True,
    ) as profiler:
        for _ in range(iterations):
            run()
    torch.cuda.synchronize()
    events = [
        event
        for event in profiler.events()
        if event.device_type.name == "CUDA"
    ]
    gemms = [
        event.self_device_time_total
        for event in events
        if "cutlass" in event.name.lower()
    ]
    if len(gemms) < iterations:
        raise RuntimeError(
            f"expected {iterations} CUTLASS kernels, found {len(gemms)}"
        )
    memsets = sum("memset" in event.name.lower() for event in events)
    return statistics.median(gemms[-iterations:]), memsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=parse_shapes, default=DEFAULT_SHAPES)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--stabilize-iterations",
        type=int,
        default=5_000,
        help="launch the largest problem repeatedly before timing to ramp clocks",
    )
    parser.add_argument("--output-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/mxfp6_torch.so"
    )
    parser.add_argument("--json", type=Path)
    options = parser.parse_args()
    if (
        options.iterations <= 0
        or options.warmup < 0
        or options.repeats <= 0
        or options.stabilize_iterations < 0
    ):
        raise ValueError(
            "iterations/repeats must be positive; warmup/stabilize nonnegative"
        )

    os.environ["MXFP6_LIBRARY_PATH"] = str(options.library.resolve())
    sys.path.insert(0, str(ROOT / "python"))
    import mxfp6

    mxfp6.load_library()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 is required")
    out_dtype = (
        torch.float16 if options.output_dtype == "fp16" else torch.bfloat16
    )

    problems = []
    for index, (m, n, k) in enumerate(options.shapes):
        generator = torch.Generator(device="cuda").manual_seed(options.seed + index)
        source = torch.randn(
            (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = mxfp6.quantize_mxfp8(source)
        weight_source = torch.randn(
            (n, k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        weight = mxfp6.quantize_mxfp6(weight_source)
        config_id = 3 if m <= 32 else 16
        torch.ops.mxfp6.set_w6a8_config(
            activation.values, m, n, k, config_id, 1, 0, out_dtype
        )

        def run(
            activation=activation,
            weight=weight,
            m=m,
            n=n,
            k=k,
        ):
            return torch.ops.mxfp6.gemm_w6a8(
                activation.values,
                weight.values,
                activation.scales,
                weight.scales,
                m,
                n,
                k,
                1.0,
                out_dtype,
            )

        problems.append(((m, n, k), config_id, run))

    if options.stabilize_iterations:
        stabilize = max(problems, key=lambda problem: problem[0][0])[2]
        for _ in range(options.stabilize_iterations):
            stabilize()
        torch.cuda.synchronize()

    legacy = {}
    for shape, config_id, run in problems:
        for _ in range(options.warmup):
            reference = run()
        torch.cuda.synchronize()
        total_us = elapsed_us(run, options.iterations, options.repeats)
        kernel_us, memsets = profile_cuda(run, options.iterations)
        legacy[shape] = {
            "config_id": config_id,
            "reference": reference,
            "total_us": total_us,
            "kernel_us": kernel_us,
            "memsets": memsets,
        }

    mxfp6.begin_workspace_planning()
    for _, _, run in problems:
        run()
    torch.cuda.synchronize()
    planning = mxfp6.finalize_workspace_planning()

    results = []
    for shape, config_id, run in problems:
        for _ in range(options.warmup):
            output = run()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, legacy[shape]["reference"], rtol=0, atol=0
        )
        total_us = elapsed_us(run, options.iterations, options.repeats)
        kernel_us, memsets = profile_cuda(run, options.iterations)
        legacy_total = float(legacy[shape]["total_us"])
        result = {
            "shape": list(shape),
            "config_id": config_id,
            "legacy_total_us": legacy_total,
            "persistent_total_us": total_us,
            "speedup": legacy_total / total_us,
            "legacy_kernel_us": float(legacy[shape]["kernel_us"]),
            "persistent_kernel_us": kernel_us,
            "legacy_memset_events": int(legacy[shape]["memsets"]),
            "persistent_memset_events": memsets,
        }
        results.append(result)
        print(
            f"{shape[0]:4d}x{shape[1]:5d}x{shape[2]:5d} cfg={config_id:2d} "
            f"legacy={legacy_total:8.3f}us persistent={total_us:8.3f}us "
            f"speedup={legacy_total / total_us:6.3f}x "
            f"kernel={result['persistent_kernel_us']:8.3f}us "
            f"memset={result['legacy_memset_events']}->{memsets}"
        )

    if not mxfp6.workspace_barriers_zero():
        raise RuntimeError("persistent Stream-K barriers did not reset to zero")
    final_workspace = mxfp6.workspace_stats()
    payload = {
        "device": torch.cuda.get_device_name(),
        "output_dtype": options.output_dtype,
        "iterations": options.iterations,
        "repeats": options.repeats,
        "workspace": final_workspace,
        "results": results,
    }
    if options.json is not None:
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"workspace={final_workspace}")


if __name__ == "__main__":
    main()
