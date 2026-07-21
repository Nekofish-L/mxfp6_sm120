#!/usr/bin/env python3
"""Interleaved W6A6/W6A8 GEMM comparison for small batches.

Activations are quantized before profiling.  The script intentionally measures
only CUTLASS GEMM kernels; use ``quantization.py`` for the independent 16->6
and 16->8 conversion cost.  Randomized interleaving reduces clock/thermal drift
between the two formats, while ``--flush-l2-mb`` makes cache state explicit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import random
import statistics
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = (
    (1, 8192, 5120),
    (1, 5120, 3072),
    (1, 7168, 5120),
    (1, 17408, 5120),
    (1, 5120, 8704),
    (16, 8192, 5120),
    (16, 5120, 3072),
    (16, 7168, 5120),
    (16, 17408, 5120),
    (16, 5120, 8704),
)


def parse_shapes(value: str) -> list[tuple[int, int, int]]:
    try:
        return [
            tuple(int(part) for part in item.lower().split("x"))
            for item in value.split(",")
        ]  # type: ignore[return-value]
    except ValueError as error:
        raise argparse.ArgumentTypeError("shapes must use MxNxK syntax") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=parse_shapes, default=DEFAULT_SHAPES)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--flush-l2-mb", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/mxfp6_torch.so"
    )
    options = parser.parse_args()
    if options.iterations <= 0 or options.warmup < 0 or options.flush_l2_mb < 0:
        raise ValueError("iterations must be positive; warmup/flush must be nonnegative")

    os.environ["MXFP6_LIBRARY_PATH"] = str(options.library.resolve())
    sys.path.insert(0, str(ROOT / "python"))
    import mxfp6

    mxfp6.load_library()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 is required")

    print(
        f"device={torch.cuda.get_device_name()} cache_flush="
        f"{options.flush_l2_mb}MB iterations={options.iterations}"
    )
    ratios: dict[int, list[float]] = defaultdict(list)
    for shape_index, (m, n, k) in enumerate(options.shapes):
        generator = torch.Generator(device="cuda").manual_seed(
            options.seed + shape_index
        )
        source = torch.randn(
            (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        a6 = mxfp6.quantize_mxfp6(source)
        a8 = mxfp6.quantize_mxfp8(source)
        b_codes = torch.randint(
            0, 32, (n, k), generator=generator, device="cuda", dtype=torch.uint8
        )
        b_scales = torch.full(
            (n, k // 32), 0x7F, device="cuda", dtype=torch.uint8
        )
        weight = mxfp6.pack_operand(b_codes, b_scales)
        flush = None
        if options.flush_l2_mb:
            flush = torch.empty(
                options.flush_l2_mb * 1_000_000 // 4,
                device="cuda",
                dtype=torch.int32,
            )

        def run(kind: str):
            if flush is not None:
                flush.zero_()
            activation = a6 if kind == "w6a6" else a8
            op = (
                torch.ops.mxfp6.gemm
                if kind == "w6a6"
                else torch.ops.mxfp6.gemm_w6a8
            )
            return op(
                activation.values,
                weight.values,
                activation.scales,
                weight.scales,
                m,
                n,
                k,
            )

        for _ in range(options.warmup):
            run("w6a6")
            run("w6a8")
        torch.cuda.synchronize()

        order: list[str] = []
        randomizer = random.Random(options.seed + shape_index)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], acc_events=True
        ) as profiler:
            for _ in range(options.iterations):
                pair = ["w6a6", "w6a8"]
                randomizer.shuffle(pair)
                for kind in pair:
                    run(kind)
                    order.append(kind)
        torch.cuda.synchronize()
        events = [
            event.self_device_time_total
            for event in profiler.events()
            if event.device_type.name == "CUDA" and "cutlass" in event.name.lower()
        ]
        if len(events) != len(order):
            raise RuntimeError(
                f"{m}x{n}x{k}: expected {len(order)} GEMMs, found {len(events)}"
            )
        samples: dict[str, list[float]] = defaultdict(list)
        for kind, duration in zip(order, events):
            samples[kind].append(duration)
        a6_us = statistics.median(samples["w6a6"])
        a8_us = statistics.median(samples["w6a8"])
        ratio = a8_us / a6_us
        ratios[m].append(ratio)
        print(
            f"{m:4d}x{n:5d}x{k:4d}  W6A6={a6_us:8.3f}us  "
            f"W6A8={a8_us:8.3f}us  W6A8/W6A6={ratio:6.3f}x"
        )

    for m, values in ratios.items():
        geomean = statistics.geometric_mean(values)
        print(f"M={m} W6A8/W6A6 geomean={geomean:.3f}x")


if __name__ == "__main__":
    main()
