#!/usr/bin/env python3
"""Compare isolated native 16->6 and 16->8 activation quantization kernels."""

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
DEFAULT_LIBRARY = ROOT / "build" / "mxfp6_torch.so"


def parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--batches must be comma-separated ints") from error
    if not batches or any(batch <= 0 for batch in batches):
        raise argparse.ArgumentTypeError("all batches must be positive")
    return batches


def isolated_pair_us(run6, run8, warmup: int, iterations: int, seed: int):
    for _ in range(warmup):
        run6()
        run8()
    torch.cuda.synchronize()
    order: list[str] = []
    randomizer = random.Random(seed)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], acc_events=True
    ) as profiler:
        for _ in range(iterations):
            pair = [("a6", run6), ("a8", run8)]
            randomizer.shuffle(pair)
            for name, run in pair:
                run()
                order.append(name)
    torch.cuda.synchronize()
    times = [
        event.self_device_time_total
        for event in profiler.events()
        if event.device_type.name == "CUDA"
        and "quantize_mx_kernel" in event.name.lower()
    ]
    if len(times) != len(order):
        raise RuntimeError(
            f"expected {len(order)} quantization kernels, found {len(times)}"
        )
    samples: dict[str, list[float]] = defaultdict(list)
    for name, duration in zip(order, times):
        samples[name].append(duration)
    return statistics.median(samples["a6"]), statistics.median(samples["a8"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=parse_batches,
        default=(1, 16, 512, 1024, 4096, 8192),
    )
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    options = parser.parse_args()
    if options.k <= 0 or options.k % 128:
        raise ValueError("K must be a positive multiple of 128")
    if options.warmup < 0 or options.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")

    os.environ["MXFP6_LIBRARY_PATH"] = str(options.library.resolve())
    sys.path.insert(0, str(ROOT / "python"))
    import mxfp6

    mxfp6.load_library()
    dtype = torch.float16 if options.dtype == "fp16" else torch.bfloat16
    print(f"Device: {torch.cuda.get_device_name()} | input={options.dtype} K={options.k}")
    print("Times include only quantize_mx_kernel, not allocations or host gaps")
    for m in options.batches:
        source = torch.randn((m, options.k), device="cuda", dtype=dtype)
        a6_us, a8_us = isolated_pair_us(
            lambda: torch.ops.mxfp6.quantize_mxfp6(source),
            lambda: torch.ops.mxfp6.quantize_mxfp8(source),
            options.warmup,
            options.iterations,
            options.seed + m,
        )
        print(
            f"M={m:5d}: 16->6={a6_us:8.3f} us  "
            f"16->8={a8_us:8.3f} us  A8/A6={a8_us / a6_us:6.3f}x"
        )


if __name__ == "__main__":
    main()
