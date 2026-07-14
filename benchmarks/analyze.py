#!/usr/bin/env python3
"""Summarize profiler results and derive a compact kernel portfolio."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


RESULT_PATTERN = re.compile(
    r"m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)_"
    r"(?P<orientation>normal|swapped)\.block_scaled_gemm\.csv$"
)
OPERATION_PREFIX = (
    "cutlass3x_sm120_bstensorop_gemm_ue8m0xe3m2_ue8m0xe3m2_"
    "f32_void_f16_"
)
MISSING_SLOWDOWN = 100.0


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class Result:
    operation: str
    runtime_us: float
    split_k_slices: int
    raster_order: str
    swizzle_size: int


def load_results(
    directories: list[Path],
) -> dict[tuple[Shape, str], list[Result]]:
    jobs: dict[tuple[Shape, str], list[Result]] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.block_scaled_gemm.csv")):
            match = RESULT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            shape = Shape(
                int(match.group("m")),
                int(match.group("n")),
                int(match.group("k")),
            )
            rows = []
            with path.open(newline="") as file:
                for row in csv.DictReader(file):
                    if row["Status"] != "success":
                        continue
                    rows.append(
                        Result(
                            operation=row["Operation"],
                            runtime_us=float(row["Runtime"]) * 1000.0,
                            split_k_slices=int(row["split_k_slices"]),
                            raster_order=row["raster_order"],
                            swizzle_size=int(row["swizzle_size"]),
                        )
                    )
            if rows:
                # Later directories override repeated jobs, which allows a
                # high-precision boundary run to replace a coarse result.
                jobs[(shape, match.group("orientation"))] = rows
    if not jobs:
        raise RuntimeError("no profiler CSV files were found")
    return jobs


def best_per_operation(rows: list[Result]) -> dict[str, Result]:
    best: dict[str, Result] = {}
    for row in rows:
        if row.operation not in best or row.runtime_us < best[row.operation].runtime_us:
            best[row.operation] = row
    return best


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def print_orientation_summary(
    jobs: dict[tuple[Shape, str], list[Result]],
) -> None:
    ratios: dict[int, list[float]] = defaultdict(list)
    shapes = sorted({shape for shape, _ in jobs}, key=lambda s: (s.m, s.n, s.k))
    for shape in shapes:
        normal = jobs.get((shape, "normal"))
        swapped = jobs.get((shape, "swapped"))
        if normal is None or swapped is None:
            continue
        normal_us = min(row.runtime_us for row in normal)
        swapped_us = min(row.runtime_us for row in swapped)
        ratios[shape.m].append(normal_us / swapped_us)

    print("Orientation (ratio > 1 favors swapped):")
    for m, values in sorted(ratios.items()):
        geometric_mean = math.exp(statistics.fmean(math.log(value) for value in values))
        print(
            f"  M={m:<5} swapped wins {sum(value > 1 for value in values):>3}/"
            f"{len(values):<3}  geo(normal/swapped)={geometric_mean:.4f}  "
            f"range=[{min(values):.3f}, {max(values):.3f}]"
        )


def choose_orientation(shape: Shape, swap_max_m: int) -> str:
    return "swapped" if shape.m <= swap_max_m else "normal"


def rule_operation(shape: Shape, orientation: str) -> str:
    """Return the profiler-derived configuration for a shape region."""
    output_elements = shape.m * shape.n
    if orientation == "swapped":
        if shape.k >= 8192:
            if shape.m <= 8:
                config = "128x8x128_1x1x1_0@stream_k_cooperative"
            elif shape.m <= 16:
                config = "128x16x128_1x1x1_0@stream_k_cooperative"
            elif output_elements <= 65536:
                config = (
                    "64x16x256_1x1x1_3@pingpong"
                    if shape.m <= 64 and shape.k <= 8192
                    else "128x16x128_1x1x1_0@stream_k_cooperative"
                )
            elif output_elements <= 524288:
                config = "128x32x128_1x1x1_3@stream_k_cooperative"
            elif shape.m > 64 and shape.n >= 8192:
                config = "128x32x128_1x1x1_3@stream_k_cooperative"
            else:
                config = "128x64x128_1x1x1_0@stream_k_cooperative"
        elif shape.k >= 2048:
            if shape.n <= 512:
                config = "64x16x256_1x1x1_3@pingpong"
            elif shape.m <= 8 and shape.n >= 8192:
                config = "128x8x128_1x1x1_4@cooperative"
            elif shape.m <= 16:
                config = (
                    "64x16x256_1x1x1_3@pingpong"
                    if shape.n <= 8192 and shape.k <= 2048
                    else (
                        "128x16x128_1x1x1_0@stream_k_cooperative"
                        if shape.n >= 8192
                        else "64x16x128_1x1x1_3@pingpong"
                    )
                )
            elif shape.n >= 8192:
                if shape.m <= 32:
                    config = (
                        "64x32x128_1x1x1_3@pingpong"
                        if shape.n <= 8192 and shape.k <= 2048
                        else "128x32x128_1x1x1_3@stream_k_cooperative"
                    )
                elif shape.m > 64 or output_elements <= 524288:
                    config = "128x32x128_1x1x1_3@stream_k_cooperative"
                else:
                    config = "128x64x128_1x1x1_0@stream_k_cooperative"
            elif shape.m > 64 and shape.k >= 4096:
                config = "128x32x128_1x1x1_3@stream_k_cooperative"
            else:
                config = "64x32x128_1x1x1_3@pingpong"
        else:
            if shape.m <= 8 and shape.n >= 8192:
                config = "128x8x128_1x1x1_2@cooperative"
            elif shape.m <= 16 and shape.n >= 8192:
                config = "64x16x128_1x1x1_3@pingpong"
            elif shape.m <= 64 and output_elements > 524288:
                config = "128x64x128_1x1x1_0@stream_k_cooperative"
            else:
                config = "64x32x128_1x1x1_3@pingpong"
    else:
        if shape.k <= 512:
            if output_elements <= 262144:
                config = "64x32x128_1x1x1_3@pingpong"
            elif output_elements <= 16777216:
                config = "64x64x128_1x1x1_2@pingpong"
            else:
                config = "128x128x128_1x1x1_0@pingpong"
        elif shape.k < 8192:
            if output_elements <= 131072 and shape.n <= 1024:
                config = "64x16x256_1x1x1_0@pingpong"
            elif output_elements <= 262144:
                config = "64x32x128_1x1x1_3@pingpong"
            elif output_elements <= 16777216:
                config = "64x64x128_1x1x1_4@pingpong"
            else:
                config = "128x128x128_1x1x1_0@pingpong"
        elif shape.n <= 128:
            if shape.m <= 512:
                config = (
                    "128x16x128_1x1x1_3@stream_k_cooperative"
                    if shape.k >= 16384
                    else "64x16x256_1x1x1_0@pingpong"
                )
            elif shape.m <= 2048:
                config = "128x32x128_1x1x1_0@stream_k_cooperative"
            elif shape.m <= 4096:
                config = "128x64x128_1x1x1_0@stream_k_cooperative"
            else:
                config = "128x128x128_1x1x1_0@stream_k_cooperative"
        elif output_elements <= 65536:
            config = "64x16x256_1x1x1_0@pingpong"
        elif output_elements <= 262144:
            config = "128x32x128_1x1x1_0@stream_k_cooperative"
        elif output_elements <= 4194304:
            config = "128x64x128_1x1x1_0@stream_k_cooperative"
        else:
            config = "128x128x128_1x1x1_0@stream_k_cooperative"
    tile, schedule = config.split("@", 1)
    layout = "tnn" if orientation == "swapped" else "tnt"
    return f"{OPERATION_PREFIX}{tile}_{layout}_align128_{schedule}_q"


def print_rule_summary(
    jobs: dict[tuple[Shape, str], list[Result]], swap_max_m: int
) -> None:
    slowdowns = []
    records = []
    missing = []
    per_m: dict[int, list[float]] = defaultdict(list)
    for shape in sorted({shape for shape, _ in jobs}, key=lambda s: (s.m, s.n, s.k)):
        orientation = choose_orientation(shape, swap_max_m)
        rows = jobs.get((shape, orientation))
        if rows is None:
            continue
        operations = best_per_operation(rows)
        selected = rule_operation(shape, orientation)
        if selected not in operations:
            missing.append((shape, selected))
            continue
        oracle_us = min(row.runtime_us for row in operations.values())
        slowdown = operations[selected].runtime_us / oracle_us
        slowdowns.append(slowdown)
        records.append((slowdown, shape, orientation, selected))
        per_m[shape.m].append(slowdown)

    print("\nRule-based dispatcher versus per-shape profiler oracle:")
    if slowdowns:
        print(
            f"  coverage={len(slowdowns)}/{len(slowdowns) + len(missing)} "
            f"geo={math.exp(statistics.fmean(math.log(x) for x in slowdowns)):.3f}x "
            f"p90={percentile(slowdowns, 0.90):.3f}x "
            f"max={max(slowdowns):.3f}x"
        )
        for m, values in sorted(per_m.items()):
            print(
                f"  M={m:<5} geo={math.exp(statistics.fmean(math.log(x) for x in values)):.3f}x "
                f"max={max(values):.3f}x"
            )
        print("  worst regions:")
        for slowdown, shape, orientation, operation in sorted(
            records, reverse=True, key=lambda record: record[0]
        )[:5]:
            print(
                f"    {shape.m}x{shape.n}x{shape.k} {orientation}: "
                f"{slowdown:.3f}x "
                f"{operation.removeprefix(OPERATION_PREFIX)}"
            )
    for shape, operation in missing[:5]:
        print(f"  missing {shape}: {operation.removeprefix(OPERATION_PREFIX)}")


def derive_portfolio(
    jobs: dict[tuple[Shape, str], list[Result]],
    swap_max_m: int,
    count: int,
    show_assignments: bool,
) -> None:
    selected_jobs: dict[Shape, dict[str, Result]] = {}
    for shape in sorted({shape for shape, _ in jobs}, key=lambda s: (s.m, s.n, s.k)):
        orientation = choose_orientation(shape, swap_max_m)
        rows = jobs.get((shape, orientation))
        if rows is not None:
            selected_jobs[shape] = best_per_operation(rows)

    oracle = {
        shape: min(result.runtime_us for result in operations.values())
        for shape, operations in selected_jobs.items()
    }
    candidates = sorted(
        {operation for operations in selected_jobs.values() for operation in operations}
    )
    current = {shape: math.inf for shape in selected_jobs}
    portfolio: list[str] = []

    print(
        f"\nGreedy portfolio: swap when M <= {swap_max_m}; "
        f"{len(selected_jobs)} profiler jobs"
    )
    for index in range(count):
        best_candidate = None
        best_score = math.inf
        best_times = None
        for candidate in candidates:
            if candidate in portfolio:
                continue
            times = {
                shape: min(
                    current[shape],
                    operations[candidate].runtime_us
                    if candidate in operations
                    else math.inf,
                )
                for shape, operations in selected_jobs.items()
            }
            slowdowns = [
                min(times[shape] / oracle[shape], MISSING_SLOWDOWN)
                for shape in selected_jobs
            ]
            score = statistics.fmean(math.log(value) for value in slowdowns)
            if score < best_score:
                best_candidate = candidate
                best_score = score
                best_times = times

        if best_candidate is None or best_times is None:
            break
        portfolio.append(best_candidate)
        current = best_times
        covered = [shape for shape, value in current.items() if math.isfinite(value)]
        slowdowns = [current[shape] / oracle[shape] for shape in covered]
        short_name = best_candidate.removeprefix(OPERATION_PREFIX)
        print(
            f"  {index + 1:>2}. {short_name}\n"
            f"      coverage={len(covered)}/{len(current)} "
            f"geo={math.exp(statistics.fmean(math.log(x) for x in slowdowns)):.3f}x "
            f"p90={percentile(slowdowns, 0.90):.3f}x "
            f"max={max(slowdowns):.3f}x"
        )

    if show_assignments:
        print("\nBest assignment within the selected portfolio:")
        for shape, operations in selected_jobs.items():
            available = [
                operations[operation]
                for operation in portfolio
                if operation in operations
            ]
            if not available:
                continue
            winner = min(available, key=lambda result: result.runtime_us)
            orientation = choose_orientation(shape, swap_max_m)
            print(
                f"  {shape.m:>5} {shape.n:>5} {shape.k:>5} {orientation:<7} "
                f"{winner.operation.removeprefix(OPERATION_PREFIX)} "
                f"{winner.runtime_us / oracle[shape]:.3f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="directories containing CUTLASS profiler CSV files",
    )
    parser.add_argument(
        "--swap-max-m",
        type=int,
        default=96,
        help="use swapped kernels at or below this M threshold",
    )
    parser.add_argument("--portfolio-size", type=int, default=10)
    parser.add_argument("--show-assignments", action="store_true")
    options = parser.parse_args()
    if options.swap_max_m < 0 or options.portfolio_size < 1:
        raise ValueError("swap-max-m must be nonnegative and portfolio-size positive")

    jobs = load_results(options.results)
    print(f"Loaded {len(jobs)} orientation-specific profiler jobs")
    print_orientation_summary(jobs)
    print_rule_summary(jobs, options.swap_max_m)
    derive_portfolio(
        jobs,
        options.swap_max_m,
        options.portfolio_size,
        options.show_assignments,
    )


if __name__ == "__main__":
    main()
