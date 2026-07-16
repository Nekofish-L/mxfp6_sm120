#!/usr/bin/env python3
"""Exhaustive CUTLASS-profiler search for SM120 MXFP6 GEMM shapes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from model_shapes import DEFAULT_BATCH_SIZES, QWEN35_27B_TP2_NK


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILERS = {
    "w6a6": ROOT / "build_profiler/tools/profiler/cutlass_profiler",
    "w6a8": ROOT / "build_profiler_a8b6/tools/profiler/cutlass_profiler",
}
DEFAULT_OUTPUT = ROOT / "benchmarks/results/latest"
MAX_LLC_CAPACITY_KIB = ((1 << 31) - 1) >> 10
KERNEL_PREFIXES = {
    "w6a6": (
        "cutlass3x_sm120_bstensorop_gemm_ue8m0xe3m2_ue8m0xe3m2_"
        "f32_void_f16_"
    ),
    "w6a8": (
        "cutlass3x_sm120_bstensorop_gemm_ue8m0xe4m3_ue8m0xe3m2_"
        "f32_void_f16_"
    ),
}


@dataclass(frozen=True)
class Job:
    m: int
    n: int
    k: int
    orientation: str
    output_base: Path

    @property
    def profiler_shape(self) -> tuple[int, int, str]:
        if self.orientation == "normal":
            return self.m, self.n, "tnt"
        return self.n, self.m, "tnn"

    @property
    def csv_path(self) -> Path:
        return Path(f"{self.output_base}.block_scaled_gemm.csv")


def parse_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_shapes(value: str) -> tuple[tuple[int, int, int], ...]:
    shapes = []
    for item in value.split(","):
        try:
            m, n, k = (int(dimension) for dimension in item.lower().split("x"))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid shape {item!r}; expected MxNxK"
            ) from error
        if min(m, n, k) <= 0:
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        shapes.append((m, n, k))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return tuple(shapes)


def make_jobs(
    output_dir: Path,
    orientations: str,
    shapes: tuple[tuple[int, int, int], ...],
) -> list[Job]:
    jobs = []
    for m, n, k in shapes:
        if orientations == "both":
            selected = ("normal", "swapped")
        elif orientations == "auto":
            selected = ("swapped",) if m <= 32 else ("normal",)
        else:
            selected = (orientations,)
        for orientation in selected:
            jobs.append(
                Job(
                    m,
                    n,
                    k,
                    orientation,
                    output_dir / f"m{m}_n{n}_k{k}_{orientation}",
                )
            )
    return jobs


def run_job(
    job: Job,
    device: int,
    profiler: Path,
    warmup: int,
    iterations: int,
    workspace_count: int,
    llc_capacity_kib: int,
    kernel_suffix_glob: str,
    split_k_slices: str,
    kernel_glob: str | None,
    kernel_prefix: str,
) -> None:
    m, n, layout = job.profiler_shape
    selected_kernels = (
        kernel_glob.format(prefix=kernel_prefix, layout=layout)
        if kernel_glob is not None
        else f"{kernel_prefix}{kernel_suffix_glob}_{layout}_*"
    )
    command = [
        str(profiler),
        "--operation=block_scaled_gemm",
        f"--kernels={selected_kernels}",
        "--enable-best-kernel-for-fixed-shape=true",
        f"--m={m}",
        f"--n={n}",
        f"--k={job.k}",
        "--split_k_mode=serial",
        f"--split_k_slices={split_k_slices}",
        f"--workspace-count={workspace_count}",
        f"--warmup-iterations={warmup}",
        f"--profiling-iterations={iterations}",
        "--sleep-duration=0",
        "--verification-enabled=false",
        "--sort-results-flops-per-sec=true",
        "--verbose=false",
        f"--output={job.output_base}",
    ]
    if llc_capacity_kib > 0:
        command.append(f"--llc-capacity={llc_capacity_kib}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(
        f"profiled {job.m}x{job.n}x{job.k} "
        f"{job.orientation} on GPU {device}",
        flush=True,
    )


def read_top(job: Job, count: int) -> list[dict[str, str]]:
    with job.csv_path.open(newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["Status"] == "success"
        ]
    rows.sort(key=lambda row: float(row["Runtime"]))
    return rows[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=Path)
    parser.add_argument(
        "--mma",
        choices=("w6a6", "w6a8"),
        default="w6a6",
        help="profile native E3M2xE3M2 or mixed E4M3xE3M2 compute",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--devices",
        default="4,5,6,7",
        help="comma-separated physical GPU indices; jobs are serialized per GPU",
    )
    parser.add_argument(
        "--orientations",
        choices=("both", "auto", "normal", "swapped"),
        default="both",
        help="'auto' uses swapped for M <= 32 and normal otherwise",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        help="comma-separated explicit MxNxK shapes",
    )
    parser.add_argument("--m-values", type=parse_values)
    parser.add_argument("--n-values", type=parse_values)
    parser.add_argument("--k-values", type=parse_values)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--workspace-count",
        type=int,
        default=8,
        help="rotate independent inputs to make profiler ranking cold-cache",
    )
    parser.add_argument(
        "--llc-capacity-kib",
        type=int,
        default=0,
        help=(
            "profiler cache-rotation capacity in KiB; use with "
            "--workspace-count=0 for strict cold-cache ranking"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--split-k-slices",
        default="1,2,4",
        help="comma-separated Stream-K/Split-K decompositions to rank",
    )
    parser.add_argument(
        "--kernel-suffix-glob",
        default="*",
        help="glob between the fixed operation prefix and layout (for example '256x*x128_*')",
    )
    parser.add_argument(
        "--kernel-glob",
        help="full operation glob with optional {prefix}/{layout} placeholders",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip profiler jobs whose output CSV already exists",
    )
    parser.add_argument("--parse-only", action="store_true")
    options = parser.parse_args()
    if options.profiler is None:
        options.profiler = DEFAULT_PROFILERS[options.mma]

    devices = [int(item) for item in options.devices.split(",")]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    if min(options.warmup, options.iterations) < 1:
        raise ValueError("warmup and iterations must be positive")
    if options.workspace_count < 0 or options.llc_capacity_kib < 0:
        raise ValueError("workspace-count and llc-capacity-kib must be nonnegative")
    if options.llc_capacity_kib > MAX_LLC_CAPACITY_KIB:
        raise ValueError(
            "llc-capacity-kib exceeds the CUTLASS profiler's signed "
            f"32-bit byte limit ({MAX_LLC_CAPACITY_KIB} KiB)"
        )
    if options.top_k < 1:
        raise ValueError("top-k must be positive")
    if options.mma == "w6a8" and options.orientations in ("both", "swapped"):
        raise ValueError(
            "mixed W6A8 profiling requires --orientations=normal (or auto "
            "for large-M-only shape lists)"
        )
    if not options.parse_only and not options.profiler.is_file():
        raise FileNotFoundError(options.profiler)

    grid_values = (options.m_values, options.n_values, options.k_values)
    if options.shapes is not None and any(value is not None for value in grid_values):
        raise ValueError("--shapes cannot be combined with --m/n/k-values")
    if any(value is not None for value in grid_values):
        if any(value is None for value in grid_values):
            raise ValueError("--m-values, --n-values, and --k-values are all required")
        shapes = tuple(product(*grid_values))
    elif options.shapes is not None:
        shapes = options.shapes
    else:
        shapes = tuple(
            (m, n, k)
            for m in DEFAULT_BATCH_SIZES
            for _, n, k in QWEN35_27B_TP2_NK
        )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(options.output_dir, options.orientations, shapes)
    if not options.parse_only:
        buckets = [[] for _ in devices]
        for index, job in enumerate(jobs):
            buckets[index % len(devices)].append(job)

        def run_bucket(device: int, bucket: list[Job]) -> None:
            for job in bucket:
                if options.resume and job.csv_path.is_file():
                    print(f"resume: {job.csv_path}", flush=True)
                    continue
                run_job(
                    job,
                    device,
                    options.profiler,
                    options.warmup,
                    options.iterations,
                    options.workspace_count,
                    options.llc_capacity_kib,
                    options.kernel_suffix_glob,
                    options.split_k_slices,
                    options.kernel_glob,
                    KERNEL_PREFIXES[options.mma],
                )

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(run_bucket, device, bucket)
                for device, bucket in zip(devices, buckets)
            ]
            for future in futures:
                future.result()

    report = []
    for job in jobs:
        top = read_top(job, options.top_k)
        winner = top[0]
        report.append(
            {
                "m": job.m,
                "n": job.n,
                "k": job.k,
                "orientation": job.orientation,
                "operation": winner["Operation"],
                "runtime_us": float(winner["Runtime"]) * 1000.0,
                "split_k_slices": int(winner["split_k_slices"]),
                "raster_order": winner["raster_order"],
                "swizzle_size": int(winner["swizzle_size"]),
            }
        )
        print(
            f"{job.m:5}x{job.n:<5}x{job.k:<5} {job.orientation:7} "
            f"{float(winner['Runtime']) * 1000.0:8.3f} us  "
            f"{winner['Operation']}  split={winner['split_k_slices']} "
            f"{winner['raster_order']}/sw{winner['swizzle_size']}"
        )
    with (options.output_dir / "winners.json").open("w") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
