#!/usr/bin/env python3
"""Retune the production W6A8 candidate registry across model shapes."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from model_shapes import DEFAULT_BATCH_SIZES, QWEN35_27B_TP2_NK


ROOT = Path(__file__).resolve().parents[1]


def parse_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_devices(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError("devices must be nonnegative integers")
    return values


def parse_dtypes(value: str) -> tuple[str, ...]:
    dtypes = tuple(item for item in value.split(",") if item)
    if not dtypes or any(dtype not in ("fp16", "bf16") for dtype in dtypes):
        raise argparse.ArgumentTypeError("dtypes must contain fp16 and/or bf16")
    return dtypes


def worker(options: argparse.Namespace) -> None:
    os.environ["MXFP6_LIBRARY_PATH"] = str(options.library.resolve())
    sys.path.insert(0, str(ROOT / "python"))
    import torch
    import mxfp6

    mxfp6.load_library()
    autotune_module = importlib.import_module("mxfp6.autotune")
    results = []
    for shape_index, shape_text in enumerate(options.worker_shapes.split(",")):
        m, n, k = (int(part) for part in shape_text.split("x"))
        generator = torch.Generator(device="cuda").manual_seed(
            options.seed + m * 1000003 + n * 101 + k + shape_index
        )
        activation_source = torch.randn(
            (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        weight_source = torch.randn(
            (n, k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = mxfp6.quantize_mxfp8(activation_source)
        weight = mxfp6.quantize_mxfp6(weight_source)
        for dtype_name in options.output_dtypes:
            out_dtype = (
                torch.float16
                if dtype_name == "fp16"
                else torch.bfloat16
            )
            config = mxfp6.autotune_w6a8(
                activation, weight, out_dtype=out_dtype, force=True
            )
            if config is None:
                raise RuntimeError(f"autotune unexpectedly skipped {m}x{n}x{k}")
            descriptor = autotune_module._descriptor(
                0, m, n, k, out_dtype
            )
            key = autotune_module._cache_key(descriptor)
            cache_entry = json.loads(
                autotune_module._entry_path(key).read_text()
            )
            measurement = cache_entry["measurement"]
            result = {
                "shape": [m, n, k],
                "output_dtype": dtype_name,
                "config_id": config.config_id,
                "kernel": config.kernel,
                "swizzle": config.swizzle,
                "raster_order": config.raster_order,
                "raster": config.raster,
                "latency_us": measurement["latency_us"],
                "fallback_us": measurement["fallback_us"],
                "runner_up_us": measurement["runner_up_us"],
            }
            results.append(result)
            print(
                f"tuned {m}x{n}x{k} {dtype_name}: "
                f"{config.kernel} {config.raster}/sw{config.swizzle}",
                file=sys.stderr,
                flush=True,
            )
    print(json.dumps(results))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=parse_devices, default=(0,))
    parser.add_argument(
        "--batch-sizes", type=parse_values, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument(
        "--output-dtypes", type=parse_dtypes, default=("fp16", "bf16")
    )
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/mxfp6_torch.so"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/runtime_autotune_v4.json",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-shapes", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        if not options.worker_shapes:
            raise ValueError("--worker-shapes is required in worker mode")
        worker(options)
        return

    shapes = [
        (m, n, k)
        for m in options.batch_sizes
        for _, n, k in QWEN35_27B_TP2_NK
    ]
    assignments = [[] for _ in options.devices]
    for index, shape in enumerate(shapes):
        assignments[index % len(assignments)].append(shape)

    processes = []
    for device, assigned in zip(options.devices, assignments):
        if not assigned:
            continue
        shape_arg = ",".join("x".join(map(str, shape)) for shape in assigned)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            f"--worker-shapes={shape_arg}",
            f"--output-dtypes={','.join(options.output_dtypes)}",
            f"--library={options.library.resolve()}",
            f"--seed={options.seed}",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        processes.append(
            (device, subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=None,
            ))
        )

    combined = []
    for device, process in processes:
        stdout, _ = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"autotune worker on physical GPU {device} failed with "
                f"exit code {process.returncode}"
            )
        combined.extend(json.loads(stdout))
    combined.sort(key=lambda item: (*item["shape"], item["output_dtype"]))

    payload = {
        "schema_version": 1,
        "candidate_abi": "native-w6a8-29-v4",
        "devices": list(options.devices),
        "batch_sizes": list(options.batch_sizes),
        "output_dtypes": list(options.output_dtypes),
        "measurement": {
            "warmup": int(os.environ.get("MXFP6_AUTOTUNE_WARMUP", "2")),
            "iterations": int(
                os.environ.get("MXFP6_AUTOTUNE_ITERATIONS", "5")
            ),
            "repeats": int(os.environ.get("MXFP6_AUTOTUNE_REPEATS", "3")),
            "minimum_improvement": float(
                os.environ.get("MXFP6_AUTOTUNE_MIN_IMPROVEMENT", "0.02")
            ),
            "flush_l2_mb": int(
                os.environ.get("MXFP6_AUTOTUNE_FLUSH_L2_MB", "0")
            ),
        },
        "results": combined,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(combined)} decisions to {options.output}")


if __name__ == "__main__":
    main()
