#!/usr/bin/env python3
"""Randomized cross-layout stress test for persistent Stream-K workspaces."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_M = (
    1,
    8,
    9,
    16,
    17,
    128,
    129,
    256,
    257,
    512,
    513,
    640,
    641,
    768,
    769,
    1024,
    1025,
    1152,
    1153,
    1280,
    1281,
    1536,
    1537,
    1664,
    1665,
    1792,
    1793,
    2048,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--concurrent-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/mxfp6_torch.so"
    )
    options = parser.parse_args()
    if options.iterations <= 0 or options.concurrent_iterations <= 0:
        raise ValueError("iteration counts must be positive")

    os.environ["MXFP6_LIBRARY_PATH"] = str(options.library.resolve())
    sys.path.insert(0, str(ROOT / "python"))
    import mxfp6

    mxfp6.load_library()
    device = torch.device("cuda")
    n, k = 128, 8704
    generator = torch.Generator(device=device).manual_seed(options.seed)
    b_codes = torch.randint(
        0, 64, (n, k), generator=generator, device=device, dtype=torch.uint8
    )
    b_scales = torch.full(
        (n, k // 32), 0x7F, device=device, dtype=torch.uint8
    )
    weight = mxfp6.pack_operand(b_codes, b_scales)

    problems = {}
    for index, m in enumerate(BOUNDARY_M):
        a_codes = torch.randint(
            0, 64, (m, k), generator=generator, device=device, dtype=torch.uint8
        )
        a_scales = torch.full(
            (m, k // 32), 0x7F, device=device, dtype=torch.uint8
        )
        a6 = mxfp6.pack_operand(a_codes, a_scales)
        a8 = torch.ops.mxfp6.expand_fp6_to_fp8(a6.values, m, k)
        out_dtype = torch.float16 if index % 2 == 0 else torch.bfloat16
        config_id = 3 if m <= 17 else 16
        swizzle = (1, 2, 4, 8)[index % 4]
        raster = index % 3
        torch.ops.mxfp6.set_w6a8_config(
            a8, m, n, k, config_id, swizzle, raster, out_dtype
        )

        def run_w6a8(
            a8=a8,
            a6=a6,
            m=m,
            out_dtype=out_dtype,
        ):
            return torch.ops.mxfp6.gemm_w6a8(
                a8,
                weight.values,
                a6.scales,
                weight.scales,
                m,
                n,
                k,
                1.0,
                out_dtype,
            )

        def run_split_k(a6=a6, m=m, out_dtype=out_dtype):
            return torch.ops.mxfp6.gemm(
                a6.values,
                weight.values,
                a6.scales,
                weight.scales,
                m,
                n,
                k,
                1.0,
                out_dtype,
            )

        problems[(m, "stream_k")] = run_w6a8
        problems[(m, "split_k")] = run_split_k

    references = {key: run() for key, run in problems.items()}
    torch.cuda.synchronize()

    mxfp6.begin_workspace_planning()
    for run in problems.values():
        run()
    torch.cuda.synchronize()
    planning = mxfp6.workspace_stats()
    if planning["layouts"] == 0 or planning["arena_bytes"] == 0:
        raise RuntimeError(f"planner did not collect Stream-K layouts: {planning}")
    mxfp6.finalize_workspace_planning()

    keys = list(problems)
    randomizer = random.Random(options.seed)
    latest = {}
    for _ in range(options.iterations):
        key = randomizer.choice(keys)
        latest[key] = problems[key]()
    torch.cuda.synchronize()
    for key, output in latest.items():
        torch.testing.assert_close(output, references[key], rtol=0, atol=0)

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    # Exercise two persistent lanes concurrently. Split-K is already mixed
    # into the randomized loop above as a control for workspace isolation.
    concurrent_keys = ((1, "stream_k"), (128, "stream_k"))
    with torch.cuda.stream(stream_a):
        output_a = problems[concurrent_keys[0]]()
    with torch.cuda.stream(stream_b):
        output_b = problems[concurrent_keys[1]]()
    stream_a.synchronize()
    stream_b.synchronize()
    for _ in range(options.concurrent_iterations):
        with torch.cuda.stream(stream_a):
            output_a = problems[concurrent_keys[0]]()
        with torch.cuda.stream(stream_b):
            output_b = problems[concurrent_keys[1]]()
    stream_a.synchronize()
    stream_b.synchronize()
    torch.testing.assert_close(
        output_a, references[concurrent_keys[0]], rtol=0, atol=0
    )
    torch.testing.assert_close(
        output_b, references[concurrent_keys[1]], rtol=0, atol=0
    )

    if not mxfp6.workspace_barriers_zero():
        raise RuntimeError("a persistent barrier remained nonzero")
    stats = mxfp6.workspace_stats()
    if stats["fallback_launches"] != 0:
        raise RuntimeError(f"unexpected persistent fallback: {stats}")
    print(
        f"PASS {len(BOUNDARY_M)} boundary M values, {options.iterations} "
        f"random and {options.concurrent_iterations} dual-stream launches; "
        f"workspace={stats}"
    )


if __name__ == "__main__":
    main()
