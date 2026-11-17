"""CUDA Graph measurement shared by native candidate tuners.

Warm and explicit-flush regimes retain different event boundaries. This module
owns graph storage until every replay has finished, including failure cleanup.
"""
from collections.abc import Callable
import statistics

import torch


def measure_graph(
    run: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
    flush: torch.Tensor | None,
) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    output = None
    captured = False
    if flush is None:
        starts = [torch.cuda.Event(enable_timing=True, external=True)]
        ends = [torch.cuda.Event(enable_timing=True, external=True)]
    else:
        starts = [
            torch.cuda.Event(enable_timing=True, external=True)
            for _ in range(iterations)
        ]
        ends = [
            torch.cuda.Event(enable_timing=True, external=True)
            for _ in range(iterations)
        ]

    try:
        # Keep PyTorch's fail-closed global capture mode.  A service with
        # conflicting CUDA work in another thread should retain the builtin
        # dispatcher instead of weakening capture-safety checks.
        with torch.cuda.graph(graph):
            if flush is None:
                starts[0].record()
            for iteration in range(iterations):
                if flush is not None:
                    flush.zero_()
                    starts[iteration].record()
                output = run()
                if flush is not None:
                    ends[iteration].record()
            if flush is None:
                ends[0].record()
        captured = True
        samples = []
        for _ in range(repeats):
            # Keep unrelated stream work out of the captured timing interval.
            torch.cuda.synchronize(device)
            graph.replay()
            ends[-1].synchronize()
            elapsed_us = [
                start.elapsed_time(end) * 1_000
                for start, end in zip(starts, ends, strict=True)
            ]
            samples.append(sum(elapsed_us) / iterations)
        return statistics.median(samples)
    finally:
        torch.cuda.synchronize(device)
        if captured:
            graph.reset()
        # The captured output owns graph-private storage until replay ends.
        del output, graph
