"""Measure W8A8 workspace bytes and warm eager/graph latency in a fresh process."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import mxfp6.mxfp8 as api


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=["temporary", "global", "per-stream"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(91)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    runs = []
    for stream, (m, n, k, t) in zip(
        streams, [(1, 128, 8192, 12), (512, 8192, 2048, 16)]
    ):
        with torch.cuda.stream(stream):
            a = api.quantize_mxfp8(
                torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            )
            b = api.quantize_mxfp8(
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            )
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            run = lambda a=a, b=b, out=out, t=t: api.gemm_w8a8(
                a, b, out=out, config=api.W8A8Config(t, 4)
            )
            run()
            runs.append((run, (m, n, k, t)))
    torch.cuda.synchronize()
    if args.mode != "temporary":
        api.begin_workspace_planning()
        for stream, (run, _) in zip(streams, runs):
            with torch.cuda.stream(stream):
                run()
        if args.mode == "global":
            api.finalize_workspace_planning()
        else:
            api.finalize_workspace_planning(per_stream=True)
    rows = []
    for stream, (run, shape) in zip(streams, runs):
        with torch.cuda.stream(stream):
            for _ in range(20):
                run()
            stream.synchronize()
            eager = []
            for _ in range(100):
                start = time.perf_counter_ns()
                out = run()
                stream.synchronize()
                eager.append((time.perf_counter_ns() - start) / 1000)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(20):
                    out = run()
            graph_us = []
            for _ in range(20):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                graph_us.append(begin.elapsed_time(end) * 1000 / 20)
            digest = hashlib.sha256(
                out.cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        rows.append(
            dict(
                shape=shape,
                sha256=digest,
                eager_median_us=statistics.median(eager),
                eager_p99_us=sorted(eager)[98],
                graph_median_us=statistics.median(graph_us),
                eager_samples_us=eager,
                graph_samples_us=graph_us,
            )
        )
    args.output.write_text(
        json.dumps(
            dict(mode=args.mode, stats=api.workspace_stats(), rows=rows), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
