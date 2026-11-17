"""Record deterministic outputs and graph/eager latency for API refactors.

Select the package via PYTHONPATH and native library via MXFP6_LIBRARY_PATH.
Run baseline and candidate in separate processes with MXFP6_AUTOTUNE=0.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import mxfp6

SHAPES = [
    (1, 8192, 5120),
    (1, 5120, 8704),
    (16, 5120, 3072),
    (24, 7168, 5120),
    (32, 17408, 5120),
    (40, 5120, 3072),
    (48, 5120, 3072),
    (64, 5120, 8704),
    (96, 17408, 5120),
    (512, 7168, 5120),
    (1024, 8192, 5120),
    (1536, 5120, 3072),
    (2048, 5120, 3072),
    (4096, 5120, 3072),
    (8192, 5120, 8704),
    (3, 136, 128),
    (17, 256, 512),
    (129, 384, 2048),
]


def measure(fn):
    for _ in range(5):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            result = fn()
    samples = []
    for _ in range(7):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    eager = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        eager.append((time.perf_counter() - start) * 1e6 / 20)
    return result, samples, eager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", help="Comma-separated MxNxK shapes")
    args = parser.parse_args()
    shapes = (
        [tuple(map(int, x.split("x"))) for x in args.shapes.split(",")]
        if args.shapes
        else SHAPES
    )
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in shapes:
            torch.manual_seed(8128)
            a = torch.randn(m, k, device="cuda", dtype=dtype) * 0.1
            b = mxfp6.quantize_mxfp6(
                torch.randn(n, k, device="cuda", dtype=dtype) * 0.1
            )
            qa = mxfp6.quantize_mxfp8(a)
            for path, x in [("float", a), ("quantized", qa)]:
                result, graph_samples, eager_samples = measure(
                    lambda: mxfp6.gemm(x, b, out_dtype=dtype)
                )
                digest = hashlib.sha256(
                    result.cpu().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                rows.append(
                    dict(
                        shape=[m, n, k],
                        dtype=str(dtype),
                        path=path,
                        sha256=digest,
                        graph_us=statistics.median(graph_samples),
                        graph_samples_us=graph_samples,
                        eager_us=statistics.median(eager_samples),
                        eager_samples_us=eager_samples,
                    )
                )
            print(dtype, m, n, k, flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                library=str(mxfp6.load_library()),
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
