#!/usr/bin/env python3
"""Paired Kineto GPU and separate host-enqueue timings for prepared W6A8.

The persistent baseline uses the serving workspace planner; ephemeral uses
ordinary CUTLASS initialization (including workspace memset) on each call.
Neither host enqueue time nor isolated GEMM duration is end-to-end latency.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', choices=('persistent', 'ephemeral'), default='persistent')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if min(args.samples, args.iterations) < 1:
        parser.error('samples and iterations must be positive')
    os.environ['MXFP6_LIBRARY_PATH'] = str(args.library.resolve())
    os.environ['MXFP6_AUTOTUNE'] = '0'
    sys.path.insert(0, str(ROOT / 'python'))
    import torch
    import mxfp6
    from kineto_utils import load_bench_kineto, kernel_tag
    bench, source = load_bench_kineto(None)
    report = dict(workspace=args.workspace, kineto_source=str(source),
                  device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  cuda=torch.version.cuda, results=[])
    shapes = [(1, 8192, 5120, 2), (33, 512, 2048, 3), (513, 1536, 2048, 16)]
    mxfp6.load_library()
    if args.workspace == 'persistent':
        mxfp6.begin_workspace_planning(torch.device('cuda'))
        for m, n, k, cid in shapes:
            a = torch.zeros(m * k, device='cuda', dtype=torch.uint8)
            b = torch.zeros(n * k * 3 // 4, device='cuda', dtype=torch.uint8)
            sa = torch.full(((m + 127) // 128 * 128 * k // 32,), 127, device='cuda', dtype=torch.uint8)
            sb = torch.full(((n + 127) // 128 * 128 * k // 32,), 127, device='cuda', dtype=torch.uint8)
            for dtype in (torch.float16, torch.bfloat16):
                torch.ops.mxfp6.set_w6a8_config(a, m, n, k, cid, 1, 0, dtype)
                torch.ops.mxfp6.gemm_w6a8(a, b, sa, sb, m, n, k, 1., dtype)
                torch.ops.mxfp6.set_w6a8_config(a, m, n, k, -1, 1, 0, dtype)
        mxfp6.finalize_workspace_planning(torch.device('cuda'))
    for m, n, k, cid in shapes:
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(17)
            a = mxfp6.quantize_activation(torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .1)
            b = mxfp6.quantize_mxfp6(torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .1)
            config = mxfp6.W6A8Config(cid, 1, 0)
            def native():
                return torch.ops.mxfp6.gemm_w6a8_config(
                    a.values, b.values, a.scales, b.scales, m, n, k, 1., cid, 1, 0, dtype)
            prepared = mxfp6.prepare_w6a8(a, b, config=config, out_dtype=dtype)
            torch.testing.assert_close(native(), prepared(), rtol=0, atol=0)
            calls, graphs, tags = {}, [], {}
            for label, fn in [('native', native), ('prepared', prepared)]:
                for _ in range(30):
                    fn()
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    for _ in range(5):
                        fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    output = fn()
                torch.cuda.current_stream().wait_stream(capture_stream)
                graphs.append((graph, output))
                for mode, call in [('eager', fn), ('graph', graph.replay)]:
                    calls[label, mode] = call
                    tags[label, mode], _ = kernel_tag(call)
            samples = {}
            for sample in range(args.samples):
                labels = ('native', 'prepared') if sample % 2 == 0 else ('prepared', 'native')
                for mode in ('eager', 'graph'):
                    for cold in (False, True):
                        for label in labels:
                            key = f'{label}_{mode}_{"cold" if cold else "warm"}_us'
                            value = bench(calls[label, mode], tags[label, mode], num_tests=args.iterations,
                                          suppress_kineto_output=True, flush_l2=cold) * 1e6
                            if value <= 0:
                                raise RuntimeError('Kineto returned no kernel timing')
                            samples.setdefault(key, []).append(value)
                for label in labels:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(100):
                        calls[label, 'eager']()
                    value = (time.perf_counter() - start) * 1e4
                    torch.cuda.synchronize()
                    samples.setdefault(label + '_host_enqueue_us', []).append(value)
            report['results'].append(dict(shape=[m, n, k], config_id=cid, dtype=str(dtype),
                samples=samples, medians={key: statistics.median(value) for key, value in samples.items()}))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            for graph, _ in graphs:
                graph.reset()
            del calls, graphs, output, prepared, a, b


if __name__ == '__main__':
    main()
