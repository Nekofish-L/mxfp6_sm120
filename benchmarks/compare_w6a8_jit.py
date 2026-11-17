#!/usr/bin/env python3
"""AOT/JIT Graph comparison using identical owned Params and device addresses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    if min(args.samples, args.iterations) < 1:
        parser.error('samples and iterations must be positive')
    os.environ['MXFP6_LIBRARY_PATH'] = str(args.library.resolve())
    os.environ['MXFP6_AUTOTUNE'] = '0'
    sys.path.insert(0, str(ROOT / 'python'))
    import torch
    import mxfp6
    from mxfp6._w6a8_jit import bind
    from kineto_utils import load_bench_kineto, kernel_tag
    bench, source = load_bench_kineto(None)
    report = dict(method=__doc__, library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  cuda=torch.version.cuda, kineto_source=str(source),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), results=[])
    # Include every configuration whose separate JIT compilation changed SASS
    # in the migration audit, plus representative large Stream-K workloads.
    shapes = [(1, 7168, 5120, 0), (1, 8192, 5120, 1),
              (16, 17408, 5120, 4), (16, 8192, 5120, 5), (16, 7168, 5120, 29),
              (1, 8192, 5120, 2), (1, 5120, 3072, 2), (1, 7168, 5120, 2),
              (33, 512, 2048, 3), (513, 1536, 2048, 16), (1024, 8192, 5120, 16),
              (2048, 7168, 5120, 16), (8192, 5120, 8704, 16)]
    for m, n, k, cid in shapes:
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(71)
            a = mxfp6.quantize_activation(torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .1)
            b = mxfp6.quantize_mxfp6(torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .1)
            run = mxfp6.prepare_w6a8(a, b, config=mxfp6.W6A8Config(cid, 1, 0), out_dtype=dtype)
            expected = run().clone()
            graphs, tags = {}, {}
            for backend in ('aot', 'jit'):
                if backend == 'jit':
                    compiled = bind(run._native, run.config, dtype, a.device)
                for _ in range(30):
                    output = run()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run()
                graphs[backend] = graph
                tags[backend], _ = kernel_tag(graph.replay)
            samples = {}
            for sample in range(args.samples):
                order = ('aot', 'jit') if sample % 2 == 0 else ('jit', 'aot')
                for cold in (False, True):
                    for backend in order:
                        key = backend + ('_cold_us' if cold else '_warm_us')
                        value = bench(graphs[backend].replay, tags[backend], num_tests=args.iterations,
                                      suppress_kineto_output=True, flush_l2=cold) * 1e6
                        if value <= 0:
                            raise RuntimeError('Kineto returned no kernel timing')
                        samples.setdefault(key, []).append(value)
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
            medians = {key: statistics.median(values) for key, values in samples.items()}
            report['results'].append(dict(shape=[m, n, k], config_id=run.config.config_id,
                dtype=str(dtype), bitwise_equal=True, cubin=compiled.cubin(), samples=samples,
                medians=medians, change_percent={regime: (medians['jit_' + regime + '_us'] /
                    medians['aot_' + regime + '_us'] - 1) * 100 for regime in ('warm', 'cold')}))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            for graph in graphs.values():
                graph.reset()


if __name__ == '__main__':
    main()
