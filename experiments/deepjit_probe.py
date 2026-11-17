#!/usr/bin/env python3
"""Compare the optional prepared JIT API with AOT using bench_kineto."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'benchmarks')]


def main():
    import torch
    import mxfp6
    import mxfp6.mxfp8 as fp8
    from mxfp6._mxfp8_config import configurations, load_manifest
    from mxfp6._mxfp8_jit import _context
    from mxfp6._mxfp8_jit_source import device_source
    from kineto_utils import load_bench_kineto, kernel_tag

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--tactic', type=int, default=560)
    parser.add_argument('--shape', type=int, nargs=3, default=[32, 512, 2560], metavar=('M', 'N', 'K'))
    args = parser.parse_args()
    root = args.source_root.resolve()
    m, n, k = args.shape
    if m <= 0 or n <= 0 or n % 128 or k <= 0 or k % 128:
        parser.error('positive M and N/K divisible by 128 are required')
    configs = {c['tactic']: c for c in configurations(load_manifest(root/'python/mxfp6/mxfp8_cp_async.json'))}
    if args.tactic not in configs:
        parser.error('tactic has not been migrated to JIT')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = (args.cache or Path(tempfile.mkdtemp(prefix='jit-cache-', dir=args.output.parent))).resolve()
    os.environ['MXFP8_JIT_CACHE_DIR'] = str(cache)
    torch.cuda.init()
    start = time.perf_counter()
    _, signature, _, _ = _context(root)
    initialization_seconds = time.perf_counter() - start
    config = fp8.W8A8Config(args.tactic)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        handle = fp8.precompile_jit(config, k, source_root=root)
        timings.append(time.perf_counter() - start)
    source = device_source(root, configs[args.tactic], k)
    report = dict(
        deepjit_commit=subprocess.check_output(['git', '-C', str(root/'third_party/deepjit'), 'rev-parse', 'HEAD'], text=True).strip(),
        cache=str(cache), backend_initialization_seconds=initialization_seconds,
        compile_load_seconds=timings, cubin=handle.cubin, arch='sm_120a',
        ptxas_register_usage_level=5, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        dependency_sha256=signature, shape=args.shape, tactic=args.tactic)
    if not args.compile_only:
        bench, bench_path = load_bench_kineto()
        torch.manual_seed(20261116)
        a = mxfp6.quantize_activation(torch.randn(m, k, device='cuda', dtype=torch.bfloat16))
        b = fp8.quantize_mxfp8(torch.randn(n, k, device='cuda', dtype=torch.bfloat16))
        aot = fp8.prepare(a, b, config=config)
        jit = fp8.prepare_jit(a, b, config=config, source_root=root)
        torch.testing.assert_close(jit(), aot(), rtol=0, atol=0)
        graphs = {}
        for name, run in [('aot', aot), ('jit', jit)]:
            for _ in range(5):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            tag, _ = kernel_tag(graph.replay)
            graphs[name] = (graph, tag)
        samples = {name: {'warm': [], 'cold': []} for name in graphs}
        eager = {name: (run, kernel_tag(run)[0]) for name, run in [('aot', aot), ('jit', jit)]}
        eager_samples = {name: {'warm': [], 'cold': []} for name in graphs}
        for trial in range(5):
            for name in (('aot', 'jit') if trial % 2 == 0 else ('jit', 'aot')):
                graph, tag = graphs[name]
                for regime in ('warm', 'cold'):
                    samples[name][regime].append(bench(graph.replay, tag, num_tests=30,
                        suppress_kineto_output=True, flush_l2=regime == 'cold') * 1e6)
                    run, eager_tag = eager[name]
                    eager_samples[name][regime].append(bench(run, eager_tag, num_tests=30,
                        suppress_kineto_output=True, flush_l2=regime == 'cold') * 1e6)
        torch.testing.assert_close(jit(), aot(), rtol=0, atol=0)
        report.update(outputs_bitwise_equal=True, timings_us=samples, eager_timings_us=eager_samples,
                      timer='bench_kineto', kineto_source=str(bench_path))
        for graph, _ in graphs.values():
            graph.reset()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
