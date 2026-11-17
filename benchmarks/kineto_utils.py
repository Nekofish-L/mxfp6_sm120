"""Load the requested bench_kineto implementation without importing DeepGEMM."""
import importlib.util
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / 'DeepGEMM/deep_gemm/testing/bench.py'


def load_bench_kineto(source=None):
    if int(os.environ.get('DG_USE_NVIDIA_TOOLS', '0')):
        raise RuntimeError('bench_kineto is disabled by DG_USE_NVIDIA_TOOLS')
    path = Path(source or os.environ.get('MXFP_BENCH_KINETO', DEFAULT_SOURCE)).resolve()
    spec = importlib.util.spec_from_file_location('_mxfp_bench_kineto_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bench_kineto, path


def kernel_tag(run):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run()
        torch.cuda.synchronize()
    names = [event.name for event in profiler.events()
             if event.device_type == torch.autograd.DeviceType.CUDA
             and not any(word in event.name.lower() for word in ('memset', 'memcpy'))]
    if len(names) != 1:
        raise RuntimeError(f'Expected one GEMM kernel, found {sorted(names)}')
    name = names[0]
    # bench_kineto renders a table with max_name_column_width=100.
    return name[:70], name


def measure_kineto(run, *, num_tests, flush_l2, bench_kineto):
    for _ in range(3):
        output = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    try:
        tag, _ = kernel_tag(graph.replay)
        value = bench_kineto(graph.replay, tag, num_tests=num_tests,
                             suppress_kineto_output=True, flush_l2=flush_l2) * 1e6
        if value <= 0:
            raise RuntimeError('bench_kineto returned no kernel timing')
        return value
    finally:
        torch.cuda.synchronize()
        graph.reset()
        del output, graph
