"""Paired PDL measurements with real weights and a rotating MLP dependency chain.

Example: CUDA_VISIBLE_DEVICES=6 PYTHONPATH=python python benchmarks/benchmark_mxfp8_pdl.py
  --checkpoint /path/to/model.safetensors --vllm-source /path/to/vllm-mach/src
  --output /tmp/pdl.json
Both arms use identical inputs, weights, tactics, and arithmetic. Graph timings
are GPU timings; they do not establish serving throughput. The chain rotates
eight layers' weights to exceed L2 capacity on RTX 5090.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys

import torch
from safetensors import safe_open

import mxfp6
import mxfp6.mxfp8 as mx


def measure(functions, args, calls):
    graphs, outputs = {}, {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for name, fn in functions.items():
        with torch.cuda.stream(stream):
            for _ in range(5):
                fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                outputs[name] = fn()
        graphs[name] = graph
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            graph = graphs[name]
            for _ in range(5):
                graph.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (calls * args.replays))
    equal = []
    for amplitude in (0, .125, 8, 1):
        args.mutate(amplitude)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        left, right = outputs['off'], outputs['on']
        if isinstance(left, torch.Tensor):
            left, right = (left,), (right,)
        equal.append(all(torch.equal(a, b) for a, b in zip(left, right)))
    if not all(equal):
        raise AssertionError(f'PDL changed graph output: {equal}')
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return dict(graph_us=medians, raw_graph_us=samples,
                speedup_pct=(medians['off'] / medians['on'] - 1) * 100,
                changing_input_bitwise_equal=equal)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vllm-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', nargs='+', type=int,
                        default=[1, 16, 32, 33, 48, 64, 65, 96, 128, 192, 256, 512, 1024, 2048])
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--replays', type=int, default=20)
    parser.add_argument('--graph-calls', type=int, default=16)
    parser.add_argument('--chain-layers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--temporary-workspace', action='store_true',
                        help='Use fresh temporary workspaces instead of the serving arena')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    mxfp6.load_library()
    library = mx.load_library()
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
              'checkpoint': args.checkpoint.name, 'rows': [], 'chains': [],
              'method': {'rounds': args.rounds, 'replays': args.replays,
                         'graph_calls': args.graph_calls, 'seed': args.seed,
                         'chain_layers': args.chain_layers,
                         'metric': 'median GPU microseconds; randomized paired CUDA Graph replay',
                         'projection_cache': 'repeated single projection, hot weights',
                         'chain_cache': 'distinct layers; total weight footprint recorded'}}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    with safe_open(args.checkpoint, framework='pt') as checkpoint:
        def weight(layer, parts):
            prefix = f'model.language_model.layers.{layer}.'
            values = torch.cat([checkpoint.get_tensor(prefix + part + '.weight').cuda()
                                for part in parts]).contiguous()
            scales = mx.pack_scales(torch.cat([
                checkpoint.get_tensor(prefix + part + '.weight_scale').cuda() for part in parts]))
            return values, scales

        specs = {'qkvz': ('linear_attn.in_proj_qkv', 'linear_attn.in_proj_z'),
                 'gateup': ('mlp.gate_proj', 'mlp.up_proj'),
                 'down': ('mlp.down_proj',), 'out': ('linear_attn.out_proj',)}
        weights = {name: weight(1, parts) for name, parts in specs.items()}
        if not args.temporary_workspace:
            mx.begin_workspace_planning()
            for values, scales in weights.values():
                wrapped = mx.MXFP8Tensor(values.view(torch.uint8), scales, *values.shape)
                for m in args.rows:
                    mx.warmup(torch.zeros(m, values.shape[1], device='cuda', dtype=torch.bfloat16),
                              wrapped, iterations=1)
            result['workspace'] = dict(mx.finalize_workspace_planning())
        else:
            result['workspace'] = {'mode': 'temporary'}
        for name, (values, scales) in weights.items():
            n, k = values.shape
            for m in args.rows:
                x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                args.mutate = lambda amplitude: x.normal_().mul_(amplitude)
                functions = {
                    'off': lambda: torch.ops.mxfp8_sm120.gemm_from_float(x, values, scales),
                    'on': lambda: torch.ops.mxfp8_sm120.gemm_from_float_pdl(x, values, scales),
                }
                calls = args.graph_calls if m <= 256 else min(4, args.graph_calls)
                row = dict(name=name, shape=[m, n, k], config=list(mx.select_config(m, n, k)),
                           graph_calls=calls, **measure(functions, args, calls))
                result['rows'].append(row)
                print(name, m, row['graph_us'], f"{row['speedup_pct']:+.2f}%", flush=True)
                save()

        if args.vllm_source:
            sys.path.insert(0, str(args.vllm_source))
            from vllm_mach.mxfp6 import dense_mxfp8, gemma_norm, mxfp8_mlp
            os.environ['VLLM_MACH_MXFP8_BACKEND'] = 'native'
            os.environ['VLLM_MACH_MXFP8_PDL_MAX_ROWS'] = str(max(args.rows))
            layers = []
            for layer in range(1, args.chain_layers + 1):
                up, us = weight(layer, ('mlp.gate_proj', 'mlp.up_proj'))
                down, ds = weight(layer, ('mlp.down_proj',))
                nw = checkpoint.get_tensor(
                    f'model.language_model.layers.{layer}.post_attention_layernorm.weight').cuda()
                layers.append((up, us, down, ds, nw))
            footprint = sum(t.numel() * t.element_size() for layer in layers for t in layer)
            for m in args.rows:
                x = torch.randn(m, 2560, device='cuda', dtype=torch.bfloat16)
                residual = torch.randn_like(x)

                def mutate(amplitude):
                    x.normal_().mul_(amplitude)
                    residual.normal_().mul_(amplitude)

                args.mutate = mutate

                def chain(enabled):
                    os.environ['VLLM_MACH_MXFP8_PDL'] = str(int(enabled))
                    y, r = x, residual
                    for up, us, down, ds, nw in layers:
                        norm, r = gemma_norm._impl(y, r, nw, 1e-6)
                        gate = dense_mxfp8._gemm(norm, up, us)
                        y = mxfp8_mlp._down(gate, down, ds)
                    return y, r

                row = dict(m=m, weight_bytes=footprint, graph_calls=1,
                           **measure({'off': lambda: chain(False), 'on': lambda: chain(True)}, args, 1))
                result['chains'].append(row)
                print('chain', m, row['graph_us'], f"{row['speedup_pct']:+.2f}%", flush=True)
                save()


if __name__ == '__main__':
    main()
