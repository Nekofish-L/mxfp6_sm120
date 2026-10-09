"""Compare native dual and single MXFP8 for every batch 1–128.

Measures GEMM with prequantized inputs and BF16 quantization + GEMM, each with
hot weights and L2 eviction. No service, input padding, or batch splitting.
Both paths use the same checkpoint MXFP8 weights and BF16 inputs. Accuracy is
relative to FP32 BF16-input GEMM with dequantized MXFP8 weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
from safetensors import safe_open

import mxfp6
import mxfp6.mxfp8 as single
import mxfp6.mxfp8_dual as dual


def capture(functions, stream, calls):
    graphs, outputs = {}, {}
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
    return graphs, outputs


def measure(functions, stream, eviction, args, rng):
    hot, _ = capture(functions, stream, args.hot_calls)
    cold, outputs = capture(functions, stream, 1)
    samples = {cache: {name: [] for name in functions} for cache in ('hot', 'cold')}
    for _ in range(args.rounds):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            graph = hot[name]
            for _ in range(5):
                graph.replay()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples['hot'][name].append(start.elapsed_time(end) * 1000 /
                                        (args.hot_calls * args.replays))
        rng.shuffle(names)
        for name in names:
            events = []
            for _ in range(args.cold_replays):
                eviction.zero_()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                cold[name].replay()
                end.record()
                events.append((start, end))
            events[-1][1].synchronize()
            samples['cold'][name].append(statistics.mean(
                start.elapsed_time(end) * 1000 for start, end in events))
    result = {}
    for cache, arms in samples.items():
        medians = {name: statistics.median(values) for name, values in arms.items()}
        result[cache] = dict(us=medians, raw_trial_us=arms,
                             dual_over_single=medians['dual'] / medians['single'],
                             dual_latency_overhead_pct=(medians['dual'] /
                                                        medians['single'] - 1) * 100)
    return result, cold, outputs


def relative_l2(out, reference):
    assert torch.isfinite(out).all()
    return ((out.float() - reference).norm() /
            reference.norm().clamp_min(1.e-20)).item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', nargs='+', type=int, default=list(range(1, 129)))
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--replays', type=int, default=20)
    parser.add_argument('--hot-calls', type=int, default=16)
    parser.add_argument('--cold-replays', type=int, default=5)
    parser.add_argument('--eviction-mib', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    if (any(not 1 <= m <= 128 for m in args.rows)
            or min(args.rounds, args.replays, args.hot_calls, args.cold_replays) < 1):
        parser.error('rows must be 1–128 and sampling counts must be positive')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    mxfp6.load_library()
    library = single.load_library()
    props = torch.cuda.get_device_properties(0)
    eviction = torch.empty(args.eviction_mib * 1024**2, dtype=torch.uint8, device='cuda')
    if eviction.numel() <= props.L2_cache_size:
        parser.error('eviction buffer must exceed device L2 capacity')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), root/'python/mxfp6/mxfp8_dual.py',
                    root/'csrc/mxfp8_dual_extension.cu', root/'csrc/mxfp8_dual_small.cu',
                    root/'csrc/mxfp8_dual_pipereg.cu', root/'csrc/include/mxfp8_gemm/dual.cuh',
                    root/'python/mxfp6/mxfp8_dispatch.json']
    result = {'schema_version': 1, 'gpu': str(props), 'torch': torch.__version__,
              'checkpoint': str(args.checkpoint),
              'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_paths},
              'method': {'boundaries': ['gemm_only', 'quantize_gemm'],
                         'single': 'current native default dispatch, PDL off',
                         'dual': 'one native dual GEMM; logical M equals requested batch',
                         'padding_or_batch_splitting': False,
                         'hot': 'repeated Graph replay; median of randomized paired trial averages',
                         'cold': 'one invocation per Graph; eviction before each timed replay',
                         'eviction_bytes': eviction.numel(), 'l2_bytes': props.L2_cache_size,
                         'rounds': args.rounds, 'hot_calls': args.hot_calls,
                         'hot_replays': args.replays, 'cold_replays': args.cold_replays,
                         'accuracy_reference': 'FP32 BF16-input @ dequantized MXFP8-weight.T; TF32 off',
                         'seed': args.seed}, 'rows': []}
    rng = random.Random(args.seed)

    def save():
        (args.output/'results.json').write_text(json.dumps(result, indent=2) + '\n')
        with (args.output/'comparison.csv').open('w', newline='') as file:
            fields = ['projection', 'm', 'n', 'k', 'dual_variant', 'single_relative_l2',
                      'dual_relative_l2', 'error_reduction_x']
            fields += [f'{boundary}_{cache}_{metric}'
                       for boundary in ('gemm_only', 'quantize_gemm')
                       for cache in ('hot', 'cold')
                       for metric in ('single_us', 'dual_us', 'dual_over_single', 'overhead_pct')]
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            for row in result['rows']:
                flat = {key: row[key] for key in fields[:8]}
                for boundary in ('gemm_only', 'quantize_gemm'):
                    for cache in ('hot', 'cold'):
                        measured = row[boundary][cache]
                        prefix = f'{boundary}_{cache}_'
                        flat.update({prefix+'single_us': measured['us']['single'],
                                     prefix+'dual_us': measured['us']['dual'],
                                     prefix+'dual_over_single': measured['dual_over_single'],
                                     prefix+'overhead_pct': measured['dual_latency_overhead_pct']})
                writer.writerow(flat)

    with safe_open(args.checkpoint, framework='pt') as checkpoint:
        weights = {}
        for name, parts in {'qkvz': ('linear_attn.in_proj_qkv', 'linear_attn.in_proj_z'),
                            'gateup': ('mlp.gate_proj', 'mlp.up_proj')}.items():
            prefix = 'model.language_model.layers.1.'
            values = torch.cat([checkpoint.get_tensor(prefix+p+'.weight').cuda()
                                for p in parts]).contiguous()
            logical = torch.cat([checkpoint.get_tensor(prefix+p+'.weight_scale').cuda()
                                 for p in parts]).contiguous()
            weights[name] = (values, single.pack_scales(logical), logical)
        single.begin_workspace_planning()
        for values, scales, _ in weights.values():
            wrapped = single.MXFP8Tensor(values.view(torch.uint8), scales, *values.shape)
            for m in args.rows:
                single.warmup(torch.zeros(m, values.shape[1], device='cuda', dtype=torch.bfloat16),
                              wrapped, iterations=1)
        result['workspace'] = dict(single.finalize_workspace_planning())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for values, scales, _ in weights.values():
                wrapped = single.MXFP8Tensor(values.view(torch.uint8), scales, *values.shape)
                for m in args.rows:
                    single.warmup(torch.zeros(m, values.shape[1], device='cuda', dtype=torch.bfloat16),
                                  wrapped, iterations=1)
        torch.cuda.current_stream().wait_stream(stream)
        for name, (values, scales, logical) in weights.items():
            n, k = values.shape
            decoded_weight = values.float() * torch.exp2(logical.float() - 127).repeat_interleave(32, 1)
            wrapped_weight = single.MXFP8Tensor(values.view(torch.uint8), scales, n, k)
            for m in args.rows:
                x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                limbs = dual.quantize(x)
                hi, sh, residual, sr = limbs
                dual_out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
                single_input = single.quantize_mxfp8(x)
                single_gemm = single.prepare(single_input, wrapped_weight)

                def dual_gemm():
                    return dual.gemm_out(hi, values, sh, scales, residual, sr, dual_out)

                def dual_pipeline():
                    dual.quantize_out(x, *limbs)
                    return dual_gemm()

                def single_pipeline():
                    return torch.ops.mxfp8_sm120.gemm_from_float(x, values, scales)

                kernel, _, _ = measure({'single': single_gemm, 'dual': dual_gemm},
                                       stream, eviction, args, rng)
                pipeline, graphs, outputs = measure({'single': single_pipeline,
                                                     'dual': dual_pipeline},
                                                    stream, eviction, args, rng)
                errors = {arm: relative_l2(out, x.float() @ decoded_weight.T)
                          for arm, out in outputs.items()}
                assert errors['dual'] < .005 and errors['single'] < .06, (m, n, errors)
                bitwise = []
                for factor in (0., .125, 8., 1.):
                    x.normal_().mul_(factor)
                    dual_out.fill_(float('nan'))
                    for tensor in limbs:
                        tensor.view(torch.uint8).fill_(255)
                    for graph in graphs.values():
                        graph.replay()
                    torch.cuda.synchronize()
                    expected = torch.empty_like(dual_out)
                    dual.gemm_out(hi, values, sh, scales, residual, sr, expected)
                    equality = torch.equal(outputs['dual'], expected)
                    assert equality, (m, n, factor, 'Graph/eager mismatch')
                    reference = x.float() @ decoded_weight.T
                    assert relative_l2(outputs['dual'], reference) < .005, (m, n, factor)
                    if factor == 0:
                        assert torch.count_nonzero(outputs['dual']).item() == 0
                    bitwise.append(equality)
                row = dict(projection=name, m=m, n=n, k=k,
                           dual_variant=dual.select_variant(m, n, k),
                           single_config=list(single_gemm.config),
                           single_relative_l2=errors['single'], dual_relative_l2=errors['dual'],
                           error_reduction_x=errors['single']/max(errors['dual'], 1.e-20),
                           graph_eager_changing_input_bitwise_equal=bitwise,
                           gemm_only=kernel, quantize_gemm=pipeline)
                result['rows'].append(row)
                print(name, m, 'hot', pipeline['hot']['us'],
                      'cold', pipeline['cold']['us'],
                      'relative_l2', errors, flush=True)
                save()


if __name__ == '__main__':
    main()
