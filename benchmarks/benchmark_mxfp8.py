#!/usr/bin/env python3
"""Hot CUDA graph launch + cold weight cache, using the user's gemm_bench.py."""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import flashinfer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from mxfp6.mxfp8 import load_library, prepare, select_config


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture(fn):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def inspect_graph(graph):
    """Ensure reference bench's average kernel time equals one GEMM invocation."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    names = [e.name for e in events]
    assert len(names) == 1, f'Expected one kernel per graph, got {names}'
    for tag in ['cutlass', 'humming', 'mxfp8_sm120::gemv', '_mm']:
        if tag in names[0]:
            return tag, names[0]
    raise AssertionError(f'Unrecognized kernel: {names}')


def measure(bench, graph, tag, tests, trials):
    samples = []
    for _ in range(trials):
        value = bench.bench_kineto(graph.replay, tag, num_tests=tests,
                                   suppress_kineto_output=True, flush_l2=True,
                                   with_multiple_kernels=False)
        assert value > 0, (tag, value)
        samples.append(value * 1e6)
    return samples


def candidates(m):
    static = [t for t in range(33) if t not in [12,13,14,15,16,30]]
    if m <= 16: static += [33,34,35,36]
    static += [37,38,41] + list(range(43,51)) + [52,53,54,55,56]
    for t in static:
        yield t,1,1
    for t in [12,13,14,15,16,39,40,42,51,57]:
        for s in [1,2,4,8]: yield t,s,1


def relative_rmse(out, reference):
    assert torch.isfinite(out).all()
    return ((out.float()-reference.float()).square().sum()/reference.float().square().sum()).sqrt().item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('mxfp8_kernel_bench'))
    parser.add_argument('--reference-bench', type=Path, default=Path('gemm_bench.py'))
    parser.add_argument('--out', type=Path, default=Path('benchmarks/results/mxfp8_decode'))
    parser.add_argument('--candidate-results', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--reuse-baselines', type=Path)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--tune-sms', action='store_true')
    parser.add_argument('--candidate-count', type=int, default=12)
    parser.add_argument('--tests', type=int, default=30)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--tune-tests', type=int, default=5)
    parser.add_argument('--bs', type=int, nargs='+')
    parser.add_argument('--nk', type=int, nargs=2)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    assert args.tests > 0 and args.trials > 0 and args.tune_tests > 0
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.baseline))
    from bench_fp8 import block_quant
    from humming_backend import HummingMXFP8
    from vllm import _custom_ops as ops
    bench = import_file('mxfp8_reference_bench', args.reference_bench)
    raw_bytes = (args.baseline/'results_humming/raw.jsonl').read_bytes()
    old = [json.loads(l) for l in raw_bytes.splitlines()]
    manifest = json.loads((args.baseline/'results_humming/filtered/filter_manifest.json').read_text())
    assert hashlib.sha256(raw_bytes).hexdigest() == manifest['original_raw_sha256']
    excluded = {(r['m'],r['n'],r['k']) for r in manifest['excluded_points']}
    keys = sorted({(r['n'],r['k']) for r in old})[args.shard::args.shards]
    if args.nk: keys = [tuple(args.nk)]
    batches = args.bs or sorted({r['m'] for r in old})
    prior = {}
    if args.candidate_results:
        for file in args.candidate_results.glob('raw_*.jsonl'):
            for line in file.read_text().splitlines():
                row = json.loads(line)
                prior[row['m'],row['n'],row['k']] = row['candidates']
    config = json.loads(args.config.read_text())['configs'] if args.config else {}
    reused={}
    reuse_hashes={}
    if args.reuse_baselines:
        previous_meta=json.loads((args.reuse_baselines/f'metadata_{args.shard}.json').read_text())
        assert previous_meta['visible_devices']==os.getenv('CUDA_VISIBLE_DEVICES'), 'Baseline must come from the same physical GPU'
        assert previous_meta['reference_bench_sha256']==hashlib.sha256(args.reference_bench.read_bytes()).hexdigest()
        for name in ['seed','bs','nk','shard','shards']:
            assert previous_meta['args'][name]==getattr(args,name), f'Input sequence differs: {name}'
        for file in args.reuse_baselines.glob('raw_*.jsonl'):
            reuse_hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            for line in file.read_text().splitlines():
                row=json.loads(line)
                reused[row['m'],row['n'],row['k']]=row['backends']
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    load_library()
    sources = list((ROOT/'csrc').glob('mxfp8*.cu')) + list((ROOT/'csrc/include/mxfp8_gemm').glob('*.hpp')) + [ROOT/'python/mxfp6/mxfp8.py']
    metadata = dict(reused_baseline_files=reuse_hashes,method='gemm_bench.bench_kineto on a pre-captured single-GEMM CUDA graph; 8e9-byte L2 flush before each replay, excluded from timing; one GPU GEMM kernel verified per replay',
                    gpu=str(torch.cuda.get_device_properties(0)), visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                    torch=torch.__version__, reference_bench_sha256=hashlib.sha256(args.reference_bench.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    baseline_manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                    baseline_policy='remeasure Humming/default heuristics, FlashInfer/exact-M autotune and vLLM on same device with same profiler logic; preserve original outlier list',
                    args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.out/f'metadata_{args.shard}.json').write_text(json.dumps(metadata,indent=2))
    with (args.out/f'raw_{args.shard}.jsonl').open('w',buffering=1) as log:
        for n,k in keys:
            w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
            b,sb = flashinfer.mxfp8_quantize(w,is_sf_swizzled_layout=True)
            bw,bs = block_quant(w,True)
            humming = HummingMXFP8(b,sb)
            for m in batches:
                x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                a,sa = flashinfer.mxfp8_quantize(x,is_sf_swizzled_layout=True)
                ba,bas = block_quant(x)
                original = x.float()@w.float().T
                hf,htuning = humming.prepare(a,sa)
                fi = lambda:flashinfer.mm_mxfp8(a,b.T,sa,sb,out_dtype=torch.bfloat16,backend='cutlass')
                buckets=tuple(batches)
                with flashinfer.autotune(True,tuning_buckets=buckets): fi()
                with flashinfer.autotune(False,tuning_buckets=buckets):
                    quant_ref = fi()
                    configs = [tuple(config.get(f'{m},{n},{k}',select_config(m,n,k)))]
                    if args.tune:
                        if (m,n,k) in prior:
                            previous=prior[m,n,k]
                            # Include candidates selected by both previous cache regimes.
                            ranked=sorted(previous,key=lambda r:r.get('cold_estimate_us',float('inf')))[:args.candidate_count//2]
                            ranked+=sorted(previous,key=lambda r:r.get('decode_us',r.get('hot_us',r.get('graph_us',float('inf')))))
                            configs=list(dict.fromkeys((r['tactic'],r['splits'],r['swizzle'],r.get('sms',0)) for r in ranked))[:args.candidate_count]
                        else: configs=list(candidates(m))
                    configs=[tuple((list(c)+[0])[:4]) for c in configs]
                    if args.tune and m>=20:
                        configs += [(t,1,1,0) for t in [52,53,54,55,56]]
                        configs += [(51,s,1,0) for s in [1,2,4]]
                        if m>=128: configs += [(57,s,1,0) for s in [1,2,4]]
                    if args.tune_sms:
                        stream_tactics=[12,13] if m<=16 else [14,15] if m<=128 else [16,40]
                        configs += [(t,1,1,sms) for t in stream_tactics for sms in [64,86,128,144]]
                    configs=list(dict.fromkeys(configs))
                    tried=[]
                    for tactic,splits,swizzle,sms in configs:
                        fn=prepare(a,b,sa,sb,tactic=tactic,splits=splits,swizzle=swizzle,sms=sms)
                        error=relative_rmse(fn(),quant_ref)
                        assert error < .005,(m,n,k,tactic,error)
                        graph,out=capture(fn)
                        tag,name=inspect_graph(graph)
                        samples=measure(bench,graph,tag,args.tune_tests,1)
                        tried.append(dict(tactic=tactic,splits=splits,swizzle=swizzle,sms=sms,decode_us=statistics.median(samples),quantized_rmse=error))
                        del graph,fn,out
                    best=min(tried,key=lambda r:r['decode_us'])
                    native=prepare(a,b,sa,sb,tactic=best['tactic'],splits=best['splits'],swizzle=best['swizzle'],sms=best['sms'])
                    fns={'native':native,'humming_mxfp8':hf,'flashinfer_mxfp8':fi,
                         'vllm_block_fp8':lambda:ops.cutlass_scaled_mm(ba,bw.T,bas,bs.T,torch.bfloat16)}
                    measured={name:data for name,data in reused.get((m,n,k),{}).items() if name!='native'}
                    order=[name for name in fns if name not in measured];shift=batches.index(m)%len(order);order=order[shift:]+order[:shift]
                    for backend in order:
                        fn=fns[backend]
                        err=relative_rmse(fn(),original)
                        assert err < .08,(backend,err)
                        graph,out=capture(fn)
                        tag,name=inspect_graph(graph)
                        samples=measure(bench,graph,tag,args.tests,args.trials)
                        measured[backend]=dict(us=statistics.median(samples),samples_us=samples,kernel_name=name,relative_rmse=err)
                        del graph,out
                    outlier=(m,n,k) in excluded
                    reference_backend=min(['vllm_block_fp8','flashinfer_mxfp8'],key=lambda key:measured[key]['us']) if outlier else 'humming_mxfp8'
                    base=measured[reference_backend]['us'];threshold=1.1 if outlier else 1.05
                    elapsed=measured['native']['us']
                    row=dict(m=m,n=n,k=k,tactic=best['tactic'],splits=best['splits'],swizzle=best['swizzle'],sms=best['sms'],
                             graph_us=elapsed,graph_samples_us=measured['native']['samples_us'],
                             relative_rmse=measured['native']['relative_rmse'],quantized_rmse=best['quantized_rmse'],
                             candidates=tried,backends=measured,humming_tuning=htuning,
                             outlier=outlier,baseline_backend=reference_backend,baseline_graph_us=base,
                             required_speedup=threshold,target_us=base/threshold,speedup=base/elapsed)
                    row['pass']=row['speedup']>threshold
                    log.write(json.dumps(row)+'\n')
                    print(f'{m=} {n=} {k=} tactic={best["tactic"]}/{best["splits"]}/sw{best["swizzle"]} native={elapsed:.3f}us reference={reference_backend}:{base:.3f}us speedup={row["speedup"]:.3f} target={threshold:.2f} pass={row["pass"]}',flush=True)

if __name__=='__main__': main()
