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
from mxfp6.mxfp8 import load_library, prepare, select_config, MXFP8Tensor, W8A8Config


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
    # Graph nodes borrow pointers to external tensors, including private workspaces.
    # Retain the prepared callable for the entire graph lifetime.
    graph._prepared_callable = fn
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
    # bench_kineto prints only the first 100 characters of kernel names.
    # Use a stable namespace prefix that remains visible in that table.
    if 'cutlass' in names[0]:
        return 'cutlass', names[0]
    import re
    match = re.search(r'(mxfp8_[A-Za-z0-9_]+)::', names[0])
    if match:
        return match.group(1) + '::', names[0]
    raise AssertionError(f'Unrecognized GEMM kernel: {names}')


def measure(bench, graph, tag, tests, trials):
    samples = []
    for _ in range(trials):
        value = bench.bench_kineto(graph.replay, tag, num_tests=tests,
                                   suppress_kineto_output=True, flush_l2=True,
                                   with_multiple_kernels=False)
        assert value > 0, (tag, value)
        samples.append(value * 1e6)
    return samples


def relative_rmse(out, reference):
    assert torch.isfinite(out).all()
    return ((out.float()-reference.float()).square().sum()/reference.float().square().sum()).sqrt().item()


def block_quant(x, weight=False):
    m,k=x.shape
    if weight:
        values=x.float().reshape(m//128,128,k//128,128)
        scales=values.abs().amax((1,3)).clamp_min(1e-4)/448
        quant=(values/scales[:,None,:,None]).to(torch.float8_e4m3fn).reshape(m,k)
    else:
        values=x.float().reshape(m,k//128,128)
        scales=values.abs().amax(2).clamp_min(1e-4)/448
        quant=(values/scales[:,:,None]).to(torch.float8_e4m3fn).reshape(m,k)
    return quant.contiguous(), scales.contiguous() if weight else scales.T.contiguous().T


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference-bench',type=Path,default=Path('gemm_bench.py'))
    parser.add_argument('--reference-dispatch',type=Path,help='Optional frozen point table for paired policy comparison')
    parser.add_argument('--shapes',type=Path,help='JSON list of [M,N,K]; defaults to both Qwen3.5 models, 24 batches')
    parser.add_argument('--tests',type=int,default=30)
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--check-graphs',action='store_true')
    args=parser.parse_args()
    assert args.tests>0 and args.trials>0 and 0<=args.shard<args.shards
    assert os.getenv('CUDA_VISIBLE_DEVICES') in ('4','5'), 'Use physical GPU 4 or 5'
    args.out.mkdir(parents=True,exist_ok=True)
    from vllm import _custom_ops as ops
    geometry=json.loads(Path(__file__).with_name('qwen35_tp1_projection_geometry.json').read_text())
    nk=sorted({(p['n'],p['k']) for model in geometry['models'] for p in model['projections']})
    batches=[1,2,4,8,10,12,14,16,20,24,28,32,36,40,48,56,64,96,128,192,256,512,1024,2048]
    shapes=json.loads(args.shapes.read_text()) if args.shapes else [[m,n,k] for n,k in nk for m in batches]
    shapes=shapes[args.shard::args.shards]
    reference=json.loads(args.reference_dispatch.read_text())['configs'] if args.reference_dispatch else {}
    bench=import_file('mxfp8_reference_bench',args.reference_bench)
    torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32=False
    load_library()
    sources=[Path(__file__),ROOT/'python/mxfp6/mxfp8.py',ROOT/'python/mxfp6/mxfp8_dispatch.json']
    library=Path(os.getenv('MXFP8_LIBRARY_PATH',str(ROOT/'build/mxfp8/mxfp8_torch.so')))
    meta=dict(visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu=str(torch.cuda.get_device_properties(0)),
        method='Single-GEMM CUDA graph; 8 GB L2 eviction before replay; kernel time only',
        sampling='Interleaved trials with cyclic backend order; median of trial means',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        reference_bench_sha256=hashlib.sha256(args.reference_bench.read_bytes()).hexdigest(),
        native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    if args.reference_dispatch:
        meta['reference_dispatch_sha256']=hashlib.sha256(args.reference_dispatch.read_bytes()).hexdigest()
    (args.out/f'metadata_{args.shard}.json').write_text(json.dumps(meta,indent=2)+'\n')
    with (args.out/f'raw_{args.shard}.jsonl').open('w',buffering=1) as log:
        for m,n,k in shapes:
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
            a,sa=flashinfer.mxfp8_quantize(x,is_sf_swizzled_layout=True)
            b,sb=flashinfer.mxfp8_quantize(w,is_sf_swizzled_layout=True)
            ba,bas=block_quant(x);bw,bs=block_quant(w,True)
            original=x.float()@w.float().T
            fi=lambda:flashinfer.mm_mxfp8(a,b.T,sa,sb,out_dtype=torch.bfloat16,backend='cutlass')
            with flashinfer.autotune(True,tuning_buckets=(m,)):fi()
            with flashinfer.autotune(False,tuning_buckets=(m,)):
                quant_ref=fi().clone()
                qa=MXFP8Tensor(a.view(torch.uint8),sa,m,k)
                qb=MXFP8Tensor(b.view(torch.uint8),sb,n,k)
                native=prepare(qa,qb)
                config=list(native.config)
                quant_error=relative_rmse(native(),quant_ref)
                assert quant_error<.005,(m,n,k,config,quant_error)
                fns=dict(native=native,flashinfer=fi,
                    block_fp8=lambda:ops.cutlass_scaled_mm(ba,bw.T,bas,bs.T,torch.bfloat16))
                key=f'{m},{n},{k}'
                if reference:
                    t,s,sw,sms=reference[key]
                    fns['reference']=prepare(qa,qb,config=W8A8Config(t,s,sw,sms))
                graphs={}
                for name,fn in fns.items():
                    error=relative_rmse(fn(),original)
                    assert error<.08,(name,m,n,k,error)
                    graph,out=capture(fn);tag,kernel=inspect_graph(graph)
                    graphs[name]=(graph,out,tag,kernel,error)
                samples={name:[] for name in graphs}
                names=list(graphs)
                for trial in range(args.trials):
                    shift=(trial+m)%len(names)
                    for name in names[shift:]+names[:shift]:
                        graph,_,tag,_,_=graphs[name]
                        samples[name]+=measure(bench,graph,tag,args.tests,1)
                measured={name:dict(us=statistics.median(samples[name]),samples_us=samples[name],kernel_name=data[3],relative_rmse=data[4]) for name,data in graphs.items()}
                dynamic_error=0.
                if args.check_graphs:
                    graph,out,_,_,_=graphs['native']
                    scale=torch.exp2(torch.arange(k//32,device='cuda').float()%16-8).repeat_interleave(32)
                    for _ in range(2):
                        qa,qsa=flashinfer.mxfp8_quantize((torch.randn_like(x).float()*scale).to(torch.bfloat16),is_sf_swizzled_layout=True)
                        qb,qsb=flashinfer.mxfp8_quantize((torch.randn_like(w).float()/scale).to(torch.bfloat16),is_sf_swizzled_layout=True)
                        a.copy_(qa);sa.copy_(qsa);b.copy_(qb);sb.copy_(qsb)
                        graph.replay()
                        error=relative_rmse(out,fi())
                        assert error<.005,(m,n,k,config,error)
                        dynamic_error=max(dynamic_error,error)
                    a.zero_();graph.replay()
                    assert torch.count_nonzero(out).item()==0
                row=dict(m=m,n=n,k=k,config=config,backends=measured,quantized_rmse=quant_error,
                    changing_graph_rmse=dynamic_error if args.check_graphs else None)
                log.write(json.dumps(row)+'\n')
                print('RESULT',m,n,k,config,{name:round(v['us'],3) for name,v in measured.items()},flush=True)
                del graphs,graph,out,fns,native


if __name__=='__main__':main()
