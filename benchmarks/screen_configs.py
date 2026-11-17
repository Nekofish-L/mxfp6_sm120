#!/usr/bin/env python3
"""Screen all registered configurations on real model shapes without changing dispatch.

This is bench_kineto evidence for a bounded shape/cache set, not a proof that a
configuration can be deleted. W8A8 coarse probes use splits/swizzle=1; W6A8
coarse probes cover every swizzle. Finalists and the builtin are remeasured.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('w6a8','w8a8'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',default='1,16,32,64,128,512')
    p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--flush-l2-mb',type=int,choices=(0,256),default=0)
    p.add_argument('--kineto-source',type=Path)
    args=p.parse_args()
    os.environ['MXFP6_AUTOTUNE']='off'
    sys.path.insert(0,str(ROOT/'python'))
    import torch
    import mxfp6
    import mxfp6.mxfp8 as fp8
    from mxfp6 import autotune as tuning
    from kineto_utils import load_bench_kineto, measure_kineto
    bench_kineto, kineto_source = load_bench_kineto(args.kineto_source)
    from model_shapes import QWEN35_27B_TP2_NK
    from audit_dispatch import geometry
    batches=[int(v) for v in args.batches.split(',')]
    nk=geometry() if args.backend=='w8a8' else [(n,k) for _,n,k in QWEN35_27B_TP2_NK]
    ids=fp8.available_configs() if args.backend=='w8a8' else list(range(30))
    stream_ids=set(fp8.available_configs(stream_k_only=True)) if args.backend=='w8a8' else {3,16}
    dtypes=(torch.bfloat16,) if args.backend=='w8a8' else (torch.float16,torch.bfloat16)
    flush = bool(args.flush_l2_mb)
    results=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(backend=args.backend,device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                  batches=batches,nk=nk,registered=len(ids),iterations=args.iterations,
                  regime='explicit-flush CUDA Graph' if flush else 'warm-cache CUDA Graph',
                  timer='bench_kineto',kineto_source=str(kineto_source),
                  flush_l2_bytes=256000000 if flush else 0,
                  flush_l2_mb=args.flush_l2_mb,
                  search='randomized coarse scan; top 8 IDs plus builtin refined; no production policy changes',
                  limitation='Not exhaustive over shapes, cache regimes, PDL, resource limits or W8A8 launch parameters; no deletion eligibility is implied.')
    for n,k in nk:
      for m in batches:
       for dtype in dtypes:
        torch.manual_seed(m*100003+n*101+k)
        a=mxfp6.quantize_activation(torch.randn(m,k,device='cuda',dtype=torch.bfloat16))
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        b=fp8.quantize_mxfp8(w) if args.backend=='w8a8' else mxfp6.quantize_mxfp6(w)
        del w
        def make_run(config):
            if args.backend=='w8a8':
                return fp8.prepare(a,b,config=None if config is None else fp8.W8A8Config(*config))
            cfg=tuning.W6A8Config(-1,1,0) if config is None else tuning.W6A8Config(*config)
            return lambda:tuning._run_config(cfg,a.values,b.values,a.scales,b.scales,m,n,k,dtype)
        def measure(config,validate=True):
            run=make_run(config)
            actual=run()
            if validate:
                diff=(actual.float()-reference).square().sum()
                error=(diff/reference_norm).sqrt().item()
                if not math.isfinite(error) or error>.005:
                    raise ValueError(f'output relative RMSE {error}')
            return measure_kineto(run,num_tests=args.iterations,flush_l2=flush,bench_kineto=bench_kineto)
        default=make_run(None)
        reference=default().float().clone()
        reference_norm=reference.square().sum().clamp_min(1e-20)
        builtin_config=list(fp8.select_config(m,n,k)) if args.backend=='w8a8' else None
        configs=([(i,1,1,0) for i in ids] if args.backend=='w8a8' else [(i,s,0) for i in ids for s in (1,2,4,8)])
        random.Random(m*100003+n*101+k).shuffle(configs)
        coarse=[]; rejected=[]
        for config in configs:
            try:
                us=measure(config)
                coarse.append(dict(config=list(config),us=us))
            except (RuntimeError,ValueError) as error:
                message=str(error)
                if any(s in message.lower() for s in ('illegal memory','device-side assert','launch failure')):
                    raise
                rejected.append(dict(config=list(config),error=message[:300]))
        best={}
        for row in sorted(coarse,key=lambda r:r['us']):
            best.setdefault(row['config'][0],row)
        finalists=[]
        for i,row in list(best.items())[:8]:
            if args.backend=='w8a8':
                finalists.extend((i,splits,swizzle,0) for splits in ((1,2,4) if i in stream_ids else (1,)) for swizzle in (1,2,4,8))
            else:
                finalists.extend((i,swizzle,raster) for swizzle in (1,2,4,8) for raster in (0,1,2))
        finalists.append(None)
        random.Random(k*100003+n*101+m).shuffle(finalists)
        refined=[]
        for config in finalists:
            try:
                samples=[measure(config) for _ in range(3)]
                refined.append(dict(config=list(config) if config is not None else None,samples_us=samples,us=statistics.median(samples)))
            except (RuntimeError,ValueError) as error:
                if any(s in str(error).lower() for s in ('illegal memory','device-side assert','launch failure')):raise
                rejected.append(dict(config=config,error=str(error)[:300]))
        refined.sort(key=lambda r:r['us'])
        default_row=next(row for row in refined if row['config'] is None)
        winner=refined[0]
        results.append(dict(shape=[m,n,k],dtype=str(dtype),builtin_config=builtin_config,
                            builtin_us=default_row['us'],winner=winner,
                            near_best=[row for row in refined if row['us']<=winner['us']*1.02],
                            coarse=coarse,refined=refined,rejected=rejected))
        args.output.write_text(json.dumps(dict(metadata=metadata,results=results),indent=2)+'\n')
        print(args.backend,[m,n,k],str(dtype),'best',winner['config'],round(winner['us'],3),'builtin',round(default_row['us'],3),flush=True)
        del a,b,reference,reference_norm,default
    counts=Counter(tuple(row['winner']['config']) if row['winner']['config'] is not None else ('builtin',) for row in results)
    print('winner configuration count:',len(counts),flush=True)


if __name__=='__main__':
    main()
