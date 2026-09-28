#!/usr/bin/env python3
"""Paired cold-weight graph measurements for wide-N, short-K MXFP8."""
import argparse,hashlib,json,os,statistics,sys
from pathlib import Path
import torch
import flashinfer
from benchmark_mxfp8 import capture,import_file,relative_rmse,measure,inspect_graph
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),'mxfp8_kernel_bench']
from mxfp6.mxfp8 import prepare
from bench_fp8 import block_quant
from humming_backend import HummingMXFP8
from vllm import _custom_ops as ops
SHAPES=[(5120,2048),(8192,2048),(10240,2560),(12288,2048),(12288,2560),(18432,2560)]
BATCHES=[1,2,4,8,10,12,14,16,20,24,28,32,36,40,48,56,64,96,128,192,256,512,1024,2048]
TRITON_CONFIGS=[(64,64,256,4,3,1,True),(64,64,256,4,3,1,False),
 (16,128,256,4,2,1,False),(128,16,128,4,3,1,True),
 (64,128,256,4,3,1,False),(128,128,128,4,3,1,False)]

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--phase',choices=['search','validate'],default='search');p.add_argument('--plan',type=Path)
 p.add_argument('--old',type=Path,default=ROOT/'benchmarks/results/mxfp8_decode_validation/dispatch.json')
 p.add_argument('--out',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
 p.add_argument('--bs',nargs='+',type=int);p.add_argument('--tests',type=int,default=30);p.add_argument('--trials',type=int,default=3)
 args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 old=json.loads(args.old.read_text())['configs'];plan=json.loads(args.plan.read_text())['configs'] if args.plan else {}
 bench_path=Path('gemm_bench.py');bench=import_file('reference_bench',bench_path)
 meta=dict(gpu=str(torch.cuda.get_device_properties(0)),visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
   method='gemm_bench.bench_kineto; single GEMM graph replay after 8GB L2 flush; kernel time only; interleaved trials',
   args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
   reference_sha256=hashlib.sha256(bench_path.read_bytes()).hexdigest(),old_dispatch_sha256=hashlib.sha256(args.old.read_bytes()).hexdigest(),
   source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
    [Path(__file__),ROOT/'python/mxfp6/_mxfp8_triton.py',ROOT/'python/mxfp6/mxfp8.py',ROOT/'csrc/mxfp8_direct.cu',ROOT/'csrc/include/mxfp8_gemm/launch.hpp']})
 (args.out/f'metadata_{args.shard}.json').write_text(json.dumps(meta,indent=2)+'\n')
 torch.manual_seed(20260928)
 batches=args.bs or BATCHES
 with (args.out/f'raw_{args.shard}.jsonl').open('w',buffering=1) as logfile:
  for n,k in SHAPES[args.shard::args.shards]:
   weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
   b,sb=flashinfer.mxfp8_quantize(weight,is_sf_swizzled_layout=True);bw,bs=block_quant(weight,True)
   humming=HummingMXFP8(b,sb)
   for m in batches:
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    a,sa=flashinfer.mxfp8_quantize(x,is_sf_swizzled_layout=True);ba,bas=block_quant(x)
    ref=x.float()@weight.float().T;key=f'{m},{n},{k}'
    def make_fn(spec):
     config=spec['config']
     if spec['kind']=='cutlass':
      return prepare(a,b,sa,sb,tactic=config[0],splits=config[1],swizzle=config[2],sms=config[3])
     return prepare(a,b,sa,sb,tactic=100+TRITON_CONFIGS.index(tuple(config)))
    previous=dict(kind='cutlass',config=old[key]);old_fn=make_fn(previous);quant_ref=old_fn().clone()
    def captured(fn,kind):
     y=fn();err=relative_rmse(y,quant_ref if kind in ['cutlass','triton'] else ref)
     assert err < (.005 if kind in ['cutlass','triton'] else .08),err
     graph,out=capture(fn)
     if kind=='triton':
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
       graph.replay();torch.cuda.synchronize()
      names=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
      assert len(names)==1 and '_mm' in names[0],names
      tag='_mm'
     else:tag,_=inspect_graph(graph)
     return graph,out,tag,err
    searched=[];errors=[]
    if args.phase=='search':
     specs=[previous]
     if m<=256:
      tactics=([58,59,64] if m<=16 else [60,61,62,65] if m<=64 else [62,63])
      specs += [dict(kind='cutlass',config=[t,1,sw,0]) for t in tactics for sw in [1,4]]
     ids=[0,2,3] if m<=32 else [0,1,4] if m<=128 else [1,4,5]
     specs += [dict(kind='triton',config=TRITON_CONFIGS[i]) for i in ids]
     for spec in specs:
      try:
       fn=make_fn(spec);graph,y,tag,err=captured(fn,spec['kind'])
       us=measure(bench,graph,tag,5,1)[0]
       searched.append(dict(**spec,us=us,error=err))
       del graph,y,fn
      except Exception as e:
       errors.append(dict(**spec,error=str(e)[:500]));print('SKIP',key,spec,str(e)[:100],flush=True)
     best=min(searched,key=lambda r:r['us']);selected=dict(kind=best['kind'],config=best['config'])
    else:selected=plan[key]
    new_fn=make_fn(selected)
    fns={'old':(old_fn,'cutlass'),'new':(new_fn,selected['kind'])}
    if args.phase=='validate':
     fi=lambda:flashinfer.mm_mxfp8(a,b.T,sa,sb,out_dtype=torch.bfloat16,backend='cutlass')
     with flashinfer.autotune(True,tuning_buckets=tuple(batches)):fi()
     hf,_=humming.prepare(a,sa)
     fns.update(flashinfer=(fi,'baseline'),block_fp8=(lambda:ops.cutlass_scaled_mm(ba,bw.T,bas,bs.T,torch.bfloat16),'baseline'),humming=(hf,'baseline'))
    with flashinfer.autotune(False,tuning_buckets=tuple(batches)):
     graphs={name:captured(fn,kind) for name,(fn,kind) in fns.items()}
     samples={name:[] for name in graphs}
     names=list(graphs)
     for trial in range(args.trials):
      shift=(trial+batches.index(m))%len(names)
      for name in names[shift:]+names[:shift]:
       g,y,tag,err=graphs[name]
       samples[name]+=measure(bench,g,tag,args.tests,1)
     measured={name:dict(us=statistics.median(v),samples_us=v,error=graphs[name][3]) for name,v in samples.items()}
     del graphs
    row=dict(m=m,n=n,k=k,old=previous,selected=selected,backends=measured,candidates=searched,errors=errors,speedup=measured['old']['us']/measured['new']['us'])
    logfile.write(json.dumps(row)+'\n')
    print('RESULT',key,selected,'old',round(measured['old']['us'],3),'new',round(measured['new']['us'],3),'speedup',round(row['speedup'],3),flush=True)
if __name__=='__main__':main()
