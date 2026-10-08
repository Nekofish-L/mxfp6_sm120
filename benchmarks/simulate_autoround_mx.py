#!/usr/bin/env python3
"""GEMM-only AutoRound-like SignSGD rounding simulation, fixed MX scales.

Not the full AutoRound package: no decoder block optimization, learned clipping,
scale optimization or model-level calibration. Training uses the STE derivative
of hard two-neighbor rounding, computed explicitly, then signed gradient descent.
"""
import argparse
from contextlib import ExitStack
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import torch
import flashinfer
from safetensors import safe_open
from compare_checkpoint_fp8_accuracy import ROOT,pack_scale,decode_fp6,metrics
from mxfp6.ops import PackedMXFP6Tensor,MXFP8Tensor,gemm_w6a8,pack_fp6
from mxfp6.mxfp8 import gemm_w8a8


def err(y,r):
    return ((y.float()-r).square().sum()/r.square().sum()).sqrt().item()


def lut(bits,device):
    codes=torch.arange(127 if bits==8 else 32,device=device,dtype=torch.uint8)
    if bits==8:values=codes.view(torch.float8_e4m3fn).float()
    else:
        e=(codes.int()>>2);m=(codes.int()&3).float()
        values=torch.where(e==0,m/16,(1+m/4)*torch.exp2(e.float()-3))
    return values


def neighbors(w,logical_s,bits):
    scales=torch.exp2(logical_s.float()-127).repeat_interleave(32,1)
    table=lut(bits,w.device)
    z=(w.abs()/scales).contiguous()
    lower=(torch.bucketize(z,table,right=True)-1).clamp(0,len(table)-1)
    upper=(lower+1).clamp(max=len(table)-1)
    sign=torch.where(w<0,-1.,1.)
    lo=table[lower]*scales*sign
    step=(table[upper]-table[lower])*scales*sign
    frac=torch.where(step!=0,(w-lo)/torch.where(step!=0,step,torch.ones_like(step)),0).clamp(0,1)
    return lo,step,frac,lower,upper


def packed_weight(bits,choice,lower,upper,w,logical_s):
    codes=torch.where(choice,upper,lower).to(torch.uint8)
    codes |= (w<0).to(torch.uint8)*(128 if bits==8 else 32)
    scales=pack_scale(logical_s)
    if bits==8:return MXFP8Tensor(codes,scales,*w.shape)
    return PackedMXFP6Tensor(pack_fp6(codes),scales,*w.shape)


def native(a,b,bits):
    if bits==8:return gemm_w8a8(a,b)
    return gemm_w6a8(a,b,out_dtype=torch.bfloat16)


def inputs(k,scenario,seed,rows,device):
    g=torch.Generator(device=device).manual_seed(seed)
    if scenario=='gaussian':x=torch.randn(rows,k,generator=g,device=device)
    else:
        # Distribution parameters fixed independently of train/validation/test samples.
        gd=torch.Generator(device=device).manual_seed(91573+k)
        mixing=torch.randn(64,k,generator=gd,device=device)
        mixing/=mixing.square().sum(0).sqrt()
        latent=torch.randn(rows,64,generator=g,device=device)
        noise=torch.randn(rows,k,generator=g,device=device)
        x=(latent@mixing+.1*noise)/(1.01**.5)
    x=x.bfloat16()
    q,s=flashinfer.mxfp8_quantize(x,is_sf_swizzled_layout=True)
    logical=s.reshape(-1,k//128,32,4,4).transpose(1,3).contiguous().reshape(-1,k//32)[:rows]
    decoded=(q.float().reshape(rows,k//32,32)*torch.exp2(logical.float()-127)[...,None]).reshape(rows,k)
    return x.float(),decoded,MXFP8Tensor(q.view(torch.uint8),s,rows,k)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'benchmarks/results/qwen35_2b_autoround_sim')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--lr',type=float,default=.005)
    p.add_argument('--calibration-rows',type=int,default=1024)
    p.add_argument('--validation-rows',type=int,default=512)
    p.add_argument('--test-rows',type=int,default=512)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--scenarios',nargs='+',default=['gaussian','correlated'],choices=['gaussian','correlated'])
    p.add_argument('--bits',nargs='+',type=int,default=[6,8],choices=[6,8])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    device='cuda';cache={};rows=[];start=time.time()
    (args.out/'metadata.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        algorithm='Hard nearest-neighbor FP rounding with STE; per-weight learnable offset, SignSGD, linear LR decay, fixed checkpoint E8M0 scales. Loss=MSE(X_MXFP8 @ W_quant.T, X_BF16 @ W_BF16.T).',
        selection='Every 10 steps score full calibration and separate validation sets. Report calibration-selected and validation-selected weights separately; baseline is eligible. Test data never used for selection.',
        seeds=dict(calibration=20261101,validation=20261102,test=[20261103,20261104,20261105]),
        scenarios=dict(gaussian='iid N(0,1)',correlated='64 latent N(0,1) channels mixed into K channels, normalized columns; independent Gaussian noise sigma=0.1; shared distribution parameters, independent samples'),
        limitations='Simplified single-GEMM rounding only, not full AutoRound. 7 representative complete attention projection matrices, fake inputs, no end-to-end quality claim.'),indent=2)+'\n')
    paths=['<LOCAL_PATH>',
           '<LOCAL_PATH>',
           '<LOCAL_PATH>']
    with ExitStack() as stack, (args.out/'raw.jsonl').open('w',buffering=1) as log:
        base,six,eight=[stack.enter_context(safe_open(p,framework='pt')) for p in paths]
        common=set(k for k in six.keys() if k.endswith('.weight_scale')) & set(eight.keys())
        selected={}
        for sk in sorted(common,key=lambda k:(int(k.split('.layers.')[1].split('.')[0]),k)):
            wk=sk.replace('.weight_scale','.weight');kind=wk.split('.layers.')[1].split('.',1)[1]
            selected.setdefault(kind,wk)
        keys=list(selected.values())[:args.limit or None]
        for wk in keys:
            w=base.get_tensor(wk).to(device).float();n,k=w.shape
            for scenario in args.scenarios:
                ck=(k,scenario)
                if ck not in cache:
                    cache[ck]=[inputs(k,scenario,seed,m,device) for seed,m in
                        [(20261101,args.calibration_rows),(20261102,args.validation_rows)]+
                        [(s,args.test_rows) for s in [20261103,20261104,20261105]]]
                datasets=cache[ck]
                targets=[x@w.T for x,_,_ in datasets]
                for bits in args.bits:
                    checkpoint=six if bits==6 else eight
                    payload=checkpoint.get_tensor(wk).to(device);logical_s=checkpoint.get_tensor(wk.replace('.weight','.weight_scale')).to(device)
                    original=(PackedMXFP6Tensor(payload,pack_scale(logical_s),n,k) if bits==6 else MXFP8Tensor(payload.view(torch.uint8),pack_scale(logical_s),n,k))
                    decoded=(decode_fp6(payload,logical_s) if bits==6 else payload.float()*torch.exp2(logical_s.float()-127).repeat_interleave(32,1))
                    lo,step,frac,lower,upper=neighbors(w,logical_s,bits)
                    # Preserve actual checkpoint rounding (including ties) at initialization.
                    init=(decoded-lo).abs()>(decoded-(lo+step)).abs()
                    assert torch.allclose(lo+step*init,decoded,atol=1e-7,rtol=0)
                    offset=torch.zeros_like(w)
                    tie=(frac-.5).abs()<1e-7
                    offset[tie]=torch.where(init[tie],1e-6,-1e-6)
                    base_choice=(frac+offset)>=.5
                    assert torch.equal(base_choice,init)
                    def scores(choice):
                        qw=lo+step*choice
                        return [err(data[1]@qw.T,target) for data,target in zip(datasets[:2],targets[:2])]
                    best_train,best_val=scores(init)
                    train_choice=init.clone();val_choice=init.clone();train_step=val_step=0
                    trace=[dict(step=0,train=best_train,validation=best_val)]
                    gen=torch.Generator(device=device).manual_seed(721)
                    xq=datasets[0][1];ref=targets[0]
                    for it in range(args.steps):
                        batch=torch.randint(args.calibration_rows,(args.batch_size,),generator=gen,device=device)
                        choice=(frac+offset)>=.5
                        qw=lo+step*choice
                        residual=xq[batch]@qw.T-ref[batch]
                        # STE dW/dOffset=step: normalization of MSE does not change SignSGD direction.
                        gradient=(residual.T@xq[batch])*step
                        rate=args.lr*(1-it/args.steps)
                        offset.add_(gradient.sign(),alpha=-rate).clamp_(-.5,.5)
                        if (it+1)%10==0 or it+1==args.steps:
                            choice=(frac+offset)>=.5
                            tr,va=scores(choice);trace.append(dict(step=it+1,train=tr,validation=va))
                            if tr<best_train:best_train=tr;train_choice=choice.clone();train_step=it+1
                            if va<best_val:best_val=va;val_choice=choice.clone();val_step=it+1
                    options={'baseline':(original,decoded,init),'train_selected':(packed_weight(bits,train_choice,lower,upper,w,logical_s),lo+step*train_choice,train_choice),
                             'validation_selected':(packed_weight(bits,val_choice,lower,upper,w,logical_s),lo+step*val_choice,val_choice)}
                    results={}
                    for name,(packed,deq,choice) in options.items():
                        evaluations=[]
                        for data,target in zip(datasets,targets):
                            y=native(data[2],packed,bits)
                            expected=(data[1]@deq.T).bfloat16()
                            check=err(y,expected)
                            assert check<.001,(wk,bits,name,check)
                            evaluations.append(dict(relative_rmse=err(y,target),kernel_vs_decoded=check))
                        results[name]=dict(calibration=evaluations[0],validation=evaluations[1],tests=evaluations[2:],
                            mean_test_relative_rmse=statistics.mean(e['relative_rmse'] for e in evaluations[2:]),
                            changed_weight_fraction=(choice!=init).float().mean().item(),weight_relative_rmse=err(deq,w))
                    row=dict(weight=wk,shape=[n,k],scenario=scenario,bits=bits,steps=args.steps,train_selected_step=train_step,
                        validation_selected_step=val_step,results=results,trace=trace,elapsed=time.time()-start)
                    rows.append(row);log.write(json.dumps(row)+'\n')
                    print(f"{len(rows)} {scenario} W{bits} {wk}: test baseline={results['baseline']['mean_test_relative_rmse']:.6f} train-selected={results['train_selected']['mean_test_relative_rmse']:.6f} val-selected={results['validation_selected']['mean_test_relative_rmse']:.6f} elapsed={time.time()-start:.1f}s",flush=True)
    summary={}
    for scenario in args.scenarios:
        summary[scenario]={}
        for bits in args.bits:
            group=[r for r in rows if r['scenario']==scenario and r['bits']==bits]
            summary[scenario][str(bits)]={name:{
                'mean_calibration':statistics.mean(r['results'][name]['calibration']['relative_rmse'] for r in group),
                'mean_validation':statistics.mean(r['results'][name]['validation']['relative_rmse'] for r in group),
                'mean_test':statistics.mean(r['results'][name]['mean_test_relative_rmse'] for r in group),
                'mean_changed_weight_fraction':statistics.mean(r['results'][name]['changed_weight_fraction'] for r in group)}
                for name in ['baseline','train_selected','validation_selected']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
