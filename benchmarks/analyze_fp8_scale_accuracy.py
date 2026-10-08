#!/usr/bin/env python3
"""Controlled activation-scale ablations and cross-weight GEMM decomposition."""
import json
from pathlib import Path
from contextlib import ExitStack
from collections import defaultdict
import statistics
import torch
import flashinfer
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/qwen35_2b_fp8_accuracy'


def rel(x,y):
    return ((x.double()-y.double()).square().sum()/y.double().square().sum()).sqrt().item()


def quant(x,group,pow2=False):
    v=x.float().reshape(x.shape[0],-1,group)
    scale=v.abs().amax(-1).clamp_min(1e-4)/448
    if pow2:scale=torch.exp2(torch.ceil(torch.log2(scale)))
    q=(v/scale[...,None]).to(torch.float8_e4m3fn)
    return (q.float()*scale[...,None]).reshape_as(x),scale


def actual_mx(x):
    m,k=x.shape
    q,s=flashinfer.mxfp8_quantize(x,is_sf_swizzled_layout=True)
    ls=s.reshape(-1,k//128,32,4,4).transpose(1,3).contiguous().reshape(-1,k//32)[:m]
    scale=torch.exp2(ls.float()-127)
    return (q.float().reshape(m,k//32,32)*scale[...,None]).reshape(m,k),scale


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    seeds=[20260929,20260930,20260931];m=512
    cache={};acts=[]
    for k in [2048,6144]:
        for seed in seeds:
            torch.manual_seed(seed+k*17+m)
            x=torch.randn(m,k,device='cuda').bfloat16()
            variants={};scales={}
            for g in [32,128]:
                for power in [False,True]:
                    name=f'g{g}_'+('pow2' if power else 'float')
                    variants[name],scales[name]=quant(x,g,power)
            variants['actual_mx'],scales['actual_mx']=actual_mx(x)
            stats={}
            for name,v in variants.items():
                group=128 if name.startswith('g128') else 32
                xx=x.float().reshape(m,-1,group);yy=v.reshape_as(xx)
                imax=xx.abs().argmax(-1,keepdim=True)
                maxima_x=xx.gather(-1,imax);maxima_y=yy.gather(-1,imax)
                e2=(yy-xx).square();max_e=e2.gather(-1,imax).sum()
                stats[name]=dict(relative_rmse=rel(v,x),
                    maxima_relative_rmse=rel(maxima_y,maxima_x),
                    maxima_fraction_of_input_energy=(maxima_x.square().sum()/xx.square().sum()).item(),
                    maxima_fraction_of_error_energy=(max_e/e2.sum()).item(),
                    zero_fraction=(v==0).float().mean().item())
            stats['pow2_32_vs_128']=dict(relative_rmse=rel(variants['g32_pow2'],variants['g128_pow2']),
                unequal_fraction=(variants['g32_pow2']!=variants['g128_pow2']).float().mean().item())
            stats['actual_vs_manual']=dict(relative_rmse=rel(variants['actual_mx'],variants['g32_pow2']),
                unequal_fraction=(variants['actual_mx']!=variants['g32_pow2']).float().mean().item(),
                unequal_scale_fraction=(scales['actual_mx']!=scales['g32_pow2']).float().mean().item())
            assert stats['actual_vs_manual']['relative_rmse']<1e-6
            acts.append(dict(k=k,seed=seed,metrics=stats));cache[k,seed]=(x,variants)
    data=Path('<LOCAL_PATH>')
    paths=[data/'Qwen3.5-2B/model.safetensors-00001-of-00001.safetensors',
           data/'Qwen3.5-2B-FP8_BLOCK/model.safetensors',data/'Qwen3.5-2B-MXFP8/model.safetensors']
    rows=[]
    with ExitStack() as stack:
        base,block,mx=[stack.enter_context(safe_open(str(p),framework='pt')) for p in paths]
        keys=sorted(k for k in mx.keys() if k.endswith('.weight_scale'))
        for i,sk in enumerate(keys):
            wk=sk.replace('.weight_scale','.weight');w=base.get_tensor(wk).cuda().float();n,k=w.shape
            bw=block.get_tensor(wk).cuda().float();bs=block.get_tensor(sk).cuda().float()
            bw*=bs.repeat_interleave(128,0).repeat_interleave(128,1)
            mw=mx.get_tensor(wk).cuda().float();ms=mx.get_tensor(sk).cuda()
            mw=(mw.reshape(n,k//32,32)*torch.exp2(ms.float()-127)[...,None]).reshape(n,k)
            for seed in seeds:
                x,vs=cache[k,seed];xf=x.float();ref=xf@w.T;stats={}
                for name,act in vs.items():
                    if name=='g32_pow2':continue # identical to actual_mx, asserted above
                    stats['activation_only/'+name]=rel(act@w.T,ref)
                    for wn,ww in [('block',bw),('mx',mw)]:
                        if name=='g128_pow2':continue
                        stats[wn+'_weight/'+name]=rel(act@ww.T,ref)
                for wn,ww in [('block',bw),('mx',mw)]:
                    stats['weight_only/'+wn]=rel(xf@ww.T,ref)
                # Decompose product error into activation, weight, and cross term.
                for wn,ww,act in [('block',bw,vs['g128_float']),('mx',mw,vs['actual_mx'])]:
                    ea=(act-xf)@w.T;ew=xf@(ww-w).T;ec=(act-xf)@(ww-w).T
                    energy=ref.square().sum()
                    stats[wn+'/cross_term_rel']=(ec.square().sum()/energy).sqrt().item()
                    stats[wn+'/activation_weight_error_cosine']=(torch.sum(ea*ew)/(ea.norm()*ew.norm())).item()
                rows.append(dict(weight=wk,seed=seed,metrics=stats))
    groups=defaultdict(list)
    for r in rows:
        for name,v in r['metrics'].items():groups[name].append(v)
    result=dict(method='M=512, 3 seeds, all 150 checkpoint weights; decoded FP32 GEMM, no output cast, TF32 disabled. Hypothetical g32_float is an ablation, not MXFP8.',
                activation_tests=acts,gemm_summary={k:dict(mean=statistics.mean(v),min=min(v),max=max(v)) for k,v in groups.items()},gemm_rows=rows)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'scale_ablation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ACTIVATIONS')
    for name in acts[0]['metrics']:
        print(name,{k:statistics.mean(a['metrics'][name][k] for a in acts) for k in acts[0]['metrics'][name]})
    print('GEMM')
    for k,v in result['gemm_summary'].items():print(k,v['mean'])

if __name__=='__main__':main()
