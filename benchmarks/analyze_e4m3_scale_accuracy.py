#!/usr/bin/env python3
"""Simulate E4M3 scales; not a hardware MXFP8 format or kernel benchmark."""
import json
from pathlib import Path
import statistics
import torch
from analyze_fp8_scale_accuracy import quant,rel

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rows=[]
    for k in [2048,6144]:
        for seed in [20260929,20260930,20260931]:
            torch.manual_seed(seed+k*17+512)
            x=torch.randn(512,k,device='cuda').bfloat16()
            for group in [32,128]:
                v=x.float().reshape(512,-1,group)
                ideal=v.abs().amax(-1)/448
                schemes={'fp32':ideal,'fp16':ideal.half().float(),'bf16':ideal.bfloat16().float(),
                         'e8m0':torch.exp2(torch.ceil(torch.log2(ideal)))}
                for normalized in [False,True]:
                    global_s=ideal.max()/448 if normalized else torch.ones((),device='cuda')
                    s=ideal/global_s
                    q=s.to(torch.float8_e4m3fn)
                    decoded=q.float()
                    # All scales positive and finite in these inputs; positive E4M3 codes are monotone.
                    codes=q.view(torch.uint8)+(decoded<s).to(torch.uint8)
                    up=codes.view(torch.float8_e4m3fn).float()
                    prefix='normalized_' if normalized else 'raw_'
                    schemes[prefix+'e4m3_nearest']=decoded*global_s
                    schemes[prefix+'e4m3_up']=up*global_s
                for name,s in schemes.items():
                    assert (s>0).all() and torch.isfinite(s).all()
                    unrounded=v/s[...,None]
                    q=unrounded.clamp(-448,448).to(torch.float8_e4m3fn)
                    dequant=q.float()*s[...,None]
                    rows.append(dict(k=k,seed=seed,group=group,scheme=name,
                        relative_rmse=rel(dequant.reshape_as(x),x),
                        outside_range_fraction=(unrounded.abs()>448).float().mean().item(),
                        scale_relative_rmse=rel(s,ideal),scale_min=s.min().item(),scale_max=s.max().item()))
    summary={}
    for group in [32,128]:
        summary[group]={}
        for name in schemes:
            rr=[r for r in rows if r['group']==group and r['scheme']==name]
            summary[group][name]={key:statistics.mean(r[key] for r in rr)
                                 for key in ['relative_rmse','outside_range_fraction','scale_relative_rmse']}
    result=dict(method='Same M=512, K=2048/6144, 3 seeds, BF16 activations as earlier. Simulated quantize/dequantize with saturation to +/-448. Recompute payload using represented scale. Normalized variants use FP32 global_scale=max(ideal_scale)/448.',summary=summary,rows=rows)
    out=Path(__file__).resolve().parents[1]/'benchmarks/results/qwen35_2b_fp8_accuracy/e4m3_scale_ablation.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
