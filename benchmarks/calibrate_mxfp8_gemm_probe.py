#!/usr/bin/env python3
"""Small held-out synthetic calibration probe; leaves GEMM kernel unchanged."""
import json
from pathlib import Path
import sys
import statistics
import torch
import flashinfer
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from mxfp6.mxfp8 import MXFP8Tensor,gemm_w8a8
from analyze_fp8_scale_accuracy import actual_mx,rel


def q(x):return actual_mx(x)[0]

@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
    out=ROOT/'benchmarks/results/qwen35_2b_fp8_accuracy/calibration_probe.json'
    rows=[]
    with safe_open('<LOCAL_PATH>',framework='pt') as base, safe_open('<LOCAL_PATH>',framework='pt') as mx:
        selected={}
        for sk in sorted(mx.keys()):
            if sk.endswith('.weight_scale'):
                wk=sk.replace('.weight_scale','.weight')
                kind=wk.split('.layers.')[1].split('.',1)[1]
                selected.setdefault(kind,wk)
        for kind,wk in selected.items():
            w=base.get_tensor(wk).cuda();n,k=w.shape
            torch.manual_seed(20261001+k)
            xc=torch.randn(128,k,device='cuda').bfloat16()
            torch.manual_seed(20261002+k)
            xt=torch.randn(512,k,device='cuda').bfloat16()
            rc=xc.float()@w.float().T;rt=xt.float()@w.float().T
            wc=q(w);baseline_c=rel(q(xc)@wc.T,rc);baseline_t=rel((q(xt)@wc.T).bfloat16(),rt)
            candidates=[('identity',torch.ones(k,device='cuda'))]
            # Non-power-of-two paired scalar transformations.
            for scalar in [1.125,1.25,1.5,1.75]:
                candidates.append((f'scalar_{scalar}',torch.full((k,),scalar,device='cuda')))
            act_max=xc.float().abs().amax(0).clamp_min(1e-8)
            wt_max=w.float().abs().amax(0).clamp_min(1e-8)
            for alpha in [0,.25,.5,.75,1]:
                d=act_max.pow(alpha)/wt_max.pow(1-alpha)
                d/=d.log().mean().exp()
                candidates.append((f'smooth_alpha_{alpha}',d))
            best=(baseline_c,'identity',candidates[0][1]);scores={}
            for name,d in candidates:
                qw=q((w.float()*d).bfloat16())
                pred=q((xc.float()/d).bfloat16())@qw.T
                score=rel(pred,rc);scores[name]=score
                if score<best[0]:best=(score,name,d)
            score,name,d=best
            ta=(xt.float()/d).bfloat16();tw=(w.float()*d).bfloat16()
            qa,sa=flashinfer.mxfp8_quantize(ta,is_sf_swizzled_layout=True)
            qb,sb=flashinfer.mxfp8_quantize(tw,is_sf_swizzled_layout=True)
            y=gemm_w8a8(MXFP8Tensor(qa.view(torch.uint8),sa,512,k),MXFP8Tensor(qb.view(torch.uint8),sb,n,k))
            # Compare the actual original checkpoint on the same held-out activations.
            from compare_checkpoint_fp8_accuracy import pack_scale
            orig_w=mx.get_tensor(wk).cuda();orig_s=mx.get_tensor(wk.replace('.weight','.weight_scale')).cuda()
            orig_a,orig_as=flashinfer.mxfp8_quantize(xt,is_sf_swizzled_layout=True)
            orig_y=gemm_w8a8(MXFP8Tensor(orig_a.view(torch.uint8),orig_as,512,k),MXFP8Tensor(orig_w.view(torch.uint8),pack_scale(orig_s),n,k))
            # Local E8M0 exponent search: no rebalancing, optimize activation MSE per group.
            v=xt.float().reshape(512,-1,32)
            s0=torch.exp2(torch.ceil(torch.log2(v.abs().amax(-1).clamp_min(1e-4)/448)))
            decoded=[];errors=[]
            for off in [-2,-1,0,1]:
                s=s0*(2.**off);z=(v/s[...,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()*s[...,None]
                decoded.append(z);errors.append((z-v).square().sum(-1))
            choice=torch.stack(errors).argmin(0)
            optimized=torch.stack(decoded).gather(0,choice[None,...,None].expand(1,*v.shape))[0]
            row=dict(weight=wk,best=name,calibration_scores=scores,baseline_calibration=baseline_c,
                baseline_requant_heldout=baseline_t,checkpoint_heldout=rel(orig_y,rt),best_heldout=rel(y,rt),
                kernel_vs_decoded=rel(y,(q(ta)@q(tw).T).bfloat16()),
                exponent_search_activation_before=rel(q(xt),xt),exponent_search_activation_after=rel(optimized.reshape_as(xt),xt))
            rows.append(row);print(json.dumps(row),flush=True)
    summary={k:statistics.mean(r[k] for r in rows) for k in ['baseline_requant_heldout','checkpoint_heldout','best_heldout','exponent_search_activation_before','exponent_search_activation_after']}
    out.write_text(json.dumps(dict(method='10 projection types, first matching weight each; 128 synthetic BF16 calibration rows, independent 512 test rows; 10 scaling candidates chosen only on calibration. Native unchanged MXFP8 GEMM verification. Exploratory, not real-activation calibration.',summary=summary,rows=rows),indent=2)+'\n')
    print('SUMMARY',summary)

if __name__=='__main__':main()
