#!/usr/bin/env python3
"""Real-checkpoint GEMM accuracy, matching benchmark_mxfp8.py conventions."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from mxfp6.mxfp8 import MXFP8Tensor, gemm_w8a8
from mxfp6.ops import PackedMXFP6Tensor, gemm_w6a8
from benchmark_mxfp8 import block_quant
import flashinfer
from vllm import _custom_ops as ops


def pack_scale(s):
    n, groups = s.shape
    p = torch.zeros(((n+127)//128*128, groups), dtype=torch.uint8, device=s.device)
    p[:n] = s
    return p.reshape(-1,4,32,groups//4,4).transpose(1,3).contiguous().flatten()


def decode_fp6(packed, scales):
    """Independently decode little-endian four-E3M2-in-three-byte storage."""
    n, packed_k = packed.shape
    b = packed.reshape(n, -1, 3).to(torch.int32)
    bits = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    codes = torch.stack([(bits >> shift) & 63 for shift in (0, 6, 12, 18)], -1)
    codes = codes.reshape(n, packed_k * 4 // 3)
    exponent, mantissa = (codes & 31) >> 2, codes & 3
    value = torch.where(exponent == 0, mantissa.float() / 16,
                        (1 + mantissa.float() / 4) * torch.exp2(exponent.float() - 3))
    value = torch.where((codes & 32) != 0, -value, value)
    return value * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)


def metrics(y, r):
    y, r = y.double(), r.double()
    assert torch.isfinite(y).all() and torch.isfinite(r).all()
    e = y-r
    sse, energy = e.square().sum().item(), r.square().sum().item()
    return dict(relative_rmse=(sse/energy)**.5, max_abs=e.abs().max().item(),
                cosine=(torch.sum(y*r)/(y.norm()*r.norm())).item(),
                sse=sse, reference_energy=energy, count=r.numel())


def summarize(rows):
    result = {}
    for mode in rows[0]['metrics']:
        ms = [r['metrics'][mode] for r in rows]
        vals = torch.tensor([m['relative_rmse'] for m in ms], dtype=torch.float64)
        result[mode] = dict(mean_relative_rmse=vals.mean().item(),
            pooled_relative_rmse=(sum(m['sse'] for m in ms)/sum(m['reference_energy'] for m in ms))**.5,
            p95_relative_rmse=vals.quantile(.95).item(),max_relative_rmse=vals.max().item(),
            mean_cosine=sum(m['cosine'] for m in ms)/len(ms),
            max_abs=max(m['max_abs'] for m in ms))
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True,help='Local model collection directory')
    p.add_argument('--out',type=Path,default=ROOT/'benchmarks/results/qwen35_2b_fp8_accuracy')
    p.add_argument('--batches',type=int,nargs='+',default=[1,16,128,512])
    p.add_argument('--seeds',type=int,nargs='+',default=[20260929,20260930,20260931])
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--mxfp6', action='store_true', help='Add checkpoint W6A8 on common quantized weights; use a separate --out directory')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.set_num_threads(4)
    paths=[args.models/'Qwen3.5-2B/model.safetensors-00001-of-00001.safetensors',
           args.models/'Qwen3.5-2B-FP8_BLOCK/model.safetensors',args.models/'Qwen3.5-2B-MXFP8/model.safetensors']
    if args.mxfp6:
        if args.out == ROOT/'benchmarks/results/qwen35_2b_fp8_accuracy':
            raise ValueError('Use a separate --out directory for the MXFP6 common-layer comparison')
        paths.append(args.models/'Qwen3.5-2B-MXFP6/model.safetensors-00001-of-00001.safetensors')
    meta=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),paths=list(map(str,paths)),
        batches=args.batches,seeds=args.seeds,input='randn FP32 cast BF16; identical values for all paths; FP16 inputs converted from BF16',
        reference='IEEE FP32 GEMM of original BF16 operands, TF32 disabled',
        accumulation='FP32 for all kernels; reduced precision reduction disabled for BF16/FP16',
        fp8_output='BF16',block_backend='vllm._custom_ops.cutlass_scaled_mm',
        mx_backend='repository mxfp6.mxfp8.gemm_w8a8 default dispatch',
        activation_quant='repository block_quant (group 128 FP32 scale); flashinfer.mxfp8_quantize (group 32 E8M0)',
        weights='checkpoint payload/scales unchanged; block BF16 scales promoted exactly to FP32; MX E8M0 scales packed',
        limitations='Synthetic activations, isolated GEMM only; no model-level accuracy claims. BF16 source weights provide no original FP32 precision to recover with FP16.')
    if args.mxfp6:
        meta.update(mxfp6_backend='repository mxfp6.ops.gemm_w6a8 default dispatch',
                    mxfp6_format='W6A8: packed E3M2 weights, group-32 E8M0 scales; identical prequantized MXFP8 activation as W8A8; BF16 output')
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    rows=[];weight_rows=[];start=time.time()
    with ExitStack() as stack, (args.out/'raw.jsonl').open('w',buffering=1) as log:
        files=[stack.enter_context(safe_open(str(f),framework='pt')) for f in paths]
        base,block,mx=files[:3]
        keys=sorted(k for k in mx.keys() if k.endswith('.weight_scale'))
        assert set(keys)=={k for k in block.keys() if k.endswith('.weight_scale')}
        if args.mxfp6:
            mx6=files[3]
            six_keys={k for k in mx6.keys() if k.endswith('.weight_scale')}
            common=sorted(set(keys) & six_keys)
            meta['coverage']=dict(original_fp8_weights=len(keys),common_weights=len(common),
                excluded_from_fp8_comparison=sorted(set(keys)-six_keys),
                fp6_only_weights=sorted(six_keys-set(keys)))
            assert common
            keys=common
            (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        if args.limit:keys=keys[:args.limit]
        for idx,sk in enumerate(keys):
            wk=sk.replace('.weight_scale','.weight')
            w=base.get_tensor(wk).cuda();bw=block.get_tensor(wk).cuda();mw=mx.get_tensor(wk).cuda()
            bs=block.get_tensor(sk).cuda().float();ms=mx.get_tensor(sk).cuda()
            n,k=w.shape
            assert w.dtype==torch.bfloat16 and bw.dtype==mw.dtype==torch.float8_e4m3fn
            assert tuple(bs.shape)==(n//128,k//128) and tuple(ms.shape)==(n,k//32)
            bd=bw.float()*bs.repeat_interleave(128,0).repeat_interleave(128,1)
            md=(mw.float().reshape(n,k//32,32)*torch.exp2(ms.float()-127)[...,None]).reshape(n,k)
            weight_rows.append(dict(weight=wk,shape=[n,k],block=metrics(bd,w),mxfp8=metrics(md,w)))
            packed_w=MXFP8Tensor(mw.view(torch.uint8),pack_scale(ms),n,k)
            if args.mxfp6:
                w6=mx6.get_tensor(wk).cuda();s6=mx6.get_tensor(sk).cuda()
                assert w6.dtype==s6.dtype==torch.uint8
                assert tuple(w6.shape)==(n,k*3//4) and tuple(s6.shape)==(n,k//32)
                packed_w6=PackedMXFP6Tensor(w6,pack_scale(s6),n,k)
                d6=decode_fp6(w6,s6)
                weight_rows[-1]['mxfp6']=metrics(d6,w)
            wf=w.float();wh=w.half()
            for seed in args.seeds:
                for m in args.batches:
                    torch.manual_seed(seed+k*17+m)
                    x=torch.randn(m,k,device='cuda',dtype=torch.float32).bfloat16()
                    ref=x.float()@wf.T
                    bf32=torch.mm(x,w.T,out_dtype=torch.float32)
                    hf32=torch.mm(x.half(),wh.T,out_dtype=torch.float32)
                    bf=torch.mm(x,w.T);hf=torch.mm(x.half(),wh.T)
                    ba,bas=block_quant(x)
                    ma,mas=flashinfer.mxfp8_quantize(x,is_sf_swizzled_layout=True)
                    by=ops.cutlass_scaled_mm(ba,bw.T,bas,bs.T,torch.bfloat16)
                    my=gemm_w8a8(MXFP8Tensor(ma.view(torch.uint8),mas,m,k),packed_w)
                    mm={name:metrics(y,ref) for name,y in [
                        ('bf16_out_fp32',bf32),('fp16_out_fp32',hf32),
                        ('bf16_out_bf16',bf),('fp16_out_fp16',hf),
                        ('block_fp8',by),('mxfp8',my)]}
                    mm.update(block_vs_bf16=metrics(by,bf),mx_vs_bf16=metrics(my,bf),
                              block_vs_fp16=metrics(by,hf),mx_vs_fp16=metrics(my,hf))
                    if args.mxfp6:
                        y6=gemm_w6a8(MXFP8Tensor(ma.view(torch.uint8),mas,m,k),packed_w6,out_dtype=torch.bfloat16)
                        mm.update(mxfp6=metrics(y6,ref),mxfp6_vs_bf16=metrics(y6,bf),mxfp6_vs_fp16=metrics(y6,hf))
                    # Isolate weight quantization and validate hardware against decoded FP32 GEMM.
                    if seed==args.seeds[0] and m==args.batches[-1]:
                        ax=ba.float().reshape(m,k//128,128)*bas[...,None]
                        # Undo the 128x4 scale swizzle produced by FlashInfer.
                        ls=mas.reshape(-1,k//128,32,4,4).transpose(1,3).contiguous().reshape(-1,k//32)[:m]
                        am=(ma.float().reshape(m,k//32,32)*torch.exp2(ls.float()-127)[...,None]).reshape(m,k)
                        wr=weight_rows[-1]
                        wr['weight_only_block_gemm']=metrics(x.float()@bd.T,ref)
                        wr['weight_only_mx_gemm']=metrics(x.float()@md.T,ref)
                        wr['block_kernel_vs_decoded']=metrics(by,(ax.reshape(m,k)@bd.T).bfloat16())
                        wr['mx_kernel_vs_decoded']=metrics(my,(am@md.T).bfloat16())
                        assert wr['block_kernel_vs_decoded']['relative_rmse']<.001
                        assert wr['mx_kernel_vs_decoded']['relative_rmse']<.001
                        if args.mxfp6:
                            wr['weight_only_mxfp6_gemm']=metrics(x.float()@d6.T,ref)
                            wr['mxfp6_kernel_vs_decoded']=metrics(y6,(am@d6.T).bfloat16())
                            assert wr['mxfp6_kernel_vs_decoded']['relative_rmse']<.001
                    row=dict(weight=wk,m=m,n=n,k=k,seed=seed,metrics=mm)
                    rows.append(row);log.write(json.dumps(row)+'\n')
            print(f'{idx+1}/{len(keys)} {wk} elapsed={time.time()-start:.1f}s',flush=True)
    result=dict(cases=len(rows),weights=len(weight_rows),overall=summarize(rows),
        by_m={str(m):summarize([r for r in rows if r['m']==m]) for m in args.batches})
    for label,fn in [('by_projection',lambda r:r['weight'].split('.layers.')[1].split('.',1)[1]),
                     ('by_shape',lambda r:f"{r['n']}x{r['k']}")]:
        groups=defaultdict(list)
        for r in rows:groups[fn(r)].append(r)
        result[label]={k:summarize(v) for k,v in groups.items()}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'weights.json').write_text(json.dumps(weight_rows,indent=2)+'\n')
    print(json.dumps(result['overall'],indent=2))

if __name__=='__main__':main()
