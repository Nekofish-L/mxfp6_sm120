#!/usr/bin/env python3
"""Write a compact performance summary from native/FlashInfer/vLLM measurements."""
import argparse
import json
from pathlib import Path
import statistics

from model_gemm_totals import calculate, GEOMETRY, render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--overlay',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    rows={}
    gpus=set()
    for directory in [args.directory]+args.overlay:
        for file in sorted(directory.glob('raw_*.jsonl')):
            meta=json.loads(file.with_name(file.name.replace('raw_','metadata_').replace('.jsonl','.json')).read_text())
            gpus.add(meta['visible_devices'])
            for line in file.read_text().splitlines():
                row=json.loads(line)
                rows[row['m'],row['n'],row['k']]=row
    assert rows and gpus<= {'4','5'}
    values=list(rows.values())
    lines=['# MXFP8 performance summary','',
        f'RTX 5090 / SM120, physical GPUs {", ".join(sorted(gpus))}; {len(values)} shapes. '
        'Single-GEMM CUDA graph, 8 GB L2 eviction before replay; only GPU kernel time is counted. '
        'Quantization, host overhead and eviction are excluded.','',
        'FlashInfer uses MXFP8 `mm_mxfp8(..., backend="cutlass")` and exact-batch autotuning. '
        'vLLM uses block FP8 `cutlass_scaled_mm`. MXFP8 inputs share E4M3 values and E8M0/32 scales; '
        'vLLM quantizes the same source tensors with 1×128 activation / 128×128 weight scales.','',
        'Speedup = baseline latency / current latency. Figures are approximate and describe this GPU and cache policy.','',
        '| Baseline | Geometric mean speedup | Shapes faster |','|---|---:|---:|']
    for label,backend in [('FlashInfer MXFP8','flashinfer'),('vLLM block FP8','block_fp8')]:
        ratios=[r['backends'][backend]['us']/r['backends']['native']['us'] for r in values]
        lines.append(f'| {label} | {statistics.geometric_mean(ratios):.2f}× | {sum(v>1 for v in ratios)}/{len(values)} |')
    normalized=[dict(m=r['m'],n=r['n'],k=r['k'],native_us=r['backends']['native']['us'],
                    flashinfer_us=r['backends']['flashinfer']['us'],block_fp8_us=r['backends']['block_fp8']['us']) for r in values]
    geometry=json.loads(GEOMETRY.read_text())
    expected={(m,p['n'],p['k']) for m in [1,2,4,8,10,12,14,16,20,24,28,32,36,40,48,56,64,96,128,192,256,512,1024,2048]
              for model in geometry['models'] for p in model['projections']}
    if set(rows)==expected:
        totals,_=calculate(normalized,geometry)
        lines+=render(totals,geometry)
        if all('reference' in r['backends'] for r in values):
            before=[dict(n, native_us=r['backends']['reference']['us']) for n,r in zip(normalized,values)]
            prior,_=calculate(before,geometry)
            lines+=['','## Range scheduling versus the former point table','',
                    'Both configurations are measured on the same GPU for each shape. Positive changes mean more time.','',
                    '| Model | Weighted latency change across batches | Worst absolute increase |','|---|---:|---:|']
            for model in geometry['models']:
                pairs=[(a,b) for a,b in zip(totals,prior) if a['model']==model['model']]
                changes=[100*(a['native_us']/b['native_us']-1) for a,b in pairs]
                worst=max(a['native_us']-b['native_us'] for a,b in pairs)
                lines.append(f"| {model['model']} | {min(changes):+.1f}% to {max(changes):+.1f}% | {worst:.0f} µs |")
    checked=sum(r.get('changing_graph_rmse') is not None for r in values)
    lines+=['',f'Changing-input/scale and zero-input graph checks passed for {checked}/{len(values)} measured shapes.','']
    output=args.output or args.directory/'BASELINES.md'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines))
    print(output)


if __name__=='__main__':main()
