#!/usr/bin/env python3
"""Report conservative promotion of the wide-N / short-K candidate kernels."""
import argparse,csv,hashlib,json,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path);p.add_argument('--write-dispatch',type=Path)
a=p.parse_args()
rows=[json.loads(l) for f in sorted((a.directory/'validation').glob('raw_*.jsonl')) for l in f.read_text().splitlines()]
assert len(rows)==144,len(rows)
assert len({(r['m'],r['n'],r['k']) for r in rows})==144
old_path=ROOT/'benchmarks/results/mxfp8_decode_validation/dispatch.json'
old=json.loads(old_path.read_text())
classification={}
for file in (ROOT/'benchmarks/results/mxfp8_decode_validation').glob('raw_*.jsonl'):
 for line in file.read_text().splitlines():
  r=json.loads(line);classification[r['m'],r['n'],r['k']]=r['outlier']
from sys import path
path.insert(0,str(ROOT/'python'))
from mxfp6._mxfp8_triton import CONFIGS
records=[];configs={};promoted=[]
for r in sorted(rows,key=lambda r:(r['n'],r['k'],r['m'])):
 key=f'{r["m"]},{r["n"]},{r["k"]}'
 b=r['backends'];candidate=r['selected']!=r['old']
 use=candidate and r['speedup']>1.02 and min(b['old']['samples_us'])>max(b['new']['samples_us'])
 chosen=r['selected'] if use else r['old']
 config=chosen['config'] if chosen['kind']=='cutlass' else [100+CONFIGS.index(tuple(chosen['config'])),1,1,0]
 configs[key]=config
 if use:promoted.append(key)
 native=b['new']['us'] if use else b['old']['us']
 fi=b['flashinfer']['us'];block=b['block_fp8']['us'];humming=b['humming']['us']
 outlier=classification[r['m'],r['n'],r['k']]
 base=min(fi,block) if outlier else humming;threshold=1.1 if outlier else 1.05
 records.append(dict(m=r['m'],n=r['n'],k=r['k'],promoted=use,tactic=config[0],old_us=b['old']['us'],
   candidate_us=b['new']['us'],deployed_us=native,flashinfer_us=fi,block_fp8_us=block,humming_us=humming,
   speedup_vs_old=b['old']['us']/native,speedup_vs_flashinfer=fi/native,speedup_vs_block=block/native,
   outlier=outlier,required_speedup=threshold,old_pass=base/b['old']['us']>threshold,new_pass=base/native>threshold))
with (a.directory/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary=dict(shapes=len(records),promoted=len(promoted),geomean_speedup=statistics.geometric_mean(r['speedup_vs_old'] for r in records),
 promoted_geomean_speedup=statistics.geometric_mean(r['speedup_vs_old'] for r in records if r['promoted']),
 old_pass=sum(r['old_pass'] for r in records),new_pass=sum(r['new_pass'] for r in records),groups=[])
lines=['# Wide-N / short-K MXFP8 optimization','',
 '## Method and promotion policy','',
 'GPUs 4/5, RTX 5090. All 144 shapes across the six N/K pairs are freshly measured with old native, candidate native, FlashInfer MXFP8, vLLM block FP8, and Humming/default heuristics on the same GPU. Three interleaved trials, 30 active measurements per trial; report the median of trial means.',
 '`gemm_bench.py::bench_kineto` times a pre-captured single-GEMM graph after an 8 GB L2 flush. Only the GEMM GPU time is counted. Quantization, eviction, and host gaps are excluded. This models hot launch + cold weights; it is not full-model inference latency.',
 'Search froze 49 proposed changes before validation. Promotion requires validation speedup >1.02x and the slowest new trial faster than the fastest old trial. Otherwise the shipped dispatch retains the old configuration. For unchanged points, deployed time equals the old measurement. Candidate measurements are retained in CSV/raw JSONL. Reported deployment gains therefore describe this conservative selection, not an additional holdout run.',
 'The original Humming outlier classification is preserved. Normal points require >1.05x freshly measured Humming; outliers require >1.1x the faster freshly measured FlashInfer/block-FP8 baseline. Counts here cover these 144 shapes only and cannot be combined directly with the earlier GPU-6/7 report.', '',
 '## Implementation','',
 'The selected custom Triton kernels use independent 4-warp CTAs, block-scaled `tl.dot_scaled`, 128/256-element K tiles, and direct BF16 stores. Small batches can compute the transposed GEMM to fit the MMA tile. E4M3 values, E8M0/32 scales, FP32 accumulation, and the external weight format are preserved. Runtime does not call FlashInfer, Humming, or vLLM.',
 'Two-stage CUTLASS variants and larger static grids were also tested. Narrowing the weight tile to 32 rows was slower on the representative 8192x2048 tests; these candidates are not in the selected dispatch.', '',
 f'Promoted **{summary["promoted"]}/144** shapes. Geometric mean speedup across all shapes: **{summary["geomean_speedup"]:.3f}x**; promoted shapes: **{summary["promoted_geomean_speedup"]:.3f}x**.',
 f'Original acceptance policy: **{summary["old_pass"]}/144 → {summary["new_pass"]}/144** pass using the same freshly measured baselines. The full performance goal remains unmet.', '',
 '## By N/K','', '| N | K | Promoted M | Geomean speedup (all 24 M) | Old pass | New pass |',
 '|---:|---:|---|---:|---:|---:|']
for n,k in sorted({(r['n'],r['k']) for r in records}):
 sub=[r for r in records if (r['n'],r['k'])==(n,k)]
 item=dict(n=n,k=k,promoted_m=[r['m'] for r in sub if r['promoted']],geomean=statistics.geometric_mean(r['speedup_vs_old'] for r in sub),old_pass=sum(r['old_pass'] for r in sub),new_pass=sum(r['new_pass'] for r in sub))
 summary['groups'].append(item)
 lines.append(f'| {n} | {k} | {", ".join(map(str,item["promoted_m"])) or "—"} | {item["geomean"]:.3f}x | {item["old_pass"]}/24 | {item["new_pass"]}/24 |')
lines+=['','## Every shape (microseconds)','',
 '| M | N | K | Promoted | Old | Deployed | FlashInfer | block FP8 | Humming | vs old | Old pass | New pass |',
 '|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|']
for r in records:
 lines.append(f'| {r["m"]} | {r["n"]} | {r["k"]} | {r["promoted"]} | {r["old_us"]:.3f} | {r["deployed_us"]:.3f} | {r["flashinfer_us"]:.3f} | {r["block_fp8_us"]:.3f} | {r["humming_us"]:.3f} | {r["speedup_vs_old"]:.3f}x | {r["old_pass"]} | {r["new_pass"]} |')
(a.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
(a.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.directory/'deployed_plan.json').write_text(json.dumps(dict(configs=configs),indent=2)+'\n')
if a.write_dispatch:
 old['configs'].update(configs)
 old['short_k_update']=dict(source=str(a.directory),previous_dispatch_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),promoted=promoted,metric='hot graph launch + cold weight cache',gpus=[4,5])
 a.write_dispatch.write_text(json.dumps(old,indent=2)+'\n')
 subprocess.run([sys.executable,str(ROOT/'benchmarks/summarize_mxfp8.py'),
                 str(ROOT/'benchmarks/results/mxfp8_decode_validation'),
                 '--baseline-update',str(a.directory.resolve())],
                check=True,stdout=subprocess.PIPE,text=True)
print(json.dumps(summary,indent=2))
