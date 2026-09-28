#!/usr/bin/env python3
"""Summarize hot-launch / cold-weight acceptance without discarding failures."""
import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--write-dispatch',type=Path)
p.add_argument('--baseline-update',type=Path,
               help='Persist a newer short-K comparison directory as a baseline overlay')
a=p.parse_args()
rows=[json.loads(line) for f in sorted(a.directory.glob('raw_*.jsonl')) for line in f.read_text().splitlines()]
assert rows, 'No measurements found'
assert len({(r['m'],r['n'],r['k']) for r in rows})==len(rows), 'Duplicate measurements'
rows.sort(key=lambda r:(r['n'],r['k'],r['m']))
fields=['m','n','k','outlier','tactic','splits','swizzle','sms','graph_us','target_us','baseline_backend','baseline_graph_us','required_speedup','speedup','pass','relative_rmse','quantized_rmse']
with (a.directory/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,lineterminator='\n',fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
lines=['# SM120 MXFP8: hot graph launch + cold weight cache','',
       'Inputs: prequantized E4M3 with E8M0 scales per 32 values; BF16 output, FP32 accumulation.',
       'Timing uses `gemm_bench.py::bench_kineto` on a pre-captured single-GEMM graph. Each replay follows an 8 GB L2 flush. Only the GPU GEMM duration is counted; flushing, host launch gaps, quantization, and tuning are excluded.',
       'Normal points require >1.05x Humming; original outliers require >1.1x the faster vLLM/FlashInfer result. All three baselines use the same GPU and test function. Frozen-config validation may reuse those measurements; metadata records the source files and hashes.',
       'The original outlier manifest is preserved. This is a GEMM microbenchmark; it does not measure a complete vLLM decode step.',
       'Runs use GPUs 6/7. The original GPU-1 measurements are used only for the shape list and fixed outlier classification.', '']
summary={'shapes':len(rows),'passed':sum(r['pass'] for r in rows),'failed':sum(not r['pass'] for r in rows)}
for label,subset in [('normal',[r for r in rows if not r['outlier']]),('outlier',[r for r in rows if r['outlier']])]:
    speeds=[r['speedup'] for r in subset]
    if not speeds:continue
    passed=sum(r['pass'] for r in subset)
    summary[label]=dict(passed=passed,total=len(subset),minimum=min(speeds),geomean=statistics.geometric_mean(speeds))
    lines.append(f'- {label}: {passed}/{len(subset)} pass; minimum {min(speeds):.3f}x, geometric mean {statistics.geometric_mean(speeds):.3f}x.')
lines+=['',f'**{summary["passed"]}/{len(rows)} pass; {summary["failed"]} fail.**', '',
        '## Shape groups','', '| N | K | Passing M | Failing M |','|---:|---:|---|---|']
for n,k in sorted({(r['n'],r['k']) for r in rows}):
    subset=[r for r in rows if (r['n'],r['k'])==(n,k)]
    passed=', '.join(str(r['m']) for r in subset if r['pass']) or '—'
    failed=', '.join(str(r['m']) for r in subset if not r['pass']) or '—'
    lines.append(f'| {n} | {k} | {passed} | {failed} |')
lines+=['','## Every shape','',
        '| M | N | K | Outlier | Tactic/splits/swizzle/SMs | Graph µs | Target µs | Speedup | Pass |',
        '|---:|---:|---:|:---:|:---:|---:|---:|---:|:---:|']
for r in rows:
    lines.append(f'| {r["m"]} | {r["n"]} | {r["k"]} | {r["outlier"]} | {r["tactic"]}/{r["splits"]}/{r["swizzle"]}/{r.get("sms",0)} | {r["graph_us"]:.3f} | {r["target_us"]:.3f} | {r["speedup"]:.3f} | {r["pass"]} |')
(a.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
(a.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

# Report both requested baselines for every shape, independently of the
# Humming outlier classification used by the original acceptance policy.
baseline_rows=[]
manifest_path=a.directory/'baseline_updates.json'
updates=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
if a.baseline_update:
    relative=os.path.relpath(a.baseline_update.resolve(),a.directory.resolve())
    updates=[u for u in updates if u!=relative]+[relative]
    manifest_path.write_text(json.dumps(updates,indent=2)+'\n')
gpu_by_shape={}
for file in a.directory.glob('raw_*.jsonl'):
    meta=json.loads((a.directory/f'metadata_{file.stem.split("_")[-1]}.json').read_text())
    for line in file.read_text().splitlines():
        r=json.loads(line);gpu_by_shape[r['m'],r['n'],r['k']]=meta['visible_devices']
for r in rows:
    native=r['backends']['native']['us']
    fi=r['backends']['flashinfer_mxfp8']['us']
    block=r['backends']['vllm_block_fp8']['us']
    baseline_rows.append(dict(m=r['m'],n=r['n'],k=r['k'],native_us=native,
        flashinfer_us=fi,block_fp8_us=block,
        speedup_vs_flashinfer=fi/native,speedup_vs_block_fp8=block/native,
        speedup_vs_best=min(fi,block)/native,
        gpu=gpu_by_shape[r['m'],r['n'],r['k']],measurement=a.directory.name))
by_shape={(r['m'],r['n'],r['k']):r for r in baseline_rows}
for relative in updates:
    directory=a.directory/relative
    update_gpu={}
    for file in (directory/'validation').glob('raw_*.jsonl'):
        meta=json.loads((directory/'validation'/f'metadata_{file.stem.split("_")[-1]}.json').read_text())
        for line in file.read_text().splitlines():
            r=json.loads(line);update_gpu[r['m'],r['n'],r['k']]=meta['visible_devices']
    with (directory/'comparison.csv').open() as f:
        for update in csv.DictReader(f):
            key=tuple(int(update[x]) for x in ['m','n','k'])
            assert key in by_shape, f'Unknown shape in baseline update: {key}'
            native=float(update['deployed_us']);fi=float(update['flashinfer_us']);block=float(update['block_fp8_us'])
            by_shape[key].update(native_us=native,flashinfer_us=fi,block_fp8_us=block,
                speedup_vs_flashinfer=fi/native,speedup_vs_block_fp8=block/native,
                speedup_vs_best=min(fi,block)/native,gpu=update_gpu[key],measurement=directory.resolve().name)
with (a.directory/'baselines.csv').open('w') as f:
    writer=csv.DictWriter(f,lineterminator='\n',fieldnames=list(baseline_rows[0]))
    writer.writeheader();writer.writerows(baseline_rows)
baseline_summary={}
report=['# Native MXFP8 vs FlashInfer and block FP8','',
    f'Last updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}. This is the rolling comparison for the selected runtime dispatch.',
    'Method: `gemm_bench.py::bench_kineto`; pre-captured single-GEMM CUDA graph replay, 8 GB L2 eviction before each replay, only GPU GEMM time counted. Quantization and eviction are excluded. This models hot launch + cold weight cache; it is not full-model latency.',
    'Each shape uses the same physical RTX 5090 for all backends. GPU and measurement source are shown per row. Sampling settings and reuse provenance are in metadata. Raw samples are retained; regenerating this report does not perform a new timing run.',
    'FlashInfer uses `mm_mxfp8(..., backend="cutlass")` with exact-M autotuning. Native and FlashInfer share E4M3 inputs and E8M0/32 scales. vLLM block FP8 uses `cutlass_scaled_mm`, activation scales per 1x128 and weight scales per 128x128, from the same source tensors. Outputs are BF16; quantization formats differ.',
    'Speedup = baseline time / native time; >1 means native is faster. All shapes have equal weight in geometric means. Counts below use strict thresholds and do not use the Humming outlier policy.',
    'This table combines the latest available measurements per shape, not one simultaneous run. The original `REPORT.md` and raw files in this directory remain the historical initial validation.', '',
    *[f'- Source `{name}`: {sum(r["measurement"]==name for r in baseline_rows)} shapes; GPUs {", ".join(sorted({r["gpu"] for r in baseline_rows if r["measurement"]==name}))}.' for name in sorted({r['measurement'] for r in baseline_rows})],
    *[f'- [Update report]({relative}/REPORT.md): promotion policy, before/after timing, and fresh baseline provenance.' for relative in updates], '',
    '## All shapes','',
    '| Baseline | Geometric mean speedup | >1x | >1.05x | >1.1x |',
    '|---|---:|---:|---:|---:|']
for label,key in [('FlashInfer MXFP8','speedup_vs_flashinfer'),('vLLM block FP8','speedup_vs_block_fp8'),('Faster of both','speedup_vs_best')]:
    values=[r[key] for r in baseline_rows]
    stats=dict(total=len(values),geomean=statistics.geometric_mean(values),
               faster=sum(v>1 for v in values),above_1_05=sum(v>1.05 for v in values),above_1_1=sum(v>1.1 for v in values))
    baseline_summary[label]=stats
    report.append(f'| {label} | {stats["geomean"]:.3f}x | {stats["faster"]}/{len(values)} | {stats["above_1_05"]}/{len(values)} | {stats["above_1_1"]}/{len(values)} |')
report+=['','## By M (geometric mean speedup across N/K)','',
    '| M | Shapes | vs FlashInfer | vs block FP8 | vs faster of both |',
    '|---:|---:|---:|---:|---:|']
for m in sorted({r['m'] for r in baseline_rows}):
    subset=[r for r in baseline_rows if r['m']==m]
    values=[statistics.geometric_mean(r[key] for r in subset) for key in
            ['speedup_vs_flashinfer','speedup_vs_block_fp8','speedup_vs_best']]
    report.append(f'| {m} | {len(subset)} | {values[0]:.3f}x | {values[1]:.3f}x | {values[2]:.3f}x |')
report+=['','## Every shape (microseconds)','',
    '| M | N | K | Native | FlashInfer | block FP8 | vs FlashInfer | vs block FP8 | GPU | Source |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in baseline_rows:
    report.append(f'| {r["m"]} | {r["n"]} | {r["k"]} | {r["native_us"]:.3f} | {r["flashinfer_us"]:.3f} | {r["block_fp8_us"]:.3f} | {r["speedup_vs_flashinfer"]:.3f}x | {r["speedup_vs_block_fp8"]:.3f}x | {r["gpu"]} | {r["measurement"]} |')
(a.directory/'BASELINES.md').write_text('\n'.join(report)+'\n')
(a.directory/'baseline_summary.json').write_text(json.dumps(baseline_summary,indent=2)+'\n')
if a.write_dispatch:
    dispatch=dict(version=1,metric='hot graph launch + cold weight cache; gemm_bench.bench_kineto',
                  source=str(a.directory),configs={f'{r["m"]},{r["n"]},{r["k"]}':[r['tactic'],r['splits'],r['swizzle'],r.get('sms',0)] for r in rows})
    a.write_dispatch.parent.mkdir(parents=True,exist_ok=True)
    a.write_dispatch.write_text(json.dumps(dispatch,indent=2)+'\n')
print(json.dumps(summary,indent=2))
