#!/usr/bin/env python3
"""Weight per-shape GEMM measurements by the model's actual projection counts."""
from pathlib import Path

GEOMETRY = Path(__file__).with_name('qwen35_tp1_projection_geometry.json')
BACKENDS = ('native_us', 'flashinfer_us', 'block_fp8_us')


def calculate(rows, geometry):
    by_shape = {}
    for source in rows:
        key = tuple(int(source[k]) for k in ('m', 'n', 'k'))
        if key in by_shape:
            raise ValueError(f'Duplicate measured shape: {key}')
        by_shape[key] = {k: float(source[k]) for k in BACKENDS}
        if any(v <= 0 for v in by_shape[key].values()):
            raise ValueError(f'Nonpositive latency: {key}')
    batches = sorted({key[0] for key in by_shape})
    totals, details = [], []
    for model in geometry['models']:
        projections = model['projections']
        assert sum(p['count'] for p in projections) == model['gemms_per_forward']
        for batch in batches:
            sums = {backend: 0. for backend in BACKENDS}
            oracle = 0.
            for projection in projections:
                key = batch, projection['n'], projection['k']
                if key not in by_shape:
                    raise ValueError(f'Missing measurement for {model["model"]}: {key}')
                times = by_shape[key]
                count = projection['count']
                winner = min(BACKENDS, key=times.__getitem__)
                best = times[winner]
                for backend in BACKENDS:
                    sums[backend] += count * times[backend]
                oracle += count * best
                details.append(dict(model=model['model'], batch=batch,
                    projection=projection['name'], n=key[1], k=key[2], count=count,
                    **times, best_backend=winner.removesuffix('_us'), best_single_us=best,
                    native_weighted_us=count*times['native_us'], oracle_weighted_us=count*best,
                    gap_weighted_us=count*(times['native_us']-best)))
            gap = max(0., sums['native_us']-oracle)
            totals.append(dict(model=model['model'], batch=batch,
                gemms_per_forward=model['gemms_per_forward'], **sums, oracle_us=oracle,
                gap_us=gap, gap_pct_vs_oracle=100*gap/oracle,
                max_reduction_pct=100*gap/sums['native_us'],
                speedup_vs_flashinfer=sums['flashinfer_us']/sums['native_us'],
                speedup_vs_vllm_block_fp8=sums['block_fp8_us']/sums['native_us']))
    return totals, details


def render(totals, geometry):
    lines = ['', '## Model-weighted GEMM estimates', '',
        'TP=1, all 24 batches from 1 to 2048. The six main quantized projections are weighted by their actual layer/call counts: '
        '96 GEMMs for Qwen3.5-2B and 128 for Qwen3.5-4B. These are sums of microbenchmarks, not full-model inference latency. '
        'BF16 projections, lm_head, attention, quantization and communication are excluded.', '',
        'The oracle chooses the fastest of current MXFP8, FlashInfer MXFP8 and vLLM block FP8 per shape before weighting. '
        'A mixed-backend model has not been implemented.', '',
        '| Model | vs FlashInfer across batches | vs vLLM across batches | Largest gap to per-shape oracle |',
        '|---|---:|---:|---:|']
    for model in geometry['models']:
        rows = [r for r in totals if r['model'] == model['model']]
        fi = [r['speedup_vs_flashinfer'] for r in rows]
        vllm = [r['speedup_vs_vllm_block_fp8'] for r in rows]
        peak = max(rows, key=lambda r:r['gap_pct_vs_oracle'])
        lines.append(f"| {model['model']} | {min(fi):.2f}–{max(fi):.2f}× | {min(vllm):.2f}–{max(vllm):.2f}× | {peak['gap_pct_vs_oracle']:.1f}% ({peak['gap_us']:.0f} µs, batch {peak['batch']}) |")
    return lines
