#!/usr/bin/env python3
"""Audit default-dispatch reachability separately from measured usefulness.

W8A8 boundary representatives cover every interval cell of the checked-in
axis-threshold policy and its generated fallback. No GEMM runs in this audit.
"""
import argparse
from collections import Counter
import importlib.util
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BATCHES = [1,2,4,8,10,12,14,16,20,24,28,32,36,40,48,56,64,96,128,192,256,512,1024,2048,4096,8192]


def geometry():
    data=json.loads((ROOT/'benchmarks/qwen35_tp1_projection_geometry.json').read_text())
    return sorted({(p['n'],p['k']) for model in data['models'] for p in model['projections']})


def boundary_shapes(policy):
    cuts={'m':set(policy['m_bounds']) | {8,32,64,128,256,2048},'n':set(),'k':{3072,8192}}
    def tree(node):
        if isinstance(node,dict):
            cuts[node['axis']].add(node['upper']); tree(node['le']); tree(node['gt'])
    for region in policy['regions']:
        for axis in ('n','k'):
            for suffix in ('min','max'):
                value=region.get(f'{axis}_{suffix}')
                if value is not None: cuts[axis].add(value)
        for config in region['configs']: tree(config)
    axes=[]
    for axis in ('m','n','k'):
        alignment=1 if axis=='m' else 128
        values={alignment}
        for cut in cuts[axis]:
            base=cut//alignment*alignment
            # Include both sides of >= and <=, including exact thresholds.
            values.update(v for v in (base-alignment,base,base+alignment) if v>0)
        axes.append(sorted(values))
    return itertools.product(*axes)


def audit():
    sys.path.insert(0,str(ROOT/'python'))
    import mxfp6.mxfp8 as fp8
    candidates=set(fp8.available_configs())
    policy=json.loads((ROOT/'python/mxfp6/mxfp8_dispatch.json').read_text())
    witnesses={}
    cells=0
    for shape in boundary_shapes(policy):
        config=fp8.select_config(*shape)
        witnesses.setdefault(config.config_id,list(shape))
        cells+=1
    real=Counter()
    shape_rows=[]
    for n,k in geometry():
        for m in BATCHES:
            config=fp8.select_config(m,n,k)
            real[config.config_id]+=1
            shape_rows.append(dict(shape=[m,n,k],config=list(config)))
    families=[]
    import re
    registry=(ROOT/'csrc/include/mxfp8_gemm/registry.def').read_text()
    for name,base,count in re.findall(r'MXFP8_FAMILY\((\w+), (\d+), (\d+),',registry):
        ids=set(range(int(base),int(base)+int(count)))
        families.append(dict(name=name,registered=len(ids),default_reachable=sorted(ids & witnesses.keys()),real_shapes_used=sorted(ids & real.keys())))
    w6=json.loads((ROOT/'python/mxfp6/w6a8_dispatch.json').read_text())
    used=set()
    def visit(rules):
        for rule in rules:
            if 'if' in rule: visit(rule['then'])
            else:used.add(rule['config'])
    visit(w6['exact_rules'])
    # Remaining native fallback calls in launch_w6a8_policy (not W6A6).
    used.update((0,1,7,10,11,12,16))
    w6_rows=[dict(id=c['id'],name=c['name']) for c in w6['candidates'] if c['id'] not in used]
    return dict(
        scope='Builtin dense dispatch only; excludes runtime overrides, explicit configurations, dual activation and MoE. Unused by defaults does not imply safe to remove.',
        w6a8=dict(registered=len(w6['candidates']),builtin_referenced=sorted(used),
                  not_referenced=w6_rows,
                  also_used_by_w6a6_mixed_path=[6,9],
                  note='Reference analysis, not exhaustive reachability. All 30 candidates remain eligible for runtime autotuning.'),
        w8a8=dict(registered=len(candidates),boundary_cells=cells,
                  reachable=sorted(witnesses),witnesses=witnesses,
                  never_selected_by_builtin=sorted(candidates-witnesses.keys()),
                  real_shape_count=len(shape_rows),real_shape_used=sorted(real),
                  real_shape_hits=dict(real),families=families,real_shape_dispatch=shape_rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    w8=report['w8a8'];w6=report['w6a8']
    print(f"W6A8: {w6['registered']} candidates; {len(w6['not_referenced'])} not referenced by builtin W6A8 policy")
    print(f"W8A8: {w8['registered']} tactics; {len(w8['reachable'])} reachable across {w8['boundary_cells']} interval cells; {len(w8['real_shape_used'])} selected on {w8['real_shape_count']} real shapes")
    print('Families never selected:',[f['name'] for f in w8['families'] if not f['default_reachable']])
