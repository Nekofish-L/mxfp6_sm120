#!/usr/bin/env python3
"""Evaluate a wave/tile shortlist offline; never install a runtime policy.

This deliberately tests the transferable geometric part of DeepGEMM's ranking.
It does not substitute SM100 resource budgets for SM120 measurements.
"""
import argparse
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def rank(candidates, m, n, sms):
    def score(candidate):
        bm,bn,_=candidate['tile']
        if candidate['launch']['kind']=='swapped':bm,bn=bn,bm
        blocks=((m+bm-1)//bm)*((n+bn-1)//bn)
        waves=(blocks+sms-1)//sms
        last=blocks%sms or sms
        return waves,-last,bm+bn,bm*bn,candidate['id']
    return [c['id'] for c in sorted(candidates,key=score)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screens',type=Path,nargs='+')
    parser.add_argument('--sm-count',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    candidates=json.loads((ROOT/'python/mxfp6/w6a8_dispatch.json').read_text())['candidates']
    cases=[]
    for path in args.screens:
        data=json.loads(path.read_text())
        if data['metadata']['backend']!='w6a8' or data['metadata']['timer']!='bench_kineto':
            raise ValueError('expected W6A8 bench_kineto screening data')
        for row in data['results']:
            by_id={}
            for probe in row['coarse']:
                cid=probe['config'][0]
                by_id[cid]=min(by_id.get(cid,float('inf')),probe['us'])
            if not by_id:continue
            m,n,_=row['shape'];order=rank(candidates,m,n,args.sm_count)
            oracle=min(by_id.values())
            regrets={count: (min(by_id.get(cid,float('inf')) for cid in order[:count])/oracle-1)*100
                     for count in (3,6,12,20,30)}
            cases.append(dict(shape=row['shape'],dtype=row['dtype'],regime=data['metadata']['regime'],regret_percent=regrets))
    summaries=[]
    for count in (3,6,12,20,30):
        values=[case['regret_percent'][count] for case in cases]
        summaries.append(dict(shortlist=count,within_2_percent=sum(v<=2 for v in values),cases=len(values),median_regret_percent=statistics.median(values),worst_regret_percent=max(values)))
    report=dict(method='wave count, last-wave utilization, tile perimeter/area; stable ID tie-break; measured coarse-candidate oracle',
                sm_count=args.sm_count,summary=summaries,cases=cases,
                decision='Diagnostic only. Keep current runtime candidate policy; resource and pipeline effects are not modeled.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
