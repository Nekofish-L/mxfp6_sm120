#!/usr/bin/env python3
"""Paired native-library regression check, using isolated processes on one GPU.

Records bench_kineto GPU durations for Graph replay and eager launches, in
warm-cache and explicit-flush regimes. Both sides
use the current Python API and identical seeded operands, so this isolates
native-library changes. Correctness compares complete outputs, not samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
W6_SHAPES = [
    (1,8192,5120), (16,5120,3072), (24,8192,5120), (32,5120,3072),
    (40,5120,3072), (48,5120,3072), (64,5120,8704), (96,17408,5120),
    (512,7168,5120), (1024,8192,5120), (2048,5120,3072),
    (4096,5120,3072), (8192,5120,8704), (17,136,128), (129,256,384),
    (513,1536,2048),
]


def worker(args):
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    import mxfp6
    import mxfp6.mxfp8 as fp8
    from kineto_utils import load_bench_kineto, kernel_tag
    bench_kineto, kineto_source = load_bench_kineto(args.kineto_source)

    torch.manual_seed(20261116)
    cases = ([(shape, dtype, None) for shape in W6_SHAPES for dtype in (torch.float16, torch.bfloat16)]
             if args.backend == "w6a8" else
             [((m, n, k), torch.bfloat16, None) for m,n,k in
              [(1,4096,2560),(16,4096,2560),(32,12288,2560),(64,4096,2560),(129,512,1024),(512,4096,2560)]]
             + [((32,512,2560), torch.bfloat16, t) for t in
                (200,500,520,560,570,600,620,660,700,720,740,1120,1140,1160,1180,1220,1240)])
    if args.builtin_only:
        cases = [case for case in cases if case[2] is None]
    results = []
    output_dir = args.output.parent / (args.output.stem + "-tensors")
    output_dir.mkdir()
    for index, ((m,n,k), dtype, tactic) in enumerate(cases):
        torch.manual_seed(20261116 + index)
        x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.1
        w = torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*.1
        a = mxfp6.quantize_activation(x)
        if args.backend == "w6a8":
            b = mxfp6.quantize_mxfp6(w)
            def run():
                return mxfp6.gemm_w6a8(a,b,out_dtype=dtype)
        else:
            b = fp8.quantize_mxfp8(w)
            config = None if tactic is None else fp8.W8A8Config(tactic,1,1,0)
            def run():
                return fp8.gemm_w8a8(a,b,config=config)
        # Warm kernels, allocators and clocks before constructing the graph.
        for _ in range(30):
            result = run()
        torch.cuda.synchronize()
        cpu_output = result.cpu()
        tensor_path = output_dir / f"{index}.pt"
        torch.save(cpu_output, tensor_path)
        digest = hashlib.sha256(cpu_output.view(torch.uint8).numpy().tobytes()).hexdigest()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        graph_tag, graph_kernel = kernel_tag(graph.replay)
        eager_tag, eager_kernel = kernel_tag(run)
        timings = {}
        for regime in args.regimes.split(','):
            for mode, fn, tag in (('graph',graph.replay,graph_tag),('eager',run,eager_tag)):
                samples = [bench_kineto(fn,tag,num_tests=args.iterations,
                                        suppress_kineto_output=True,flush_l2=regime=='cold')*1e6
                           for _ in range(args.samples)]
                if min(samples)<=0:
                    raise RuntimeError('bench_kineto returned no kernel timing')
                timings[f'{mode}_{regime}_us'] = samples
        eager_result = run()
        torch.testing.assert_close(result.cpu(), cpu_output, rtol=2e-3, atol=.5)
        torch.testing.assert_close(eager_result.cpu(), cpu_output, rtol=2e-3, atol=.5)
        results.append(dict(shape=[m,n,k],dtype=str(dtype),tactic=tactic,
                            timings=timings,graph_kernel=graph_kernel,eager_kernel=eager_kernel,
                            output_sha256=digest,output_path=str(tensor_path)))
        graph.reset()
        del graph, result, eager_result, a, b, x, w
    library = Path(os.environ["MXFP6_LIBRARY_PATH" if args.backend == "w6a8" else "MXFP8_LIBRARY_PATH"])
    args.output.write_text(json.dumps(dict(
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        stream_k=os.environ.get("MXFP6_STREAM_K", "1"),
        kineto_source=str(kineto_source),kineto_sha256=hashlib.sha256(kineto_source.read_bytes()).hexdigest(),
        iterations=args.iterations,cpu_affinity=sorted(os.sched_getaffinity(0)),results=results),indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before",type=Path)
    p.add_argument("--after",type=Path)
    p.add_argument("--backend",choices=("w6a8","w8a8"),default="w6a8")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=2)
    p.add_argument("--samples",type=int,default=3)
    p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--kineto-source",type=Path)
    p.add_argument("--builtin-only",action="store_true")
    p.add_argument("--regimes",default="warm,cold")
    p.add_argument("--cpu",type=int,help="Pin workers to the same allowed CPU for eager wall-time comparisons")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = p.parse_args()
    if any(regime not in ('warm','cold') for regime in args.regimes.split(',')):
        p.error('--regimes must contain warm and/or cold')
    if min(args.rounds,args.samples,args.iterations) < 1:
        p.error("rounds, samples and iterations must be positive")
    if args.worker:
        worker(args); return
    if args.before is None or args.after is None:
        p.error("--before and --after are required")
    import torch
    runs = {"before":[],"after":[]}
    with tempfile.TemporaryDirectory(prefix="mxfp-refactor-") as temp:
        for rnd in range(args.rounds):
            for label in (("before","after") if rnd%2==0 else ("after","before")):
                output = Path(temp)/f"{label}-{rnd}.json"
                env = dict(os.environ,MXFP6_AUTOTUNE="off")
                env["MXFP6_LIBRARY_PATH" if args.backend=="w6a8" else "MXFP8_LIBRARY_PATH"] = str(getattr(args,label).resolve())
                command = [sys.executable,__file__,"--worker","--backend",args.backend,
                                "--output",str(output),"--samples",str(args.samples),
                                "--iterations",str(args.iterations),"--regimes",args.regimes]
                if args.kineto_source is not None:
                    command.extend(["--kineto-source", str(args.kineto_source.resolve())])
                if args.builtin_only:
                    command.append("--builtin-only")
                if args.cpu is not None:
                    command.extend(["--cpu", str(args.cpu)])
                subprocess.run(command,env=env,check=True)
                runs[label].append(json.loads(output.read_text()))
                print(f"finished {label} round {rnd+1}",flush=True)
        comparisons=[]
        for i, reference in enumerate(runs['before'][0]['results']):
            ref=torch.load(reference['output_path'],weights_only=True)
            for group in runs.values():
                for run in group:
                    item=run['results'][i]
                    assert (item['shape'],item['dtype'],item['tactic']) == (reference['shape'],reference['dtype'],reference['tactic'])
                    torch.testing.assert_close(torch.load(item['output_path'],weights_only=True),ref,rtol=2e-3,atol=.5)
            entry={k:reference[k] for k in ('shape','dtype','tactic')}
            entry['bitwise_equal']=all(run['results'][i]['output_sha256']==reference['output_sha256'] for group in runs.values() for run in group)
            for metric in reference['timings']:
                before=statistics.median([v for run in runs['before'] for v in run['results'][i]['timings'][metric]])
                after=statistics.median([v for run in runs['after'] for v in run['results'][i]['timings'][metric]])
                entry[metric]=dict(before=before,after=after,change_percent=(after/before-1)*100)
            comparisons.append(entry)
        for group in runs.values():
            for run in group:
                for item in run['results']:
                    item.pop('output_path')
    report=dict(backend=args.backend,method="Alternating isolated processes; identical seeded quantized inputs; bench_kineto GPU kernel durations for single-GEMM Graph replay and eager launches, warm-cache and reference-controlled L2 flush",runs=runs,comparisons=comparisons)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(comparisons,indent=2))


if __name__=="__main__":
    main()
