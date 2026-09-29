"""Public API and native workspace regressions shared with the MXFP6 workflow."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
import mxfp6
import mxfp6.mxfp8 as mxfp8

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,n,k", [(3, 128, 128), (17, 256, 512), (65, 384, 1024)])
def test_public_gemm_and_float_activation(m, n, k, dtype):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    qa = mxfp8.quantize_activation(a)
    qb = mxfp8.quantize_mxfp8(b)
    expected = mxfp8.gemm(qa, qb)
    torch.testing.assert_close(mxfp8.gemm(a, qb), expected, rtol=0, atol=0)
    torch.testing.assert_close(mxfp8.gemm_from_float(a, qb), expected, rtol=0, atol=0)
    torch.testing.assert_close(mxfp8.gemm_w8a8(qa, qb), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        mxfp8.gemm_packed(qa.values, qb.values, qa.scales, qb.scales, m, n, k),
        expected,
        rtol=0,
        atol=0,
    )
    assert mxfp8.warmup(a, qb, iterations=1) == mxfp8.select_config(m, n, k)
    for alpha, dtype in [(2.0, torch.bfloat16), (1.0, torch.float16)]:
        with pytest.raises(ValueError):
            mxfp8.gemm(qa, qb, alpha, out_dtype=dtype)


def test_compiled_policy_matches_source_intervals():
    mxfp8.load_library()
    path = ROOT / "python/mxfp6/mxfp8_dispatch.json"
    policy = json.loads(path.read_text())
    assert (
        torch.ops.mxfp8_sm120.policy_sha256()
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    # Include both sides of every threshold and unseen combinations of axes.
    dimensions = {"m": {1, 2049, 8192}, "n": {128, 32768}, "k": {128, 12288}}
    for v in policy["m_bounds"]:
        dimensions["m"].update([v, v + 1])

    def visit(c):
        if isinstance(c, dict):
            dimensions[c["axis"]].update([c["upper"], c["upper"] + 1])
            visit(c["le"])
            visit(c["gt"])

    for r in policy["regions"]:
        for axis in ("n", "k"):
            if r[axis + "_max"] is not None:
                dimensions[axis].update([r[axis + "_max"], r[axis + "_max"] + 1])
        for c in r["configs"]:
            visit(c)

    def expected(m, n, k):
        for r in policy["regions"]:
            if (r["n_max"] is None or n <= r["n_max"]) and (
                r["k_max"] is None or k <= r["k_max"]
            ):
                for bound, c in zip(policy["m_bounds"], r["configs"]):
                    if m <= bound:
                        while isinstance(c, dict):
                            c = c[
                                (
                                    "le"
                                    if dict(m=m, n=n, k=k)[c["axis"]] <= c["upper"]
                                    else "gt"
                                )
                            ]
                        return tuple(c)
                return (79, 1, 1, 0)
        if m > 2048:
            return (79, 1, 1, 0)
        if k > 3072:
            return (
                (
                    12
                    if m <= 8
                    else 13 if m <= 32 else 14 if m <= 64 else 15 if m <= 256 else 16
                ),
                (4 if k >= 8192 else 2) if m <= 128 else 1,
                1,
                0,
            )
        return (
            0 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 7 if m <= 64 else 9,
            1,
            1,
            0,
        )

    for m in dimensions["m"]:
        for n in dimensions["n"]:
            for k in dimensions["k"]:
                assert tuple(mxfp8.select_config(m, n, k)) == expected(m, n, k)


def test_native_workspace_lanes_and_backend_isolation():
    code = r"""
import torch,mxfp6
import mxfp6.mxfp8 as api
runs=[]
for m,n,k,t in [(1,128,8704,12),(17,256,4096,13),(96,384,2048,15)]:
    a=api.quantize_mxfp8(torch.randn(m,k,device='cuda',dtype=torch.bfloat16))
    b=api.quantize_mxfp8(torch.randn(n,k,device='cuda',dtype=torch.bfloat16))
    config=api.W8A8Config(t,4,1,0)
    runs.append((a,b,lambda a=a,b=b,c=config:api.gemm_w8a8(a,b,config=c)))
refs=[fn().clone() for a,b,fn in runs]
api.begin_workspace_planning()
for a,b,fn in runs:fn()
assert api.workspace_stats()['layouts']>=2
assert mxfp6.workspace_stats()['planning']==0
stats=api.finalize_workspace_planning()
assert stats['frozen'] and stats['arena_bytes']>0
streams=[torch.cuda.Stream(),torch.cuda.Stream()]
graphs=[]
for i,stream in enumerate(streams):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runs[i][2]()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=runs[i][2]()
    graphs.append((graph,out))
for _ in range(50):
    for stream,(graph,out) in zip(streams,graphs):
        with torch.cuda.stream(stream):graph.replay()
torch.cuda.synchronize()
for i,(_,out) in enumerate(graphs):torch.testing.assert_close(out,refs[i],rtol=0,atol=0)
# Mutate values and scales without allocating replacement workspace.
for i,(a,b,fn) in enumerate(runs[:2]):
    replacement=api.quantize_mxfp8(torch.randn(a.rows,a.k,device='cuda',dtype=torch.bfloat16)*.125)
    a.values.copy_(replacement.values);a.scales.copy_(replacement.scales)
    replacement=api.quantize_mxfp8(torch.randn(b.rows,b.k,device='cuda',dtype=torch.bfloat16)*8)
    b.values.copy_(replacement.values);b.scales.copy_(replacement.scales)
    graph,out=graphs[i];graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(out,fn(),rtol=0,atol=0)
assert api.workspace_barriers_zero()
assert api.workspace_stats()['fallback_launches']==0
assert api.workspace_stats()['lanes']==3
assert mxfp6.workspace_stats()['frozen']==0
# Planned ordinary GEMM replay contains no allocation/reset GPU kernels.
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    graphs[0][0].replay();torch.cuda.synchronize()
events=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
assert len(events)==1,[e.name for e in events]
try:api.begin_workspace_planning()
except RuntimeError as e:assert 'already frozen' in str(e)
else:raise AssertionError('a frozen arena must not be resized')
print('PASS native pool, two concurrent graph lanes and independent backend pools')
"""
    env = dict(os.environ, MXFP6_AUTOTUNE="off")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=180)
