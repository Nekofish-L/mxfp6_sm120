"""Workspace allocation and graph lifetime contracts, using isolated pools."""

import json
import subprocess
import sys

import pytest
import torch
import mxfp6.mxfp8 as api


@pytest.mark.parametrize("tactic", [79, 16])
def test_workspace_query_does_not_allocate_output(tactic):
    a = api.quantize_mxfp8(torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16))
    b = api.quantize_mxfp8(torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16))
    out = torch.empty((512, 8192), device="cuda", dtype=torch.bfloat16)
    config = api.W8A8Config(tactic, 4)
    expected = api.allocate_workspace(a, b, config=config, out=out)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    workspace = api.allocate_workspace(a, b, config=config)
    if expected is None:
        assert workspace is None
        assert torch.cuda.max_memory_allocated() == before
    else:
        assert workspace.numel() == expected.numel()
        assert torch.cuda.max_memory_allocated() - before <= workspace.numel() + 512
    reference = api.gemm_w8a8(a, b, config=config)
    actual = api.gemm_w8a8(a, b, config=config, out=out, workspace=workspace)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("per_stream", [False, True])
@pytest.mark.parametrize("backend", ["mxfp6", "mxfp8"])
def test_capacity_and_graph_replay(per_stream, backend):
    code = r"""
import json,os,sys,torch,mxfp6
os.environ['MXFP6_AUTOTUNE']='0'
os.environ['MXFP6_STREAM_K']='1'
per_stream=bool(int(sys.argv[1]))
import mxfp6.mxfp8 as w8
is_w8=sys.argv[2]=='mxfp8'
api,other=(w8,mxfp6) if is_w8 else (mxfp6,w8)
streams = [torch.cuda.Stream(), torch.cuda.Stream()]
runs=[]
shapes=[(1,128,8192,12),(512,8192,2048,16)] if is_w8 else [(1,5120,8704,3),(672,5120,3072,16)]
for stream,(m,n,k,t) in zip(streams,shapes):
    with torch.cuda.stream(stream):
        a=api.quantize_mxfp8(torch.randn(m,k,device='cuda',dtype=torch.bfloat16))
        quantize_weight=api.quantize_mxfp8 if is_w8 else api.quantize_mxfp6
        b=quantize_weight(torch.randn(n,k,device='cuda',dtype=torch.bfloat16))
        if is_w8:
            out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            run=lambda a=a,b=b,out=out,t=t:api.gemm_w8a8(a,b,config=api.W8A8Config(t,4),out=out)
        else:
            run=lambda a=a,b=b:api.gemm_w6a8(a,b,out_dtype=torch.bfloat16)
        reference=run().clone()
        runs.append((a,run,reference))
torch.cuda.synchronize()
api.begin_workspace_planning()
assert other.workspace_stats()["planning"]==0
for stream,(_,run,_) in zip(streams,runs):
    with torch.cuda.stream(stream):run()
api.finalize_workspace_planning(per_stream=per_stream)
graphs=[]
for stream,(_,run,_) in zip(streams,runs):
    with torch.cuda.stream(stream):
        run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=run()
    graphs.append((graph,out))
for _ in range(20):
    for stream,(graph,out) in zip(streams,graphs):
        with torch.cuda.stream(stream):graph.replay()
torch.cuda.synchronize()
for (_,_,reference),(_,out) in zip(runs,graphs):torch.testing.assert_close(out,reference,rtol=0,atol=0)
for stream,(a,run,_),(graph,out) in zip(streams,runs,graphs):
    with torch.cuda.stream(stream):
        replacement=api.quantize_mxfp8(torch.randn(a.shape,device='cuda',dtype=torch.bfloat16)*.125)
        a.values.copy_(replacement.values);a.scales.copy_(replacement.scales)
        expected=run().clone();graph.replay()
    stream.synchronize()
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
assert api.workspace_barriers_zero()
stats=api.workspace_stats()
assert stats['fallback_launches']==0
if per_stream:
    try:runs[0][1]()
    except RuntimeError as e:assert 'not included' in str(e)
    else:raise AssertionError('unplanned stream accepted')
    with torch.cuda.stream(streams[0]):
        try:runs[1][1]()
        except RuntimeError as e:assert 'capacity' in str(e)
        else:raise AssertionError('oversized layout accepted')
assert other.workspace_stats()['frozen']==0
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    graphs[0][0].replay();torch.cuda.synchronize()
events=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
assert len(events)==1,[e.name for e in events]
try:api.begin_workspace_planning()
except RuntimeError as e:assert 'already frozen' in str(e)
else:raise AssertionError('frozen arena resized')
print(json.dumps(stats))
"""
    stats = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(int(per_stream)), backend], text=True, timeout=180
        )
    )
    assert stats["lanes"] == (2 if per_stream else 3)
    assert stats["per_stream"] == int(per_stream)
    if per_stream:
        assert stats["total_arena_bytes"] < 2 * stats["arena_bytes"]
    else:
        assert stats["total_arena_bytes"] == 3 * stats["arena_bytes"]
