"""GPU correctness tests for native W8A8, including non-tile-aligned M."""
import pytest
import torch
from mxfp6.mxfp8 import TACTICS, mm, load_library, allocate_workspace


@pytest.mark.parametrize('entry', ['gemm_mxfp8', 'prepare_mxfp8'])
def test_first_call_loads_automatically(entry):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code=f'''
import sys, torch
sys.path.insert(0, {str(root / 'tests')!r})
from test_mxfp8 import quantize
from mxfp6 import {entry}
import mxfp6.mxfp8 as impl
assert not impl._LOADED
a,sa,ar=quantize(torch.randn(16,256,device='cuda'))
b,sb,br=quantize(torch.randn(256,256,device='cuda'))
value={entry}(a,b,sa,sb,tactic=100)
if callable(value):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): value()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=value()
    graph.replay()
else: out=value
assert impl._LOADED
ref=ar@br.T
assert ((out.float()-ref).square().sum()/ref.square().sum()).sqrt() < .004
'''
    env=dict(os.environ)
    env['PYTHONPATH']=str(root/'python')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-c',code],env=env,check=True,
                   capture_output=True,text=True,timeout=180)


def quantize(x):
    """Independent E4M3/E8M0 quantizer with CUTLASS 128x4 scale layout."""
    m,k=x.shape
    groups=x.float().reshape(m,k//32,32)
    exponent=torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(2.0**-120)/448)).clamp(-127,127)
    scale=torch.exp2(exponent)
    q=(groups/scale[...,None]).to(torch.float8_e4m3fn).reshape(m,k)
    padded=torch.zeros(((m+127)//128*128,k//32),device=x.device,dtype=torch.uint8)
    padded[:m]=(exponent+127).to(torch.uint8)
    sf=padded.reshape(-1,4,32,k//128,4).transpose(1,3).contiguous().reshape(-1)
    reference=(q.float().reshape(m,k//32,32)*scale[...,None]).reshape(m,k)
    return q,sf,reference


@pytest.mark.parametrize('m,n,k',[(1,256,128),(14,256,512),(33,384,256),(129,256,384),(256,128,1024)])
def test_exact_quantized_input(m,n,k):
    load_library()
    torch.manual_seed(45)
    a,sa,ar=quantize(torch.randn(m,k,device='cuda')*.2)
    b,sb,br=quantize(torch.randn(n,k,device='cuda')*3)
    reference=ar@br.T
    for tactic in TACTICS:
        out=mm(a,b,sa,sb,tactic=tactic,splits=2 if tactic in range(12,17) else 1)
        error=((out.float()-reference).square().sum()/reference.square().sum()).sqrt()
        assert error < .004, (tactic,error)


@pytest.mark.parametrize('tactic', [13, 66, 100, 104])
def test_graph_replays_new_data(tactic):
    load_library()
    a,sa,_=quantize(torch.randn(16,256,device='cuda'))
    b,sb,br=quantize(torch.randn(256,256,device='cuda'))
    out=torch.empty((16,256),device='cuda',dtype=torch.bfloat16)
    workspace=allocate_workspace(a,b,sa,sb,tactic=tactic,splits=2,out=out)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mm(a,b,sa,sb,out=out,tactic=tactic,splits=2,workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): mm(a,b,sa,sb,out=out,tactic=tactic,splits=2,workspace=workspace)
    for _ in range(4):
        new_a,new_sa,ar=quantize(torch.randn(16,256,device='cuda'))
        a.copy_(new_a);sa.copy_(new_sa)
        graph.replay()
        reference=ar@br.T
        assert ((out.float()-reference).square().sum()/reference.square().sum()).sqrt() < .004


def test_invalid_inputs():
    load_library()
    a,sa,_=quantize(torch.randn(16,128,device='cuda'))
    b,sb,_=quantize(torch.randn(128,128,device='cuda'))
    with pytest.raises(RuntimeError,match='scales'):
        mm(a,b,sa[:1],sb)
    with pytest.raises(ValueError,match='Unknown'):
        mm(a,b,sa,sb,tactic=999)


def test_private_workspaces_on_two_streams():
    from mxfp6.mxfp8 import prepare
    load_library()
    runs=[]
    for seed in [72,91]:
        torch.manual_seed(seed)
        a,sa,ar=quantize(torch.randn(16,1024,device='cuda'))
        b,sb,br=quantize(torch.randn(256,1024,device='cuda'))
        run=prepare(a,b,sa,sb,tactic=13,splits=4)
        out=run()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        runs.append((stream,run,out,ar@br.T))
    for _ in range(20):
        for stream,run,_,_ in runs:
            with torch.cuda.stream(stream):run()
    for stream,_,out,ref in runs:
        stream.synchronize()
        error=((out.float()-ref).square().sum()/ref.square().sum()).sqrt()
        assert error < .004


@pytest.mark.parametrize('tactic', [2, 13, 27, 33, 38, 43, 49, 66, 100, 104])
def test_zero_and_varying_group_scales(tactic):
    load_library()
    torch.manual_seed(105)
    k=512
    scale=torch.exp2(torch.arange(k//32,device='cuda').float()-8).repeat_interleave(32)
    a,sa,ar=quantize(torch.randn(8,k,device='cuda')*scale)
    b,sb,br=quantize(torch.randn(256,k,device='cuda')/scale)
    ref=ar@br.T
    out=mm(a,b,sa,sb,tactic=tactic)
    assert ((out.float()-ref).square().sum()/ref.square().sum()).sqrt() < .004
    a.zero_()
    assert torch.count_nonzero(mm(a,b,sa,sb,tactic=tactic)) == 0


@pytest.mark.parametrize('sms',[64,86,128,144])
def test_stream_k_sm_count(sms):
    from mxfp6.mxfp8 import prepare
    load_library()
    a,sa,ar=quantize(torch.randn(16,2048,device='cuda'))
    b,sb,br=quantize(torch.randn(256,2048,device='cuda'))
    run=prepare(a,b,sa,sb,tactic=13,sms=sms)
    ref=ar@br.T
    for _ in range(3):
        assert ((run().float()-ref).square().sum()/ref.square().sum()).sqrt() < .004
