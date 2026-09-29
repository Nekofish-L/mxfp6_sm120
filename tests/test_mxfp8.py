"""GPU correctness tests for native W8A8, including non-tile-aligned M."""
import pytest
import torch
import mxfp6.mxfp8 as api
from mxfp6.mxfp8 import load_library


def operand(values, scales):
    return api.MXFP8Tensor(values.view(torch.uint8), scales, *values.shape)


def mm(a,b,sa,sb,*,tactic=None,splits=1,swizzle=1,sms=0,out=None,workspace=None):
    config=None if tactic is None else api.W8A8Config(tactic,splits,swizzle,sms)
    return api.gemm_packed(a,b,sa,sb,a.shape[0],b.shape[0],a.shape[1],config=config,out=out,workspace=workspace)


def prepare(a,b,sa,sb,*,tactic=None,splits=1,swizzle=1,sms=0,out=None):
    config=None if tactic is None else api.W8A8Config(tactic,splits,swizzle,sms)
    return api.prepare(operand(a,sa),operand(b,sb),config=config,out=out)


def allocate_workspace(a,b,sa,sb,*,tactic,splits=1,swizzle=1,sms=0,out=None):
    return api.allocate_workspace(operand(a,sa),operand(b,sb),config=api.W8A8Config(tactic,splits,swizzle,sms),out=out)



@pytest.mark.parametrize('entry', ['gemm', 'prepare'])
def test_first_call_loads_automatically(entry):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code=f'''
import sys, torch
# Model an unavailable optional package, including PyTorch's find_spec probe.
sys.modules['triton'] = None
sys.path.insert(0, {str(root / 'tests')!r})
from test_mxfp8 import quantize
import mxfp6.mxfp8 as impl
from mxfp6 import _loader
assert _loader._mxfp8.path is None
a,sa,ar=quantize(torch.randn(16,256,device='cuda'))
b,sb,br=quantize(torch.randn(256,256,device='cuda'))
qa=impl.MXFP8Tensor(a.view(torch.uint8),sa,16,256)
qb=impl.MXFP8Tensor(b.view(torch.uint8),sb,256,256)
value=impl.{entry}(qa,qb)
if callable(value):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): value()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=value()
    graph.replay()
else: out=value
assert _loader._mxfp8.path is not None
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
    for tactic in api.available_configs():
        if 550 <= tactic < 556 and k not in (2048, 2560):
            with pytest.raises(RuntimeError, match="requires K=2048 or K=2560"):
                mm(a,b,sa,sb,tactic=tactic)
            continue
        out=mm(a,b,sa,sb,tactic=tactic,splits=2 if tactic in range(12,17) else 1)
        error=((out.float()-reference).square().sum()/reference.square().sum()).sqrt()
        assert error < .004, (tactic,error)


@pytest.mark.parametrize('tactic', [13, 66, 72, 73, 78, 79, 200, 202, 203, 221, 222, 223, 313, 373, 427, 500, 502, 504, 520, 537, 560, 563, 566, 570, 572, 576, 600, 602, 606, 608, 611, 621, 624, 628, 665, 666, 671, 700, 706, 710, 711, 720, 730, 735, 740, 743, 747, 751, 760, 774, 775, 780, 785, 786, 791, 802, 813, 827, 913, 1014, 1120, 1122, 1125, 1131, 1142, 1145, 1148, 1153, 1162, 1165, 1168, 1173, 1180, 1188, 1192, 1197, 1203, 1344, 1358, 1366, 1390, 1458, 1460, 1569, 1595, 1220, 1225, 1230, 1231])
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


@pytest.mark.parametrize("tactic", [0, 760, 780, 813, 1344])
def test_invalid_inputs(tactic):
    load_library()
    a,sa,_=quantize(torch.randn(16,128,device='cuda'))
    b,sb,_=quantize(torch.randn(128,128,device='cuda'))
    with pytest.raises(RuntimeError,match='scales'):
        mm(a,b,sa[:1],sb,tactic=tactic)
    with pytest.raises(RuntimeError,match='Unknown'):
        mm(a,b,sa,sb,tactic=999)
    with pytest.raises(RuntimeError,match='Unknown'):
        mm(a,b,sa,sb,tactic=100)


@pytest.mark.parametrize("tactic", [13, 813, 913, 1013])
def test_private_workspaces_on_two_streams(tactic):
    load_library()
    runs=[]
    for seed in [72,91]:
        torch.manual_seed(seed)
        a,sa,ar=quantize(torch.randn(16,1024,device='cuda'))
        b,sb,br=quantize(torch.randn(256,1024,device='cuda'))
        run=prepare(a,b,sa,sb,tactic=tactic,splits=4)
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


@pytest.mark.parametrize('tactic', [2, 13, 27, 33, 38, 43, 49, 66, 72, 73, 78, 79, 200, 202, 203, 221, 222, 223, 313, 373, 427, 500, 502, 504, 520, 537, 560, 563, 566, 570, 572, 576, 600, 602, 606, 608, 611, 621, 624, 628, 665, 666, 671, 700, 706, 710, 711, 720, 730, 735, 740, 743, 747, 751, 760, 774, 775, 780, 785, 786, 791, 802, 813, 827, 913, 1014, 1120, 1122, 1125, 1131, 1142, 1145, 1148, 1153, 1162, 1165, 1168, 1173, 1180, 1188, 1192, 1197, 1203, 1344, 1358, 1366, 1390, 1458, 1460, 1569, 1595, 1220, 1225, 1230, 1231])
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
    load_library()
    a,sa,ar=quantize(torch.randn(16,2048,device='cuda'))
    b,sb,br=quantize(torch.randn(256,2048,device='cuda'))
    run=prepare(a,b,sa,sb,tactic=13,sms=sms)
    ref=ar@br.T
    for _ in range(3):
        assert ((run().float()-ref).square().sum()/ref.square().sum()).sqrt() < .004


@pytest.mark.parametrize('k', [2048, 2560])
@pytest.mark.parametrize('m', [14, 33])
def test_static_shape_graph_changes_both_inputs(m, k):
    load_library()
    a,sa,_=quantize(torch.randn(m,k,device='cuda'))
    b,sb,_=quantize(torch.randn(256,k,device='cuda'))
    scale=torch.exp2(torch.arange(k//32,device='cuda').float()%16-8).repeat_interleave(32)
    for tactic in range(550,556):
        out=torch.empty((m,256),device='cuda',dtype=torch.bfloat16)
        mm(a,b,sa,sb,out=out,tactic=tactic)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mm(a,b,sa,sb,out=out,tactic=tactic)
        for _ in range(3):
            aa,ss,ar=quantize(torch.randn(m,k,device='cuda')*scale)
            bb,tt,br=quantize(torch.randn(256,k,device='cuda')/scale)
            a.copy_(aa);sa.copy_(ss);b.copy_(bb);sb.copy_(tt)
            graph.replay()
            ref=ar@br.T
            assert ((out.float()-ref).square().sum()/ref.square().sum()).sqrt()<.004


@pytest.mark.parametrize('m,n,k', [
    (3,128,128), (17,3072,1536), (9,1920,6400), (31,3456,5120),
    (81,2176,6400), (65,3584,8192), (97,7168,2176), (111,11776,2304), (113,11776,2304),
    (193,6784,2176), (33,16384,2304), (17,4224,3200), (2049,256,128),
])
def test_default_on_unseen_shapes(m,n,k):
    a,sa,_=quantize(torch.randn(m,k,device='cuda'))
    b,sb,_=quantize(torch.randn(n,k,device='cuda'))
    run=prepare(a,b,sa,sb)
    run()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=run()
    qa,qsa,ar=quantize(torch.randn(m,k,device='cuda')*.25)
    qb,qsb,br=quantize(torch.randn(n,k,device='cuda')*4)
    a.copy_(qa);sa.copy_(qsa);b.copy_(qb);sb.copy_(qsb)
    graph.replay()
    ref=ar@br.T
    assert ((out.float()-ref).square().sum()/ref.square().sum()).sqrt().item()<.004
    a.zero_();graph.replay()
    assert torch.count_nonzero(out).item()==0
