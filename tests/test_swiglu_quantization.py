"""SwiGLU producer oracle, including signed zero and finite saturation inputs."""
import pytest
import torch
import mxfp6
from test_quantization_padding import reference

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('m',[1,2,4,8,16,24,32])
@pytest.mark.parametrize('k',[96,128,3072,8704])
def test_fused_producer(m,k,dtype):
    torch.manual_seed(902+m+k)
    x=torch.randn(m,2*k,device='cuda',dtype=dtype)
    mxfp6.load_library()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops.mxfp6.silu_and_mul_mxfp8(x)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            values,scales=torch.ops.mxfp6.silu_and_mul_mxfp8(x)
        for factor in (0.,1e-5,1.,32. if dtype == torch.float16 else 100.):
            x.normal_().mul_(factor)
            values.fill_(255);scales.fill_(255)
            graph.replay()
            gate,up=x.double().chunk(2,-1)
            denominator=(1.0+torch.exp(-gate).float()).float()
            activated=(gate.float()/denominator).to(dtype)
            activated=(activated.double()*(up+0.0)).to(dtype)
            assert torch.isfinite(activated).all()
            v,s=reference(activated)
            torch.testing.assert_close(values,v,atol=0,rtol=0)
            torch.testing.assert_close(scales,s,atol=0,rtol=0)
    torch.cuda.current_stream().wait_stream(stream)
