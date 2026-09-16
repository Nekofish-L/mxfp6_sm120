"""Pin norm reduction order, quantization bytes, padding and graph lifetime."""
import pytest
import torch
import mxfp6
from mxfp6.gdn import gated_norm_mxfp8
from test_quantization_padding import reference as quant_oracle
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd


@pytest.mark.parametrize('m', [1,2,4,8,16,24,32])
@pytest.mark.parametrize('dtype', [torch.bfloat16,torch.float32])
@pytest.mark.parametrize('strided', [False,True])
def test_exact_producer(m,dtype,strided):
    torch.manual_seed(121909 + m + 17*strided)
    x=torch.empty(m,24,128,device='cuda',dtype=torch.bfloat16)
    storage=torch.empty(m,8192 if strided else 3072,device='cuda',dtype=torch.bfloat16)
    z=storage[:,-3072:].view(m,24,128)
    w=torch.randn(128,device='cuda',dtype=dtype)
    rounded=torch.empty_like(x)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x.zero_();storage.zero_()
        gated_norm_mxfp8(x,z,w,rounded_out=rounded)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            q,s=gated_norm_mxfp8(x,z,w,rounded_out=rounded)
            qp,sp=gated_norm_mxfp8(x,z,w)  # production has no BF16 write
        for i in range(120):
            factor=(0.,1.e-20,1.e-5,1.,10.,100.)[i%6]
            x.normal_().mul_(factor);storage.normal_().mul_(10. if i%6==5 else 1.)
            if i%17==0:
                x.flatten()[::2] = -0.
            q.fill_(255);s.fill_(213);qp.fill_(255);sp.fill_(213);rounded.fill_(float('nan'))
            graph.replay()
            expected=layer_norm_fwd(x.reshape(-1,128),w,None,1.e-6,
                z=z.reshape(-1,128),is_rms_norm=True,activation='silu')[0].view_as(x)
            assert torch.equal(rounded.view(torch.int16),expected.view(torch.int16))
            oq,os=quant_oracle(expected.reshape(m,3072))
            assert torch.equal(q,oq) and torch.equal(s,os)
            assert torch.equal(qp,oq) and torch.equal(sp,os)
            xd,zd=x.double(),z.double()
            oracle=xd*torch.rsqrt(xd.square().mean(-1,keepdim=True)+1.e-6)*w.double()*(zd*torch.sigmoid(zd))
            torch.testing.assert_close(rounded.double(),oracle,rtol=.004,atol=1.e-7)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize('m', [1,2,4,8,16,24,32])
def test_fused_projection(m):
    torch.manual_seed(913+m)
    mxfp6.load_library()
    x=torch.randn(m,24,128,device='cuda',dtype=torch.bfloat16)
    storage=torch.randn(m,8192,device='cuda',dtype=torch.bfloat16)
    z=storage[:,5120:].view_as(x)
    w=torch.randn(128,device='cuda',dtype=torch.bfloat16)
    packed=mxfp6.quantize_mxfp6(torch.randn(5120,3072,device='cuda',dtype=torch.bfloat16))
    def fused():
        return mxfp6.gemm_from_gdn(x,z,w,packed.values.reshape(5120,2304),packed.scales)
    # Use public float path as comparator, with the same BF16 norm boundary.
    def original():
        y=layer_norm_fwd(x.reshape(-1,128),w,None,1.e-6,
            z=z.reshape(-1,128),is_rms_norm=True,activation='silu')[0]
        return mxfp6.gemm_from_float(y.reshape(m,3072),packed,out_dtype=torch.bfloat16)
    fused();original()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual=fused()
    for i in range(100):
        x.normal_();storage.normal_()
        graph.replay()
        torch.testing.assert_close(actual,original(),rtol=0,atol=0)
