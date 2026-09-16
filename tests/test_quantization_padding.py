"""Independent finite-value quantizer/scale-layout oracle and poisoned replay."""
import pytest
import torch
import mxfp6

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def reference(x):
    # Preserve the published FP32 reciprocal-multiply boundary, then compute
    # the exponent in FP64 independently of the CUDA integer-bit rounding.
    m, k = x.shape
    groups = x.double().reshape(m, k//32, 32)
    raw = torch.clamp(groups.abs().amax(-1).float() / 448, min=1e-30).double()
    mantissa, exponent = torch.frexp(raw)
    exponent = exponent - (mantissa == 0.5).int()
    scaled = groups / torch.pow(2.0, exponent[..., None])
    values = scaled.float().to(torch.float8_e4m3fn).view(torch.uint8).reshape(m, k)
    padded_groups = (k//32+3)//4*4
    packed = torch.full((((m+127)//128)*128*padded_groups,),127,device=x.device,dtype=torch.uint8)
    r = torch.arange(m,device=x.device)[:,None]
    g = torch.arange(k//32,device=x.device)[None,:]
    offsets = (r//128*(padded_groups//4)+g//4)*512+(r%32)*16+(r%128//32)*4+g%4
    packed[offsets] = (exponent+127).to(torch.uint8)
    return values,packed


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('m,k',[(m,k) for m in (1,2,4,8,16,24,32,127,128,129) for k in (32,96,128,3072,5120,8704)])
def test_poisoned_changing_input_graph(m,k,dtype):
    torch.manual_seed(24120+m+k)
    x=torch.randn(m,k,device='cuda',dtype=dtype)
    mxfp6.load_library()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mxfp6.quantize_mxfp8(x)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            actual=mxfp6.quantize_mxfp8(x)
        for factor in (0.0,1e-5,1.0,1000.0):
            x.normal_().mul_(factor)
            actual.values.view(torch.uint8).fill_(255)
            actual.scales.fill_(255)
            graph.replay()
            v,s=reference(x)
            torch.testing.assert_close(actual.values.view(torch.uint8),v,atol=0,rtol=0)
            torch.testing.assert_close(actual.scales,s,atol=0,rtol=0)
    torch.cuda.current_stream().wait_stream(stream)
