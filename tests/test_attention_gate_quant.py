"""Exact gate rounding, independent packed-scale oracle and poisoned replay."""
import pytest
import torch
from mxfp6.attention import sigmoid_gate_mxfp8
from test_quantization_padding import reference


@pytest.mark.parametrize('m', [1, 2, 4, 8, 16, 24, 32])
@pytest.mark.parametrize('strided', [False, True])
def test_gate_replay(m, strided):
    torch.manual_seed(1709 + m)
    x = torch.empty(m, 3072, device='cuda', dtype=torch.bfloat16)
    storage = torch.empty(m, 7168 if strided else 3072, device='cuda', dtype=x.dtype)
    gate = storage[:, -3072:]
    rounded = torch.empty_like(x)
    x.zero_(); storage.zero_()
    sigmoid_gate_mxfp8(x, gate, rounded_out=rounded)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = sigmoid_gate_mxfp8(x, gate, rounded_out=rounded)
        qp, sp = sigmoid_gate_mxfp8(x, gate)
    for i in range(120):
        x.normal_().mul_((0., 1.e-20, 1.e-5, 1., 100., 1000.)[i % 6])
        storage.normal_().mul_((1., 10., 100.)[i % 3])
        q.fill_(255); s.fill_(213); qp.fill_(255); sp.fill_(213)
        rounded.fill_(float('nan'))
        graph.replay()
        expected = x * torch.sigmoid(gate)
        assert torch.equal(rounded.view(torch.int16), expected.view(torch.int16))
        oq, os = reference(expected)
        assert torch.equal(q, oq) and torch.equal(s, os)
        assert torch.equal(qp, oq) and torch.equal(sp, os)
        torch.testing.assert_close(rounded.double(), x.double() * torch.sigmoid(gate.double()),
                                   atol=1.e-34, rtol=.008)


def test_all_finite_bf16_gate_values():
    # Exhaustive gate values include sigmoid rounding thresholds and saturation.
    values = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    x = torch.ones(32, 3072, device='cuda', dtype=torch.bfloat16)
    gate = torch.zeros_like(x)
    gate.flatten()[:values.numel()] = values
    rounded = torch.empty_like(x)
    sigmoid_gate_mxfp8(x, gate, rounded_out=rounded)
    assert torch.equal(rounded, torch.sigmoid(gate))


@pytest.mark.parametrize('m', [1, 2, 4, 8, 16, 24, 32])
def test_projection_replay(m):
    import mxfp6

    torch.manual_seed(900 + m)
    mxfp6.load_library()
    x = torch.randn(m, 3072, device='cuda', dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = mxfp6.quantize_mxfp6(torch.randn(5120, 3072, device='cuda', dtype=x.dtype))

    def fused():
        q, scales = sigmoid_gate_mxfp8(x, gate)
        return torch.ops.mxfp6.gemm_w6a8_pdl(q, weight.values, scales, weight.scales,
                                           m, 5120, 3072, 1., x.dtype)

    fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused()
    for _ in range(120):
        x.normal_(); gate.normal_()
        graph.replay()
        expected = mxfp6.gemm_from_float(x * torch.sigmoid(gate), weight, out_dtype=x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all_finite_bf16_attention_values():
    values = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    x = torch.zeros(32, 3072, device='cuda', dtype=torch.bfloat16)
    x.flatten()[:values.numel()] = values
    gate = torch.empty_like(x)
    rounded = torch.empty_like(x)
    for value in (-100., -1., 0., 1., 100.):
        gate.fill_(value)
        sigmoid_gate_mxfp8(x, gate, rounded_out=rounded)
        expected = x * torch.sigmoid(gate)
        assert torch.equal(rounded.view(torch.int16), expected.view(torch.int16))
