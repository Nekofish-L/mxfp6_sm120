"""Fixed-shape dual activation contract and changing-input graph components."""
import pytest
import torch
from mxfp6 import mxfp8_dual as dual

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 CUDA required",
)


def test_variants_and_input_errors():
    assert [dual.select_variant(*s) for s in dual.SUPPORTED_SHAPES] == [
        "mlp32", "qkvz32", "qkvz64_pipereg_cfg2"
    ]
    for shape in ((16, 12288, 2560), (64, 18432, 2560), (32, 12288, 128)):
        with pytest.raises(ValueError, match="Unsupported dual"):
            dual.select_variant(*shape)
    with pytest.raises(TypeError):
        dual.quantize(None)
    with pytest.raises(ValueError, match="CUDA BF16"):
        dual.quantize(torch.empty(32, 2560, dtype=torch.bfloat16))


def pack_scales(logical):
    m, groups = logical.shape
    packed = torch.zeros(((m + 127) // 128) * 128 * groups,
                         dtype=torch.uint8, device=logical.device)
    row = torch.arange(m, device=logical.device)[:, None]
    group = torch.arange(groups, device=logical.device)[None, :]
    offset = (row // 128 * (groups // 4) + group // 4) * 512
    offset = offset + row % 32 * 16 + row % 128 // 32 * 4 + group % 4
    packed[offset] = logical
    return packed


def reference_limb(x):
    """Frexp exponent oracle, independent of the Triton log2 implementation."""
    m, k = x.shape
    groups = x.float().reshape(m, k // 32, 32)
    maximum = groups.abs().amax(-1)
    raw = (maximum * (1.0 / 448.0)).double()
    fraction, exponent = torch.frexp(raw)
    exponent = exponent - (fraction == .5).int()
    encoded = (exponent + 127).clamp(0, 254)
    encoded = torch.where(maximum == 0, 0, encoded)
    scales = torch.exp2(encoded.float() - 127)
    scales = torch.where(maximum == 0, 1., scales)
    values = (groups / scales[..., None]).to(torch.float8_e4m3fn).reshape(m, k)
    reconstructed = (values.float().reshape(m, k // 32, 32)
                     * scales[..., None]).reshape(m, k)
    return values, pack_scales(encoded.to(torch.uint8)), reconstructed


def reference_dual(x):
    hi, s_hi, decoded = reference_limb(x)
    residual = (x.float() - decoded).to(torch.bfloat16)
    res, s_res, _ = reference_limb(residual)
    return hi, s_hi, res, s_res


@CUDA
@pytest.mark.parametrize("m", [32, 64])
def test_quantization_eager_and_poisoned_graph(m):
    torch.manual_seed(120 + m)
    x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
    outputs = dual.quantize(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dual.quantize_out(x, *outputs)
    for factor in (0., 1.e-5, 1., 100.):
        x.normal_().mul_(factor)
        for tensor in outputs:
            tensor.view(torch.uint8).fill_(255)
        graph.replay()
        expected = reference_dual(x)
        eager = dual.quantize(x)
        for actual, reference, immediate in zip(outputs, expected, eager):
            assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
            assert torch.equal(actual.view(torch.uint8), immediate.view(torch.uint8))
    with pytest.raises(ValueError, match="alias"):
        dual.quantize_out(x, outputs[0], outputs[1], outputs[0], outputs[3])


@CUDA
@pytest.mark.parametrize("m,n,k", dual.SUPPORTED_SHAPES)
def test_gemm_eager_and_changing_input_graph(m, n, k):
    torch.manual_seed(120 + m + n)
    # Unit E8M0 weight scales make this independent of the existing quantizer.
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    s_weight = torch.full((n * (k // 32),), 127, device="cuda", dtype=torch.uint8)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    limbs = dual.quantize(x)
    hi, s_hi, res, s_res = limbs
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    dual.gemm_out(hi, weight, s_hi, s_weight, res, s_res, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dual.quantize_out(x, *limbs)
        dual.gemm_out(hi, weight, s_hi, s_weight, res, s_res, out)
    for factor in (0., 1., 10.):
        x.normal_().mul_(factor)
        out.fill_(float("nan"))
        for tensor in limbs:
            tensor.view(torch.uint8).fill_(255)
        graph.replay()
        eager = torch.empty_like(out)
        dual.gemm_out(hi, weight, s_hi, s_weight, res, s_res, eager)
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
        _, _, hi_decoded = reference_limb(x)
        _, _, res_decoded = reference_limb((x.float() - hi_decoded).to(torch.bfloat16))
        # FP64 mathematical oracle bounds BF16 output rounding. It does not
        # claim the MMA's FP32 operation order or prototype bitwise identity.
        expected = (hi_decoded.double() @ weight.double().T
                    + res_decoded.double() @ weight.double().T)
        torch.testing.assert_close(out.double(), expected, rtol=.004, atol=.02)
