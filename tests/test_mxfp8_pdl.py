"""PDL ordering across decode, prefill, and scheduler boundaries."""

import pytest
import torch

import mxfp6
import mxfp6.mxfp8 as api


# Small-batch families and larger-M static, split/Stream-K, and dynamic tiles.
SHAPES = [
    (1, 128, 128, 43),
    (1, 128, 4096, 12),
    (1, 5120, 128, 0),
    (1, 10240, 128, 1458),
    (16, 128, 8192, 13),
    (16, 5120, 128, 709),
    (16, 10240, 128, 563),
    (16, 16384, 128, 624),
    (32, 5120, 128, 666),
    (32, 10240, 128, 576),
    (32, 16384, 128, 665),
    (33, 256, 128, 4),
]
SHAPES += [
    (m, n, k, tactic)
    for m, n, _, tactic in SHAPES
    if tactic in (709, 563, 624, 666, 576, 665)
    for k in (2048, 2560)
]
SHAPES += [
    (33, 12288, 2560, 572),
    (48, 18432, 2560, 747),
    (64, 256, 8192, 14),
    (65, 256, 8192, 15),
    (96, 12288, 2560, 707),
    (128, 12288, 2560, 789),
    (129, 256, 128, 500),
    (192, 256, 8192, 51),
    (256, 12288, 2560, 775),
    (257, 18432, 2560, 379),
    (512, 256, 128, 775),
    (1024, 256, 128, 810),
    (2048, 256, 128, 781),
    (2049, 256, 128, 79),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,n,k,tactic", SHAPES)
def test_pdl_graph_changing_inputs(m, n, k, tactic, dtype):
    mxfp6.load_library()
    api.load_library()
    assert api.select_config(m, n, k).config_id == tactic
    source = torch.randn(m, k, device="cuda", dtype=dtype)
    activation = torch.empty_like(source)
    weight = api.quantize_mxfp8(torch.randn(n, k, device="cuda", dtype=dtype))
    b = weight.dequantized_values()
    ops = torch.ops.mxfp8_sm120

    def run():
        # An ordinary predecessor, then explicit and fused PDL chains, followed
        # by an ordinary consumer. This exercises both dependency boundaries.
        activation.copy_(source)
        values, scales = torch.ops.mxfp6.quantize_mxfp8_pdl(activation)
        a = values.view(m, k).view(torch.float8_e4m3fn)
        packed = ops.gemm_pdl(a, b, scales, weight.scales)
        fused = ops.gemm_from_float_pdl(activation, b, weight.scales)
        return values, scales, packed, fused, fused + 1

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _, _, eager_packed, eager_fused, eager_downstream = run()
        eager_expected = ops.gemm_from_float(source, b, weight.scales)
        for actual, reference in [
            (eager_packed, eager_expected),
            (eager_fused, eager_expected),
            (eager_downstream, eager_expected + 1),
        ]:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            values, scales, packed, fused, downstream = run()
    torch.cuda.current_stream().wait_stream(stream)

    for amplitude in [0, .125, 8, 1] * 3:
        source.copy_(torch.randn_like(source) * amplitude)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        expected = ops.gemm_from_float(source, b, weight.scales)
        expected_values, expected_scales = torch.ops.mxfp6.quantize_mxfp8(source)
        for actual, reference in [
            (values, expected_values),
            (scales, expected_scales),
            (packed, expected),
            (fused, expected),
            (downstream, expected + 1),
        ]:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(), (128,), (1, 1, 128)])
def test_pdl_quantizer_rejects_nonmatrix(shape):
    mxfp6.load_library()
    with pytest.raises(RuntimeError, match=r"shape \[M,K\]"):
        torch.ops.mxfp6.quantize_mxfp8_pdl(
            torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        )
