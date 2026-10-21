"""Unified decode/prefill dispatch, accuracy and changing-input graphs."""
import json
from pathlib import Path

import pytest
import torch

from mxfp6 import mxfp8
from test_mxfp8 import quantize


GEOMETRY = json.loads((Path(__file__).resolve().parents[1] /
                       "benchmarks/qwen35_tp1_projection_geometry.json").read_text())
SHAPES = sorted({(p["n"], p["k"]) for model in GEOMETRY["models"]
                 for p in model["projections"]})


@pytest.mark.parametrize("n,k,batches", [
    (n, k, range(1, 65)) for n, k in SHAPES
] + [
    (n, k, [*range(1, 65), 65, 95, 96, 127, 128, 129, 192, 256,
            257, 511, 512, 513, 1024, 1025, 2048, 2049, 4096, 8192])
    for n, k in [(1024, 2048), (1024, 3584), (5120, 1024),
                 (7168, 1024), (8192, 1024)]
])
def test_decode_and_prefill_batches(n, k, batches):
    torch.manual_seed(20261013)
    # Opposite exponents across K groups exercise the swizzled scale layout.
    scale = torch.exp2(torch.arange(k // 32, device="cuda") % 13 - 6).repeat_interleave(32)
    b, sb, br = quantize(torch.randn(n, k, device="cuda") / scale)
    qb = mxfp8.MXFP8Tensor(b.view(torch.uint8), sb, n, k)
    for m in batches:
        a, sa, ar = quantize(torch.randn(m, k, device="cuda") * scale)
        qa = mxfp8.MXFP8Tensor(a.view(torch.uint8), sa, m, k)
        fn = mxfp8.prepare(qa, qb)
        if m <= 32 and (n, k) in SHAPES:
            assert 1240 <= fn.config.config_id < 1300, (m, n, k, fn.config)
        actual = fn().float()
        reference = ar @ br.T
        error = ((actual - reference).square().sum() / reference.square().sum()).sqrt()
        assert error < .004, (m, n, k, fn.config, error)


def check_partial_tiles_graph(m, n, k, pdl, config=None):
    a, sa, _ = quantize(torch.randn(m, k, device="cuda"))
    b, sb, _ = quantize(torch.randn(n, k, device="cuda"))
    qa = mxfp8.MXFP8Tensor(a.view(torch.uint8), sa, m, k)
    qb = mxfp8.MXFP8Tensor(b.view(torch.uint8), sb, n, k)
    fn = mxfp8.prepare(qa, qb, use_pdl=pdl, config=config)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    for exponent in (-5, 4):
        na, nsa, ar = quantize(torch.randn(m, k, device="cuda") * 2.0 ** exponent)
        nb, nsb, br = quantize(torch.randn(n, k, device="cuda") * 2.0 ** -exponent)
        a.copy_(na)
        sa.copy_(nsa)
        b.copy_(nb)
        sb.copy_(nsb)
        output.fill_(float("nan"))
        graph.replay()
        reference = ar @ br.T
        error = ((output.float() - reference).square().sum() / reference.square().sum()).sqrt()
        assert error < .004
    a.zero_()
    output.fill_(float("nan"))
    graph.replay()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("m,n,k", [(3, 384, 128), (9, 2176, 384),
                                   (17, 4224, 2176), (31, 10368, 2688),
                                   (33, 2176, 384), (49, 10368, 2688),
                                   (63, 4224, 2176), (31, 12416, 2688),
                                   # Compact/short-K regions and prefill routes.
                                   (19, 1152, 1920), (71, 896, 3456),
                                   (99, 5504, 896), (257, 7296, 1152),
                                   (4097, 1024, 3584), (4097, 8192, 1024)])
@pytest.mark.parametrize("pdl", [False, True])
def test_partial_tiles_graph(m, n, k, pdl):
    check_partial_tiles_graph(m, n, k, pdl)


@pytest.mark.parametrize("tactic", (1249, 1250, 1251, 1252, 1253, 1254))
@pytest.mark.parametrize("pdl", [False, True])
def test_decode_kernel_graph(tactic, pdl):
    check_partial_tiles_graph(49, 384, 640, pdl,
                              mxfp8.W8A8Config(tactic, 1, 1, 0))
