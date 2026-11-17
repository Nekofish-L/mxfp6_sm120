"""GPU regression tests for fixed-storage W6A8 execution."""
import gc
import os

os.environ.setdefault("MXFP6_AUTOTUNE", "0")

import pytest
import torch
import mxfp6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires SM120",
)


def operands(m=33, n=512, k=2048):
    torch.manual_seed(71)
    a = mxfp6.quantize_activation(torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * .1)
    b = mxfp6.quantize_mxfp6(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .1)
    return a, b


def reference(a, b, config, dtype, alpha=1.):
    return torch.ops.mxfp6.gemm_w6a8_config(
        a.values, b.values, a.scales, b.scales, a.rows, b.rows, a.k,
        alpha, config.config_id, config.swizzle, config.raster_order, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("config_id", range(30))
@pytest.mark.parametrize("pdl", [False, True])
def test_all_configs(dtype, config_id, pdl):
    a, b = operands()
    config = mxfp6.W6A8Config(config_id, 2, config_id % 3)
    run = mxfp6.prepare_w6a8(a, b, config=config, out_dtype=dtype, pdl=pdl)
    expected = reference(a, b, config, dtype)
    pointer = run().data_ptr()
    for _ in range(3):
        actual = run()
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    a.values.zero_()
    for _ in range(3):
        graph.replay()
    assert torch.count_nonzero(captured) == 0


@pytest.mark.parametrize("config_id", [2, 3, 16])
def test_streams_and_lifetime(config_id):
    a, b = operands(513)
    config = mxfp6.W6A8Config(config_id, 1, 0)
    runs = [mxfp6.prepare_w6a8(a, b, config=config) for _ in range(2)]
    expected = reference(a, b, config, torch.float16)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    for run, stream in zip(runs, streams):
        with torch.cuda.stream(stream):
            outputs.append(run())
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runs[0]()
    del runs, a, b
    gc.collect()
    for _ in range(3):
        graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_storage_mutation_and_invalid_config():
    a, b = operands()
    run = mxfp6.prepare_w6a8(a, b, config=mxfp6.W6A8Config(2, 1, 0))
    a.values.set_(a.values.clone())
    with pytest.raises(RuntimeError, match="must remain unchanged"):
        run()
    for config in [mxfp6.W6A8Config(30, 1, 0), mxfp6.W6A8Config(2, 3, 0),
                   mxfp6.W6A8Config(2, 1, 3)]:
        with pytest.raises(RuntimeError):
            mxfp6.prepare_w6a8(a, b, config=config)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_and_stream_k_fallback(dtype):
    a, b = operands()
    anchor = a.values
    for config_id, disabled_id in ((3, 1), (16, 12), (2, 2)):
        torch.ops.mxfp6.set_w6a8_config(anchor, a.rows, b.rows, a.k,
                                      config_id, 4, 2, dtype)
        try:
            run = mxfp6.prepare_w6a8(a, b, out_dtype=dtype)
            expected_id = config_id if mxfp6.is_stream_k_enabled() else disabled_id
            assert run.config == mxfp6.W6A8Config(expected_id, 4, 2)
            torch.testing.assert_close(run(), mxfp6.gemm_w6a8(a, b, out_dtype=dtype),
                                       rtol=0, atol=0)
        finally:
            torch.ops.mxfp6.set_w6a8_config(anchor, a.rows, b.rows, a.k, -1, 1, 0, dtype)
