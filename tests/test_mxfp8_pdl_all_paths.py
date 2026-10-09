"""PDL coverage for every registered MXFP8 tactic and Python GEMM entry."""
import pytest
import torch

import mxfp6
import mxfp6.mxfp8 as api
from mxfp6 import mxfp8_dual as dual

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 CUDA required",
)


def programmatic_edges(graph):
    """Query captured CUDA dependencies independently of the launch wrapper."""
    import ctypes as ct

    class Edge(ct.Structure):
        _fields_ = [("from_port", ct.c_ubyte), ("to_port", ct.c_ubyte),
                    ("type", ct.c_ubyte), ("reserved", ct.c_ubyte * 5)]

    runtime = ct.CDLL("libcudart.so.13")
    query = runtime.cudaGraphGetEdges
    query.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p),
                      ct.POINTER(Edge), ct.POINTER(ct.c_size_t)]
    query.restype = ct.c_int
    count = ct.c_size_t()
    assert query(graph.raw_cuda_graph(), None, None, None, ct.byref(count)) == 0
    source = (ct.c_void_p * count.value)()
    target = (ct.c_void_p * count.value)()
    data = (Edge * count.value)()
    assert query(graph.raw_cuda_graph(), source, target, data, ct.byref(count)) == 0
    return sum(edge.type == 1 for edge in data)


@pytest.mark.skipif(not torch.version.cuda or int(torch.version.cuda.split(".")[0]) < 13,
                    reason="Graph edge introspection uses the CUDA 13 runtime ABI")
def test_every_tactic_pdl_graph_edges():
    """An ignored flag could pass numeric tests; verify actual PDL graph edges."""
    a = api.quantize_mxfp8(torch.randn(33, 2560, device="cuda", dtype=torch.bfloat16))
    b = api.quantize_mxfp8(torch.randn(256, 2560, device="cuda", dtype=torch.bfloat16))
    for tactic in api.available_configs():
        for enabled in (False, True):
            run = api.prepare(a, b, config=api.W8A8Config(tactic, 2, 1), use_pdl=enabled)
            run()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                # Use a kernel predecessor: memcpy/memset nodes cannot signal
                # a programmatic dependency through a kernel output port.
                a.values.add_(0)
                run()
            assert bool(programmatic_edges(graph)) == enabled, (tactic, enabled)

    for m in (1, 32, 33, 128):
        for n in (12288, 18432):
            x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
            hi, sh, residual, sr = dual.quantize(x)
            w = torch.randn(n, 2560, device="cuda").to(torch.float8_e4m3fn)
            sw = torch.full((n * 80,), 127, device="cuda", dtype=torch.uint8)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            for enabled in (False, True):
                dual.gemm_out(hi, w, sh, sw, residual, sr, out, use_pdl=enabled)
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph):
                    out.add_(0)
                    dual.gemm_out(hi, w, sh, sw, residual, sr, out, use_pdl=enabled)
                assert bool(programmatic_edges(graph)) == enabled, (m, n, enabled)


@pytest.mark.parametrize("k", (512, 2048, 2560))
def test_every_tactic_pdl_graph(k):
    """Explicit configs must honor PDL, including private Stream-K workspace."""
    mxfp6.load_library()
    api.load_library()
    torch.manual_seed(120)
    m, n = 33, 256  # Tail rows, dynamic K and both fixed-K portfolios.
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    activation = torch.empty_like(source)
    weight = api.quantize_mxfp8(torch.randn(n, k, device="cuda", dtype=source.dtype))
    values, scales = torch.ops.mxfp6.quantize_mxfp8(source)
    a = api.MXFP8Tensor(values, scales, m, k)

    for tactic in api.available_configs():
        if 550 <= tactic < 556 and k not in (2048, 2560):
            continue  # This family requires one of the two fixed K shapes.
        config = api.W8A8Config(tactic, 2, 1)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        workspace = api.allocate_workspace(a, weight, config=config, out=out)

        def chain():
            activation.copy_(source)  # Ordinary predecessor of the PDL chain.
            q, sq = torch.ops.mxfp6.quantize_mxfp8_pdl(activation)
            result = api.gemm_packed(q, weight.values, sq, weight.scales,
                                    m, n, k, config=config, out=out,
                                    workspace=workspace, use_pdl=True)
            return result + 1  # Ordinary consumer must see completed stores.

        chain()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            downstream = chain()
        for amplitude in (0., .125, 8., 1.):
            source.normal_().mul_(amplitude)
            graph.replay()
            q, sq = torch.ops.mxfp6.quantize_mxfp8(source)
            expected = api.gemm_packed(q, weight.values, sq, weight.scales,
                                      m, n, k, config=config, workspace=workspace)
            torch.testing.assert_close(out, expected, rtol=0, atol=0,
                                       msg=f"PDL tactic {tactic}")
            torch.testing.assert_close(downstream, expected + 1, rtol=0, atol=0,
                                       msg=f"PDL consumer after tactic {tactic}")


@pytest.mark.parametrize("entry", ("gemm_float", "gemm_quantized", "gemm_w8a8",
                                    "gemm_from_float", "gemm_packed", "prepare"))
def test_python_pdl_entries(entry):
    source = torch.randn(33, 256, device="cuda", dtype=torch.bfloat16)
    activation = torch.empty_like(source)
    weight = api.quantize_mxfp8(torch.randn(256, 256, device="cuda", dtype=source.dtype))
    a = api.quantize_mxfp8(source)
    prepared = api.prepare(a, weight, use_pdl=True)

    def run():
        activation.copy_(source)
        if entry == "gemm_float":
            return api.gemm(activation, weight, use_pdl=True)
        if entry == "gemm_from_float":
            return api.gemm_from_float(activation, weight, use_pdl=True)
        if entry == "gemm_quantized":
            return api.gemm(a, weight, use_pdl=True)
        if entry == "gemm_w8a8":
            return api.gemm_w8a8(a, weight, use_pdl=True)
        if entry == "gemm_packed":
            return api.gemm_packed(a.values, weight.values, a.scales, weight.scales,
                                   a.rows, weight.rows, a.k, use_pdl=True)
        return prepared()

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for amplitude in (0., .125, 8., 1.):
        source.normal_().mul_(amplitude)
        quantized = api.quantize_mxfp8(source)
        a.values.copy_(quantized.values)
        a.scales.copy_(quantized.scales)
        graph.replay()
        expected = api.gemm(source if entry in ("gemm_float", "gemm_from_float") else a,
                            weight)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", (1, 31, 32, 33, 64, 65, 127, 128))
@pytest.mark.parametrize("n", (12288, 18432))
def test_dual_to_single_pdl_chain(m, n):
    """Dual must signal its successor and wait for its own producer."""
    x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
    activation = torch.empty_like(x)
    weight = torch.randn(n, 2560, device="cuda").to(torch.float8_e4m3fn)
    sw = torch.full((n * 80,), 127, device="cuda", dtype=torch.uint8)
    limbs = dual.quantize(x)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    next_weight = api.quantize_mxfp8(torch.randn(256, n, device="cuda", dtype=x.dtype))

    def chain(use_pdl):
        activation.copy_(x)
        dual.quantize_out(activation, *limbs)
        hi, sh, residual, sr = limbs
        dual.gemm_out(hi, weight, sh, sw, residual, sr, out, use_pdl=use_pdl)
        return api.gemm_from_float(out, next_weight, use_pdl=use_pdl)

    chain(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        downstream = chain(True)
    for amplitude in (0., .125, 8., 1.):
        x.normal_().mul_(amplitude)
        graph.replay()
        expected = chain(False)
        torch.testing.assert_close(downstream, expected, rtol=0, atol=0)
