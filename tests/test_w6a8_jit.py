"""Optional W6A8 JIT execution checks; enable with MXFP6_TEST_JIT=1."""
import gc
import os
import subprocess
import sys

import pytest
import torch
import mxfp6
from test_w6a8_prepared import operands, reference

GPU = pytest.mark.skipif(os.environ.get('MXFP6_TEST_JIT') != '1' or
                         not torch.cuda.is_available(), reason='requires optional JIT toolchain and SM120')


def test_default_import_does_not_load_jit():
    subprocess.run([sys.executable, '-c',
                    "import sys,mxfp6; assert 'mxfp6._w6a8_jit' not in sys.modules"], check=True)


@GPU
@pytest.mark.parametrize('config_id', range(30))
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('pdl', [False, True])
def test_configs(config_id, dtype, pdl):
    a, b = operands()
    config = mxfp6.W6A8Config(config_id, 2, config_id % 3)
    run = mxfp6.prepare_w6a8(a, b, config=config, out_dtype=dtype, pdl=pdl, backend='jit')
    assert run.backend == 'jit'
    expected = reference(a, b, config, dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        a.values.add_(0)
        captured = run()
    if int(torch.version.cuda.split('.')[0]) >= 13:
        from test_mxfp8_pdl_all_paths import programmatic_edges
        assert bool(programmatic_edges(graph)) == pdl
    del run
    gc.collect()
    for _ in range(3):
        graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    a.values.zero_()
    for _ in range(3):
        graph.replay()
    assert torch.count_nonzero(captured) == 0


@GPU
@pytest.mark.parametrize('config_id', [2, 3, 16])
def test_dynamic_shape_cache_and_storage(config_id):
    runs = []
    for m, k in ((1, 128), (17, 2048), (513, 2560)):
        a, b = operands(m=m, k=k)
        config = mxfp6.W6A8Config(config_id, 1, 0)
        run = mxfp6.prepare_w6a8(a, b, alpha=-.25, config=config, backend='jit')
        expected = reference(a, b, config, torch.float16, alpha=-.25)
        torch.testing.assert_close(run(), expected, rtol=0, atol=0)
        runs.append(run)
    assert len({run._compiled.cubin() for run in runs}) == 1
    a.values.set_(a.values.clone())
    with pytest.raises(RuntimeError, match='must remain unchanged'):
        run()


@GPU
def test_reject_mismatched_handle():
    a, b = operands()
    run = mxfp6.prepare_w6a8(a, b, config=mxfp6.W6A8Config(2, 1, 0), backend='jit')
    other = mxfp6.prepare_w6a8(a, b, config=mxfp6.W6A8Config(16, 1, 0))
    with pytest.raises(RuntimeError, match='do not match'):
        run._compiled.bind(other._native)


@GPU
def test_independent_preparations_and_devices():
    from concurrent.futures import ThreadPoolExecutor
    a, b = operands()
    config = mxfp6.W6A8Config(2, 1, 0)
    def prepare(_):
        return mxfp6.prepare_w6a8(a, b, config=config, backend='jit')
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(prepare, range(4)))
    assert len({run._compiled.cubin() for run in runs}) == 1
    outputs = [run() for run in runs]
    assert len({output.data_ptr() for output in outputs}) == 4
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
    if torch.cuda.device_count() > 1:
        with torch.cuda.device(1):
            a1, b1 = operands()
            other = mxfp6.prepare_w6a8(a1, b1, config=config, backend='jit')
            expected = reference(a1, b1, config, torch.float16)
        device = torch.cuda.current_device()
        torch.testing.assert_close(other(), expected, rtol=0, atol=0)
        assert torch.cuda.current_device() == device
        with pytest.raises(RuntimeError, match='do not match'):
            other._compiled.bind(runs[0]._native)
