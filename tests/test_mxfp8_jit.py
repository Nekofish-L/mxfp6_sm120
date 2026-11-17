"""Opt-in JIT contracts. GPU/compiler checks require MXFP8_TEST_JIT=1."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import mxfp6
from mxfp6 import mxfp8 as api

ROOT = Path(__file__).resolve().parents[1]
GPU = pytest.mark.skipif(os.environ.get('MXFP8_TEST_JIT') != '1' or not torch.cuda.is_available(),
                         reason='set MXFP8_TEST_JIT=1 on SM120 with the optional toolchain')


def test_default_import_does_not_load_jit():
    subprocess.run([sys.executable, '-c', "import sys; import mxfp6.mxfp8 as a; assert len(a.available_jit_configs()) == 88; assert 'mxfp6._mxfp8_jit' not in sys.modules"], check=True)


def test_source_dependencies_invalidate_signature(tmp_path):
    from mxfp6._mxfp8_jit_source import dependency_signature
    (tmp_path/'csrc').mkdir()
    (tmp_path/'csrc/mxfp8_jit.cpp').write_text('bridge')
    header = tmp_path/'third_party/cutlass/include/header.h'
    header.parent.mkdir(parents=True)
    header.write_text('before')
    before = dependency_signature(tmp_path)
    header.write_text('after')
    assert dependency_signature(tmp_path) != before


@pytest.mark.parametrize('config,k,error', [(None,128,TypeError), (api.W8A8Config(560,2),128,ValueError),
                                           (api.W8A8Config(560),127,ValueError)])
def test_invalid_requests_fail_before_cuda(config, k, error):
    with pytest.raises(error):
        api.precompile_jit(config, k)


def operands(m=32, n=512, k=2560, device='cuda'):
    return (mxfp6.quantize_activation(torch.randn(m,k,device=device,dtype=torch.bfloat16)),
            api.quantize_mxfp8(torch.randn(n,k,device=device,dtype=torch.bfloat16)))


@GPU
@pytest.mark.parametrize('tactic,m,k', [(200,32,128),(500,32,2560),(520,32,2048),(560,32,2560),
                                      (571,32,3072),(575,17,2560),(700,32,2560),(703,129,128)])
@pytest.mark.parametrize('pdl', [False,True])
def test_families_graph_and_stream(tactic, m, k, pdl):
    from test_mxfp8_pdl_all_paths import programmatic_edges
    a,b = operands(m=m,k=k)
    config = api.W8A8Config(tactic)
    run = api.prepare_jit(a,b,config=config,use_pdl=pdl)
    reference = api.prepare(a,b,config=config,use_pdl=pdl)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual,reference(),rtol=0,atol=0)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        a.values.add_(0)
        run()
    if int(torch.version.cuda.split('.')[0]) >= 13:
        assert bool(programmatic_edges(graph)) == pdl
    del run
    gc.collect()
    api.precompile_jit(api.W8A8Config(561),2560)
    graph.replay()
    torch.testing.assert_close(actual,reference(),rtol=0,atol=0)


@GPU
def test_thread_single_flight(monkeypatch):
    from mxfp6 import _mxfp8_jit as jit
    module = jit._context(ROOT)[0]
    original = module.compile_kernel
    calls = []
    def count(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, 'compile_kernel', count)
    with ThreadPoolExecutor(max_workers=4) as pool:
        handles = list(pool.map(lambda _: api.precompile_jit(api.W8A8Config(563),2560), range(4)))
    assert len(calls) == 1
    assert all(h is handles[0] for h in handles)


@GPU
def test_dynamic_k_reuses_handle_and_fixed_k_does_not():
    config = api.W8A8Config(560)
    dynamic = api.precompile_jit(config,128)
    assert dynamic is api.precompile_jit(config,3072)
    assert dynamic is not api.precompile_jit(config,2048)


@GPU
def test_multiple_handles_and_private_outputs():
    a,b = operands()
    first = api.prepare_jit(a,b,config=api.W8A8Config(560))
    second = api.prepare_jit(a,b,config=api.W8A8Config(571))
    assert first().data_ptr() != second().data_ptr()
    for run in (first,second):
        torch.testing.assert_close(run(),api.prepare(a,b,config=run.config)(),rtol=0,atol=0)


@GPU
def test_precompile_and_prepare_are_forbidden_during_capture():
    a,b = operands()
    config = api.W8A8Config(560)
    run = api.prepare_jit(a,b,config=config)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError,match='before CUDA Graph capture'):
            api.precompile_jit(config,2560)
        with pytest.raises(RuntimeError,match='before CUDA Graph capture'):
            api.prepare_jit(a,b,config=config)
        run()
    graph.replay()


@GPU
def test_output_validation_and_unknown_tactic():
    a,b = operands()
    with pytest.raises(RuntimeError,match='BF16 output'):
        api.prepare_jit(a,b,config=api.W8A8Config(560),out=torch.empty(32,512,device='cuda'))
    with pytest.raises(ValueError,match='not been migrated'):
        api.precompile_jit(api.W8A8Config(12),2560)


@GPU
def test_device_guard_and_per_device_handles():
    if torch.cuda.device_count() < 2:
        pytest.skip('two SM120 devices required')
    config = api.W8A8Config(560)
    first = api.precompile_jit(config,2560,device='cuda:0')
    a,b = operands(device='cuda:1')
    run = api.prepare_jit(a,b,config=config)
    assert run.jit_kernel is not first and run.jit_kernel.device == 1
    torch.cuda.set_device(0)
    torch.testing.assert_close(run(),api.prepare(a,b,config=config)(),rtol=0,atol=0)
    assert torch.cuda.current_device() == 0


@GPU
def test_process_single_flight_and_disk_hit(tmp_path):
    env = dict(os.environ, MXFP8_JIT_CACHE_DIR=str(tmp_path/'cache'), MXFP8_JIT_PRINT_COMPILER_COMMAND='1')
    code = "from mxfp6 import mxfp8 as a; print(a.precompile_jit(a.W8A8Config(565),2560).cubin)"
    workers = [subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
               for _ in range(2)]
    try:
        outputs = [p.communicate(timeout=180)[0] for p in workers]
        assert all(p.returncode == 0 for p in workers), outputs
        assert sum(s.count('--cubin') for s in outputs) == 1, outputs
        cubins = list((tmp_path/'cache/cache').glob('*/kernel.cubin'))
        assert len(cubins) == 1
        before = cubins[0].stat().st_mtime_ns
        hit = subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,check=True,timeout=120)
        assert '--cubin' not in hit.stdout + hit.stderr
        assert cubins[0].stat().st_mtime_ns == before
    finally:
        for p in workers:
            if p.poll() is None:
                p.kill()
                p.wait()


@GPU
def test_failed_compile_can_retry(monkeypatch):
    from mxfp6 import _mxfp8_jit as jit
    module = jit._context(ROOT)[0]
    original = module.compile_kernel
    def fail(*args):
        raise RuntimeError('intentional compile failure')
    monkeypatch.setattr(module,'compile_kernel',fail)
    with pytest.raises(RuntimeError,match='intentional compile failure'):
        api.precompile_jit(api.W8A8Config(566),2560)
    monkeypatch.setattr(module,'compile_kernel',original)
    assert api.precompile_jit(api.W8A8Config(566),2560).cubin


@GPU
def test_prepared_launch_never_enters_compiler_or_cache(monkeypatch):
    from mxfp6 import _mxfp8_jit as jit
    a,b = operands()
    run = api.prepare_jit(a,b,config=api.W8A8Config(560))
    reference = api.prepare(a,b,config=run.config)
    def fail(*args,**kwargs):
        raise AssertionError('compiler/cache accessed during prepared launch')
    for name in ('_context','precompile','device_source','file_lock'):
        monkeypatch.setattr(jit,name,fail)
    out = run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(out,reference(),rtol=0,atol=0)


@GPU
def test_backend_selection_preserves_tactic_and_propagates_failure(monkeypatch):
    from mxfp6 import _mxfp8_jit as jit
    a,b = operands()
    config = api.W8A8Config(560)
    native = api.prepare(a,b,config=config,backend='auto')
    assert native.backend == 'aot' and native.config == config
    selected = api.prepare(a,b,backend='auto')
    assert selected.config == api.select_config(a.rows,b.rows,a.k)
    forced = api.prepare(a,b,config=config,backend='jit')
    assert forced.backend == 'jit' and forced.config == config
    monkeypatch.setattr(api,'available_configs',lambda: ())
    fallback = api.prepare(a,b,config=config,backend='auto')
    assert fallback.backend == 'jit' and fallback.config == config
    torch.testing.assert_close(fallback(),native(),rtol=0,atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fallback()
    graph.replay()
    torch.testing.assert_close(actual,native(),rtol=0,atol=0)
    def fail(*args,**kwargs):
        raise RuntimeError('intentional backend failure')
    monkeypatch.setattr(jit,'prepare',fail)
    with pytest.raises(RuntimeError,match='intentional backend failure'):
        api.prepare(a,b,config=config,backend='auto')


@GPU
def test_backend_rejects_unmigrated_tactic_and_invalid_scheduler(monkeypatch):
    a,b = operands()
    monkeypatch.setattr(api,'available_configs',lambda: ())
    with pytest.raises(ValueError,match='not been migrated'):
        api.prepare(a,b,config=api.W8A8Config(12),backend='auto')
    with pytest.raises(ValueError,match='splits=1'):
        api.prepare(a,b,config=api.W8A8Config(560,2),backend='auto')


def test_invalid_backend_fails_before_cuda():
    with pytest.raises(ValueError,match='backend must'):
        api.prepare(None,None,backend='unknown')


@GPU
def test_auto_native_does_not_import_jit():
    subprocess.run([sys.executable, '-c', '''
import sys, torch
from mxfp6 import mxfp8 as a
x = a.quantize_mxfp8(torch.randn(32,2560,device='cuda',dtype=torch.bfloat16))
w = a.quantize_mxfp8(torch.randn(512,2560,device='cuda',dtype=torch.bfloat16))
run = a.prepare(x,w,config=a.W8A8Config(560),backend='auto')
assert run.backend == 'aot' and 'mxfp6._mxfp8_jit' not in sys.modules
run()
'''], check=True)


def test_backend_discovery_without_cuda_or_jit_import():
    subprocess.run([sys.executable, '-c', """
import sys
from mxfp6 import mxfp8 as a
assert a.available_configs(backend='jit') == a.available_jit_configs()
assert a.available_configs(backend='jit',stream_k_only=True) == ()
assert 'mxfp6._mxfp8_jit' not in sys.modules
try: a.available_configs(backend='invalid')
except ValueError: pass
else: raise AssertionError('invalid backend accepted')
"""],check=True)
