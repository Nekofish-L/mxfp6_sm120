"""Regression coverage for externally allocated CUDA graph buffers."""
import gc
import importlib.util
from pathlib import Path
import weakref
import torch


def test_capture_retains_prepared_workspace():
    path=Path(__file__).resolve().parents[1]/'benchmarks/benchmark_mxfp8.py'
    spec=importlib.util.spec_from_file_location('mxfp8_benchmark_lifetime',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def prepare(value):
        workspace=torch.full((131072,),value,device='cuda')
        output=torch.empty_like(workspace)
        def run():
            torch.add(workspace,2,out=output)
            return output
        return run

    graphs=[]
    refs=[]
    for value in [1.,3.,7.]:
        run=prepare(value)
        refs.append(weakref.ref(run))
        graph,out=module.capture(run)
        graphs.append((graph,out,value+2))
        del run
    gc.collect()
    assert all(ref() is not None for ref in refs)
    for graph,out,expected in graphs:
        graph.replay()
        assert torch.all(out==expected).item()
    del graph,out
    graphs.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)
