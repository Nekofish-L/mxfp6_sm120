"""Check model weighting and the per-shape (not whole-model) oracle."""
import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'benchmarks/model_gemm_totals.py'
spec=importlib.util.spec_from_file_location('model_gemm_totals',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_weighted_oracle_chooses_each_shape_independently():
    geometry={'models':[{'model':'example','gemms_per_forward':5,'projections':[
        {'name':'first','n':128,'k':256,'count':2},
        {'name':'second','n':256,'k':128,'count':3}]}]}
    rows=[dict(m=1,n=128,k=256,native_us=10,flashinfer_us=5,block_fp8_us=30),
          dict(m=1,n=256,k=128,native_us=20,flashinfer_us=30,block_fp8_us=10)]
    totals,details=module.calculate(rows,geometry)
    total=totals[0]
    assert (total['native_us'],total['flashinfer_us'],total['block_fp8_us'])==(80,100,90)
    assert total['speedup_vs_flashinfer']==1.25
    assert total['speedup_vs_vllm_block_fp8']==1.125
    assert total['oracle_us']==40
    assert total['gap_us']==40
    assert total['gap_pct_vs_oracle']==100
    assert total['max_reduction_pct']==50
    assert sum(row['gap_weighted_us'] for row in details)==total['gap_us']
    rows[0]['native_us']=2
    assert module.calculate(rows,geometry)[0][0]['oracle_us']==34
    with pytest.raises(ValueError,match='Missing measurement'):
        module.calculate(rows[:1],geometry)
    with pytest.raises(ValueError,match='Duplicate'):
        module.calculate(rows+rows,geometry)


def test_checkpoint_projection_multiplicities_and_shared_output_shape():
    geometry=json.loads(module.GEOMETRY.read_text())
    for model,layers,linear,full,calls in zip(geometry['models'],[24,32],[18,24],[6,8],[96,128]):
        assert model['layers']==layers
        assert model['linear_attention_layers']==linear
        assert model['full_attention_layers']==full
        projections=model['projections']
        assert [p['count'] for p in projections]==[linear,linear,full,full,layers,layers]
        assert sum(p['count'] for p in projections)==calls
        assert (projections[1]['n'],projections[1]['k'])==(projections[3]['n'],projections[3]['k'])
        assert len({(p['n'],p['k']) for p in projections})==5
