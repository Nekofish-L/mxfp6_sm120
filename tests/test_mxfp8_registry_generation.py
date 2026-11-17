import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('generate_mxfp8',ROOT/'scripts/generate_mxfp8_dispatch.py')
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
POLICY=ROOT/'python/mxfp6/mxfp8_dispatch.json'
REGISTRY=ROOT/'csrc/include/mxfp8_gemm/registry.def'


def test_full_registry_is_unchanged():
    assert generator.generate_registry(POLICY,REGISTRY,False)==REGISTRY.read_text()


def test_pruning_retains_every_referenced_family():
    unused=generator.unused_families(POLICY,REGISTRY)
    assert len(unused)==12
    for name in unused:
        assert (ROOT/f'csrc/mxfp8_{name}.cu').is_file()
        assert f'MXFP8_FAMILY({name},' not in generator.generate_registry(POLICY,REGISTRY,True)
    assert 'MXFP8_FAMILY(small_batch,' in generator.generate_registry(POLICY,REGISTRY,True)


def test_even_shadowed_leaves_keep_a_family(tmp_path):
    policy=json.loads(POLICY.read_text())
    policy['regions'].append({'configs':[[550,1,1,0]]})
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy))
    assert 'static_shape' not in generator.unused_families(path,REGISTRY)


def test_jit_pruning_keeps_non_migrated_and_default_families():
    manifest = ROOT/'python/mxfp6/mxfp8_cp_async.json'
    assert generator.unused_families(POLICY,REGISTRY,manifest) == ['cute','specialized']
    result = generator.generate_registry(POLICY,REGISTRY,True,manifest)
    for name in ('tma4','tma_oddwarp','small_batch','vector_store','eightwarp'):
        assert f'MXFP8_FAMILY({name},' in result
    assert 'MXFP8_FAMILY(cute,' not in result
    assert 'MXFP8_FAMILY(specialized,' not in result


def test_partial_jit_family_cannot_be_pruned(tmp_path):
    import pytest
    manifest = json.loads((ROOT/'python/mxfp6/mxfp8_cp_async.json').read_text())
    manifest['families'][0]['configs'].pop()
    path = tmp_path/'partial.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='JIT range disagrees'):
        generator.unused_families(POLICY,REGISTRY,path)
