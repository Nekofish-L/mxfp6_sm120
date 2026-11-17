"""Migration contracts for shared AOT/JIT configuration descriptions."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from mxfp6._mxfp8_config import SOURCE, configurations, kernel_expression, load_manifest

ROOT = Path(__file__).resolve().parents[1]
generate = runpy.run_path(str(ROOT/'scripts/generate_mxfp8_configs.py'))['generate']
REGISTRY = ROOT/'csrc/include/mxfp8_gemm/registry.def'


def test_legacy_instantiations_and_launch_resources():
    # Captured independently from the six handwritten adapters at 3f54d19.
    # Covers all 88 IDs, both fixed K variants and two dynamic K shapes.
    configs = list(configurations(load_manifest()))
    assert len(configs) == 88
    records = [[c['tactic'], k, kernel_expression(c, k), c['threads'], c['shared_bytes']]
               for c in configs for k in (128, 2048, 2560, 3072)]
    digest = hashlib.sha256(json.dumps(records, separators=(',', ':')).encode()).hexdigest()
    assert digest == 'ddc1995ad24e4f9f7a070e411c16143a0f00c9e8545d77fb907f6848a6978649'


@pytest.mark.parametrize('mutation,match', [
    (lambda d: d['families'][0]['configs'][1].__setitem__(0, 0), 'IDs'),
    (lambda d: d['families'][0]['configs'][0].__setitem__(5, 1), 'booleans'),
    (lambda d: d['families'][0].__setitem__('fixed_k', [1024, 0]), 'specialization'),
    (lambda d: d['families'][0].__setitem__('threads', 96), 'thread count'),
    (lambda d: d['families'][0].__setitem__('shared_memory', 'unknown'), 'shared memory'),
    (lambda d: d['families'][1].__setitem__('first_tactic', 200), 'Overlapping'),
])
def test_invalid_descriptors_fail_before_compilation(tmp_path, mutation, match):
    data = json.loads(SOURCE.read_text())
    mutation(data)
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_manifest(path)


def test_native_registry_range_must_match(tmp_path):
    registry = tmp_path/'registry.def'
    registry.write_text(REGISTRY.read_text().replace('cute, 200, 28', 'cute, 200, 27'))
    with pytest.raises(ValueError, match='range disagrees'):
        generate(SOURCE, registry)


def test_every_family_has_an_adapter():
    outputs = generate(SOURCE, REGISTRY)
    assert set(outputs) == {f['header']+'.cuh' for f in load_manifest()['families']}
    for family in load_manifest()['families']:
        output = outputs[family['header']+'.cuh']
        assert output.count('    case ') == len(family['configs'])
        assert 'mxfp8_common::validate(a, b, sa, sb, out)' in output
        assert 'c10::cuda::CUDAGuard guard(a.device())' in output


def test_incomplete_manifest_cannot_leave_stale_adapters(tmp_path):
    data = json.loads(SOURCE.read_text())
    data['families'].pop()
    path = tmp_path/'incomplete.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='six migrated family headers'):
        generate(path, REGISTRY)
