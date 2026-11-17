"""Build/JIT descriptions for the migrated cp.async families, not dispatch policy.

Tactics outside this manifest retain their existing implementation. Presence here
also does not imply availability in a family-pruned native build.
"""
import json
from pathlib import Path

SOURCE = Path(__file__).with_name('mxfp8_cp_async.json')
FIELDS = ['id', 'bm', 'bn', 'bk', 'stages', 'swap', 'warp_m', 'packed_sf']


def load_manifest(source=SOURCE):
    manifest = json.loads(Path(source).read_text())
    if manifest.get('version') != 1 or manifest.get('fields') != FIELDS:
        raise ValueError('Unsupported cp.async configuration schema')
    names, tactics = set(), set()
    for family in manifest['families']:
        name = family['name']
        if not name.isidentifier() or not family['header'].isidentifier() or name in names:
            raise ValueError('Invalid or duplicate family name')
        names.add(name)
        if family['threads'] not in (128, 256):
            raise ValueError('Unsupported thread count')
        if family['fixed_k'] not in ([0], [2048, 2560, 0]):
            raise ValueError('Unsupported K specialization policy')
        if family['shared_memory'] not in ('pipeline', 'max_pipeline_output'):
            raise ValueError('Unsupported shared memory policy')
        for local_id, row in enumerate(family['configs']):
            if len(row) != len(FIELDS):
                raise ValueError('Invalid configuration fields')
            c = dict(zip(FIELDS, row))
            if type(c['id']) is not int or c['id'] != local_id:
                raise ValueError('Configuration IDs must be contiguous and ordered')
            if any(type(c[k]) is not int or c[k] <= 0 for k in ('bm', 'bn', 'bk', 'stages', 'warp_m')):
                raise ValueError('Tile, stages and warp_m must be positive integers')
            if type(c['swap']) is not bool or type(c['packed_sf']) is not bool:
                raise ValueError('Layout flags must be booleans')
            if c['bk'] % 128 or c['bm'] % 16 or c['bn'] % 8:
                raise ValueError('Invalid block-scaled tile alignment')
            if c['warp_m'] not in (1, 2, 4, 8) or c['warp_m'] > family['threads'] // 32:
                raise ValueError('Invalid warp layout')
            tactic = family['first_tactic'] + local_id
            if tactic in tactics:
                raise ValueError('Overlapping tactic ranges')
            tactics.add(tactic)
    return manifest


def configurations(manifest):
    for family in manifest['families']:
        for row in family['configs']:
            c = dict(zip(FIELDS, row))
            c.update(family=family['name'], header=family['header'],
                     tactic=family['first_tactic'] + c['id'],
                     threads=family['threads'], fixed_k=family['fixed_k'])
            pipeline = (c['bm'] + c['bn']) * (c['bk'] + c['bk'] // 32) * c['stages']
            rows, cols = (c['bn'], c['bm']) if c['swap'] else (c['bm'], c['bn'])
            c['shared_bytes'] = (max(pipeline, rows * (cols + 8) * 2)
                                 if family['shared_memory'] == 'max_pipeline_output' else pipeline)
            yield c


def kernel_expression(config, k):
    """Same template arguments as the AOT adapter, for an optional JIT unit."""
    fixed_k = k if k in config['fixed_k'] else 0
    args = [config[x] for x in ('bm', 'bn', 'bk', 'stages', 'swap')]
    if config['fixed_k'] != [0]:
        args.extend([fixed_k, config['warp_m'], config['packed_sf']])
    values = ','.join(str(x).lower() for x in args)
    return f"mxfp8_{config['family']}::gemm<{values}>"
