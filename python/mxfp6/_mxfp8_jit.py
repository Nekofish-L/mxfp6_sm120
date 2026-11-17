"""Opt-in, source-checkout JIT backend; never imported by the default AOT path.

Source roots and compiler settings are immutable for a process lifetime.
Compiled handles are retained for that lifetime so captured Graphs remain valid
when their prepared Python callable is released. There is intentionally no eviction.
"""
import hashlib
import json
import os
from pathlib import Path
import threading

import torch

from ._autotune_cache import file_lock
from ._mxfp8_config import configurations, load_manifest
from ._mxfp8_jit_source import dependency_signature, device_source

_STATE_LOCK = threading.Lock()
_CONTEXTS = {}
_KEY_LOCKS = {}
_KERNELS = {}


def _outside_capture(device):
    if getattr(torch.compiler, 'is_compiling', lambda: False)():
        raise RuntimeError('Precompile/prepare JIT calls before torch.compile or CUDA Graph capture')
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Precompile/prepare JIT calls before CUDA Graph capture')


def _source_root(source_root):
    root = (Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]).resolve()
    for name in ('csrc/mxfp8_jit.cpp', 'third_party/deepjit/include/deep_jit',
                 'third_party/cutlass/include/cutlass', 'python/mxfp6/mxfp8_cp_async.json'):
        if not (root/name).exists():
            raise RuntimeError('JIT requires a matching source checkout with initialized CUTLASS/DeepJIT; '
                               'pass source_root=... when using an installed wheel')
    return root


def _context(root, tactic=None):
    # Serialize host extension initialization and dependency hashing. A context
    # owns its snapshot; editing source requires restarting the process.
    with _STATE_LOCK:
        if root not in _CONTEXTS:
            configs = {c['tactic']: c for c in configurations(load_manifest(root/'python/mxfp6/mxfp8_cp_async.json'))}
            if tactic is not None and tactic not in configs:
                raise ValueError('This tactic has not been migrated to JIT')
            from torch.utils.cpp_extension import load
            source = root/'csrc/mxfp8_jit.cpp'
            name = 'mxfp8_deepjit_' + hashlib.sha256(str(root).encode() + source.read_bytes()).hexdigest()[:12]
            module = load(name=name, sources=[str(source)],
                          extra_include_paths=[str(root/'third_party/deepjit/include'), str(root/'csrc/include')],
                          extra_cflags=['-O3', '-std=c++20'], with_cuda=True)
            signature = dependency_signature(root)
            cache = os.environ.get('MXFP8_JIT_CACHE_DIR', os.environ.get('DJ_JIT_CACHE_DIR'))
            locks = Path(cache.split(':')[0]) if cache else Path.home()/'.dj'
            _CONTEXTS[root] = (module, signature, configs, locks/'mxfp8-locks')
        return _CONTEXTS[root]


def precompile(config, k, *, device=None, source_root=None):
    from .mxfp8 import W8A8Config
    if not isinstance(config, W8A8Config):
        raise TypeError('JIT requires an explicit W8A8Config')
    if config.splits != 1 or config.swizzle != 1 or config.sms != 0:
        raise ValueError('cp.async JIT supports splits=1, swizzle=1, sms=0')
    if type(k) is not int or k <= 0 or k % 128:
        raise ValueError('JIT requires positive K divisible by 128')
    device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
    if device.type != 'cuda':
        raise ValueError('JIT requires a CUDA device')
    index = torch.cuda.current_device() if device.index is None else device.index
    _outside_capture(index)
    if torch.cuda.get_device_capability(index) != (12, 0):
        raise RuntimeError('MXFP8 JIT requires SM120')
    root = _source_root(source_root)
    module, signature, configs, locks = _context(root, config.config_id)
    if config.config_id not in configs:
        raise ValueError('This tactic has not been migrated to JIT')
    c = configs[config.config_id]
    fixed_k = k if k in c['fixed_k'] else 0
    key = (root, config.config_id, fixed_k, index)
    with _STATE_LOCK:
        if key in _KERNELS:
            return _KERNELS[key]
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        with _STATE_LOCK:
            if key in _KERNELS:
                return _KERNELS[key]
        source = device_source(root, c, k)
        # Device-independent lock: processes/devices can reuse the same disk
        # CUBIN, but each device needs its own loaded function handle.
        disk_key = hashlib.sha256((signature + source + json.dumps(c, sort_keys=True)).encode()).hexdigest()
        with file_lock(locks, disk_key), torch.cuda.device(index):
            compiled = module.compile_kernel(str(root), signature, source,
                f'mxfp8_tactic{config.config_id}_k{fixed_k}', index,
                c['bm'], c['bn'], c['threads'], c['shared_bytes'], fixed_k, c['swap'])
        with _STATE_LOCK:
            _KERNELS[key] = compiled
        return compiled


def prepare(a, b, *, config, out=None, use_pdl=False, source_root=None):
    from .mxfp8 import _operands
    av, bv, sa, sb = _operands(a, b)
    _outside_capture(a.device)
    compiled = precompile(config, a.k, device=a.device, source_root=source_root)
    root = _source_root(source_root)
    module = _context(root)[0]
    if out is None:
        out = torch.empty((a.rows, b.rows), device=a.device, dtype=torch.bfloat16)
    prepared = module.PreparedCall(compiled, av, bv, sa, sb, out, use_pdl)

    def run():
        return prepared()

    run.config = config
    run.backend = "jit"
    run.jit_kernel = compiled
    return run
