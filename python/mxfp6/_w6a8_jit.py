"""Lazy DeepJIT compilation of W6A8 CUTLASS kernels; shapes remain runtime data."""
import hashlib
import json
import os
from pathlib import Path
import re
import threading

import torch

from ._autotune_cache import file_lock
from ._loader import load_library

_LOCK = threading.Lock()
_CONTEXT = None
_KERNELS = {}
_KEY_LOCKS = {}


def device_source(root, config, dtype):
    launch = config['launch']
    kind = 'normal' if launch['kind'] == 'normal_128' else launch['kind']
    alias = ('KernelW6A8_128x128x128' + launch['kernel']
             if launch['kind'] == 'normal_128' else launch['kernel'])
    included = set()
    def inline(text):
        def replace(match):
            name = match[1]
            if name.startswith('mxfp6_gemm/'):
                if name in included:
                    return ''
                included.add(name)
                return inline((root / 'csrc/include' / name).read_text())
            return '#include <' + name + '>'
        return re.sub(r'#include "([^"]+)"', replace, text)
    source = inline((root / f'csrc/include/mxfp6_gemm/kernel_{kind}.hpp').read_text())
    element = 'half_t' if dtype == torch.float16 else 'bfloat16_t'
    return source + (f'\nusing JitKernel = mxfp6_gemm::{kind}::{alias}::RebindOutput<cutlass::{element}>;\n'
                     'static void instantiate() {\n'
                     '  auto ptr = reinterpret_cast<void*>(&cutlass::device_kernel<JitKernel::GemmKernel>);\n'
                     '}\n')


def _context(source_root):
    global _CONTEXT
    root = (Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]).resolve()
    with _LOCK:
        if _CONTEXT is not None:
            if _CONTEXT[0] != root:
                raise RuntimeError('W6A8 JIT source_root is fixed for the process lifetime')
            return _CONTEXT
        for name in ('csrc/w6a8_jit.cpp', 'third_party/deepjit/include/deep_jit',
                     'third_party/cutlass/include/cutlass', 'python/mxfp6/w6a8_dispatch.json'):
            if not (root / name).exists():
                raise RuntimeError('W6A8 JIT requires a matching checkout with CUTLASS and DeepJIT submodules')
        digest = hashlib.sha256()
        for relative in ('csrc/include', 'third_party/cutlass/include', 'third_party/deepjit/include'):
            for path in sorted((root / relative).rglob('*')):
                if path.is_file():
                    digest.update(str(path.relative_to(root)).encode())
                    digest.update(path.read_bytes())
        source = root / 'csrc/w6a8_jit.cpp'
        digest.update(source.read_bytes())
        digest.update(Path(load_library()).read_bytes())
        signature = digest.hexdigest()
        from torch.utils.cpp_extension import load
        load(name='w6a8_deepjit_' + signature[:12], sources=[str(source)],
             extra_include_paths=[str(root / 'third_party/deepjit/include'), str(root / 'csrc/include')],
             extra_cflags=['-O3', '-std=c++20'], with_cuda=True, is_python_module=False)
        manifest = json.loads((root / 'python/mxfp6/w6a8_dispatch.json').read_text())
        cache = os.environ.get('MXFP6_JIT_CACHE_DIR', os.environ.get('DJ_JIT_CACHE_DIR'))
        locks = Path(cache.split(':')[0]) if cache else Path.home() / '.dj'
        _CONTEXT = (root, signature, manifest['candidates'], locks / 'w6a8-locks')
        return _CONTEXT


def bind(native, config, dtype, device, source_root=None):
    root, signature, candidates, locks = _context(source_root)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (config.config_id, dtype, index)
    with _LOCK:
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if key not in _KERNELS:
            source = device_source(root, candidates[config.config_id], dtype)
            disk_key = hashlib.sha256((signature + source).encode()).hexdigest()
            with file_lock(locks, disk_key), torch.cuda.device(index):
                _KERNELS[key] = torch.ops.mxfp6_jit.compile(
                    str(root), signature, source, config.config_id, dtype, index)
        compiled = _KERNELS[key]
    compiled.bind(native)
    return compiled
