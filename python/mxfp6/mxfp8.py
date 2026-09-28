"""Native SM120 MXFP8 W8A8 GEMM with FlashInfer-compatible scale storage."""
import json
import os
import threading
from pathlib import Path

import torch

_LOADED = False
_LOAD_LOCK = threading.Lock()
_DISPATCH = None
STREAM_K_TACTICS = frozenset([12, 13, 14, 15, 16, 39, 40, 42, 51, 57])
TRITON_TACTICS = range(100, 106)
TACTICS = tuple(t for t in range(70) if t != 30) + tuple(TRITON_TACTICS)


def load_library():
    """Load the installed extension, or JIT-build from a patched source tree."""
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        root = Path(__file__).resolve().parents[2]
        override = os.getenv('MXFP8_LIBRARY_PATH')
        candidates = [Path(override).expanduser()] if override else [
            Path(__file__).parent / 'mxfp8_torch.so', root / 'build/mxfp8_torch.so']
        if override and not candidates[0].is_file():
            raise ImportError(f'MXFP8_LIBRARY_PATH does not exist: {candidates[0]}')
        for library in candidates:
            if library.is_file():
                torch.ops.load_library(str(library))
                _LOADED = True
                return
        if not (root / 'csrc/mxfp8.cu').is_file():
            raise ImportError('mxfp8_torch.so is missing; rebuild/install the wheel.')
        builder = root / 'third_party/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl'
        scheduler = root / 'third_party/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp'
        if (not builder.is_file() or 'sSFATileShape_M' not in builder.read_text()
                or not scheduler.is_file() or 'get_workspace_layout' not in scheduler.read_text()):
            raise ImportError('Apply the required patches: bash scripts/apply_cutlass_patches.sh --runtime-only')
        from torch.utils.cpp_extension import load
        load(
            name='mxfp8_sm120',
            sources=[str(root / 'csrc' / name) for name in
                     ['mxfp8.cu', 'mxfp8_gemv.cu', 'mxfp8_wide.cu', 'mxfp8_direct.cu', 'mxfp8_extra.cu']],
            extra_include_paths=[str(root / p) for p in
                                 ['csrc/include', 'third_party/cutlass/include',
                                  'third_party/cutlass/tools/util/include']],
            extra_cuda_cflags=['-O3', '-arch=sm_120a', '--expt-relaxed-constexpr',
                               '-Xcudafe=--diag_suppress=20012',
                               '-Xcudafe=--diag_suppress=20013',
                               '-Xcudafe=--diag_suppress=20015'],
            is_python_module=False,
        )
        _LOADED = True


def select_config(m, n, k):
    """Select a measured configuration or a conservative untuned fallback."""
    global _DISPATCH
    if _DISPATCH is None:
        path = Path(__file__).with_name('mxfp8_dispatch.json')
        _DISPATCH = json.loads(path.read_text())['configs'] if path.exists() else {}
    key = f'{m},{n},{k}'
    if key in _DISPATCH:
        return tuple((_DISPATCH[key] + [0])[:4])
    tactic = 0 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 7 if m <= 64 else 9
    return tactic, 1, 1, 0


def _op(tactic):
    if tactic not in TACTICS:
        raise ValueError(f'Unknown MXFP8 tactic: {tactic}')
    if 51 <= tactic <= 57:
        return torch.ops.mxfp8_sm120.extra_out
    if tactic >= 43:
        return torch.ops.mxfp8_sm120.direct_out
    if tactic >= 37:
        return torch.ops.mxfp8_sm120.wide_out
    return torch.ops.mxfp8_sm120.mm_out


def mm(a, b, sa, sb, *, tactic=None, splits=1, swizzle=1, out=None, workspace=None, sms=0):
    """Compute A[M,K] @ B[N,K].T -> BF16 using E4M3 and swizzled E8M0/32.

    Loads the extension automatically on first use, like the MXFP6 API. No
    input quantization or layout conversion is performed. Before CUDA graph
    capture, use ``prepare`` to load, compile, and allocate private workspace.
    """
    load_library()
    if tactic is None:
        tactic, splits, swizzle, sms = select_config(a.shape[0], b.shape[0], a.shape[1])
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    if tactic in TRITON_TACTICS:
        # Reuse the native validation path without launching a GPU kernel.
        torch.ops.mxfp8_sm120.mm_out(a, b, sa, sb, out, 0, splits, swizzle, None, True, sms)
        from ._mxfp8_triton import CONFIGS, run as triton_run
        with torch.cuda.device(a.device):
            triton_run(a, b, sa, sb, out, CONFIGS[tactic - 100])
        return out
    op = _op(tactic)
    if 33 <= tactic <= 36:
        torch.ops.mxfp8_sm120.gemv_out(a, b, sa, sb, out, 1 << (tactic - 33))
    else:
        op(a, b, sa, sb, out, tactic, splits, swizzle, workspace, False, sms)
    return out


def allocate_workspace(a, b, sa, sb, *, tactic, splits=1, swizzle=1, out=None, sms=0):
    """Allocate zeroed Stream-K workspace, or return None for other tactics.

    Each concurrent execution needs private workspace. Patched barriers reset
    after each call; do not share workspace across overlapping executions or
    different configurations. Allocate before CUDA graph capture.
    """
    load_library()
    if tactic in TRITON_TACTICS:
        return None
    op = _op(tactic)
    if tactic not in STREAM_K_TACTICS:
        return None
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    size = op(a, b, sa, sb, out, tactic, splits, swizzle, None, True, sms)
    return torch.zeros(size, device=a.device, dtype=torch.uint8)


def prepare(a, b, sa, sb, *, tactic=None, splits=1, swizzle=1, out=None, sms=0):
    """Prepare a reusable call with private output and Stream-K workspace.

    Input contents may change between calls. Concurrent executions must use
    separate prepared calls and outputs. Call before CUDA graph capture.
    """
    load_library()
    if tactic is None:
        tactic, splits, swizzle, sms = select_config(a.shape[0], b.shape[0], a.shape[1])
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    workspace = allocate_workspace(a, b, sa, sb, tactic=tactic, splits=splits,
                                   swizzle=swizzle, out=out, sms=sms)

    def run():
        return mm(a, b, sa, sb, tactic=tactic, splits=splits, swizzle=swizzle,
                  out=out, workspace=workspace, sms=sms)

    run.config = (tactic, splits, swizzle, sms)
    if tactic in TRITON_TACTICS:
        # Compile this exact shape before the caller starts graph capture.
        run()
    return run
