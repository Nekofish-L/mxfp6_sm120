"""Explicit two-limb MXFP8 activations for SM120 QKVZ/MLP batches 1–128.

The residual is rounded to BF16 before quantization. Ordinary MXFP8 APIs keep
using their existing single-activation path. Triton is imported only on launch.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from ._loader import load_mxfp8_library as load_library

SUPPORTED_SHAPES = tuple((m, n, 2560) for m in range(1, 129)
                         for n in (12288, 18432))


def select_variant(m: int, n: int, k: int) -> str:
    """Return the tile family; original family names remain API compatible."""
    if 1 <= m <= 128 and k == 2560:
        if n == 18432:
            if m <= 32:
                return "mlp32"
            return "mlp_pipereg_cfg2"
        if n == 12288:
            if m <= 32:
                return "qkvz32"
            return "qkvz64_pipereg_cfg2"
    raise ValueError(f"Unsupported dual MXFP8 shape {(m, n, k)}; expected "
                     "1<=M<=128, N12288 or N18432, K2560")


def _check_input(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if (x.ndim != 2 or not 1 <= x.shape[0] <= 128 or x.shape[1] != 2560
            or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous()):
        raise ValueError("Expected contiguous CUDA BF16 [M,2560], 1<=M<=128")


def _check_buffers(x, hi, s_hi, res, s_res):
    for name, tensor, shape, dtype in (
        ("hi", hi, tuple(x.shape), torch.float8_e4m3fn),
        ("res", res, tuple(x.shape), torch.float8_e4m3fn),
        ("s_hi", s_hi, (10240,), torch.uint8),
        ("s_res", s_res, (10240,), torch.uint8),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                or tensor.device != x.device or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on {x.device}")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must have 16-byte-aligned storage")
    tensors = (x, hi, s_hi, res, s_res)
    for i, lhs in enumerate(tensors):
        for rhs in tensors[i + 1:]:
            if torch._C._is_alias_of(lhs, rhs):
                raise ValueError("Input and quantization output buffers must not alias")


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def run(X, HI, LO, SHI, SLO, K: tl.constexpr, M: tl.constexpr,
            GROUPS: tl.constexpr):
        row = tl.program_id(1)
        group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
        lane = tl.arange(0, 32)
        col = group[:, None] * 32 + lane[None, :]
        x = tl.load(X + row * K + col, col < K, other=0).to(tl.float32)
        maximum = tl.max(tl.abs(x), axis=1)
        e = tl.minimum(tl.maximum(tl.ceil(tl.log2(maximum * (1.0 / 448.0))) + 127.0,
                                  0.0), 254.0)
        scale = tl.where(maximum == 0, 1.0, tl.exp2(e - 127.0))
        q = (x / scale[:, None]).to(HI.dtype.element_ty)
        # Deliberate BF16 rounding is the numerical screen's residual contract.
        residual = (x - q.to(tl.float32) * scale[:, None]).to(X.dtype.element_ty)
        residual = residual.to(tl.float32)
        rmax = tl.max(tl.abs(residual), axis=1)
        re = tl.minimum(tl.maximum(tl.ceil(tl.log2(rmax * (1.0 / 448.0))) + 127.0,
                                   0.0), 254.0)
        rscale = tl.where(rmax == 0, 1.0, tl.exp2(re - 127.0))
        rq = (residual / rscale[:, None]).to(LO.dtype.element_ty)
        tl.store(HI + row * K + col, q, col < K)
        tl.store(LO + row * K + col, rq, col < K)
        # CUTLASS 128x4 scale swizzle. Each valid logical group has one writer.
        off = (((row // 128 * (K // 128) + group // 4) * 32 + row % 32)
               * 4 + (row % 128) // 32) * 4 + group % 4
        tl.store(SHI + off, e.to(tl.uint8), group < K // 32)
        tl.store(SLO + off, re.to(tl.uint8), group < K // 32)
        # Padding is defined in the same launch; row0 alone owns invalid rows.
        if row == 0:
            pr = M + tl.arange(0, 128)
            po = (((pr[:, None] // 128 * (K // 128) + group[None, :] // 4) * 32
                   + pr[:, None] % 32) * 4 + (pr[:, None] % 128) // 32) * 4 + group[None, :] % 4
            mask = (pr[:, None] < tl.cdiv(M, 128) * 128) & (group[None, :] < K // 32)
            tl.store(SHI + po, 0, mask)
            tl.store(SLO + po, 0, mask)

    return run


def quantize_out(x, hi, s_hi, res, s_res):
    """Quantize finite BF16 X into high/residual limbs in supplied buffers.

    Scale padding is written in the same launch. All buffers must be distinct;
    retain them for changing-input CUDA Graph replay. Nonfinite X is unsupported.
    """
    _check_input(x)
    _check_buffers(x, hi, s_hi, res, s_res)
    m = x.shape[0]
    with torch.cuda.device(x.device):
        _kernel()[(20, m)](x, hi, res, s_hi, s_res, K=2560, M=m, GROUPS=4,
                          num_warps=4, num_stages=1, enable_fp_fusion=False)
    return hi, s_hi, res, s_res


def quantize(x):
    """Allocate and quantize finite contiguous CUDA BF16 [M,2560], M=1–128."""
    _check_input(x)
    hi = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    res = torch.empty_like(hi)
    s_hi = torch.empty(10240, dtype=torch.uint8, device=x.device)
    s_res = torch.empty_like(s_hi)
    return quantize_out(x, hi, s_hi, res, s_res)


def gemm_out(hi, weight, s_hi, s_weight, res, s_res, out, *, use_pdl=False):
    """Compute (high @ W.T + residual @ W.T), rounding once to BF16.

    Both terms use independent FP32 accumulators. All values are contiguous
    E4M3FN matrices; scales are flat uint8 CUTLASS 128x4 swizzled E8M0 buffers.
    No workspace is needed. Unsupported shapes raise instead of falling back.
    With use_pdl=True, wait for the predecessor before reading either limb and
    permit a following PDL kernel to begin its independent setup early.
    """
    if not all(isinstance(t, torch.Tensor) for t in
               (hi, weight, s_hi, s_weight, res, s_res, out)):
        raise TypeError("All operands and out must be torch.Tensor instances")
    if hi.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected hi and weight matrices")
    select_variant(hi.shape[0], weight.shape[0], hi.shape[1])
    load_library()
    torch.ops.mxfp8_sm120.dual_gemm_out(hi, weight, s_hi, s_weight, res, s_res, out, use_pdl)
    return out


__all__ = ["SUPPORTED_SHAPES", "select_variant", "quantize", "quantize_out",
           "gemm_out", "load_library"]
