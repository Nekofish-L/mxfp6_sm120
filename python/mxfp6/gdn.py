"""Exact TP2 Qwen gated RMS norm + packed MXFP8 producer."""
import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit
def _producer(X, Z, W, Q, S, Y, M: tl.constexpr, ZS: tl.constexpr,
              EPS, R: tl.constexpr, WRITE_ROUNDED: tl.constexpr):
    # Freeze the original vLLM reduction tree. The automatic Triton layout
    # changes to 16 values/lane once FP8 conversion is fused (instead of 8),
    # changing rounding at M16/24/32 even though the source arithmetic matches.
    L: tl.constexpr = BlockedLayout([1, 4] if R == 1 else [1, 8],
        [1, 32] if R == 1 else [2, 16], [1, 1], [1, 0])
    heads = tl.program_id(0) * R + tl.arange(0, R, layout=SliceLayout(1, L))
    cols = tl.arange(0, 128, layout=SliceLayout(0, L))
    x = tl.load(X + heads[:, None] * 128 + cols[None, :], heads[:, None] < M * 24, 0).to(tl.float32)
    z = tl.load(Z + (heads[:, None] // 24) * ZS + (heads[:, None] % 24) * 128 + cols[None, :], heads[:, None] < M * 24, 0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    var = tl.sum(x * x, 1) / 128
    y = ((x * tl.rsqrt(var + EPS)[:, None]) * w[None, :]) * (z * (1. / (1. + tl.exp(-z))))
    y = y.to(tl.bfloat16).to(tl.float32)
    if WRITE_ROUNDED:
        tl.store(Y + heads[:, None] * 128 + cols[None, :], y, heads[:, None] < M * 24)
    groups = tl.reshape(y, (R, 4, 32))
    maximum = tl.max(tl.abs(groups), 2)
    raw = tl.maximum(maximum / 448., 1.e-30)
    bits = (raw.to(tl.int32, bitcast=True) + 0x7fffff) & 0x7f800000
    inv = 1. / bits.to(tl.float32, bitcast=True)
    q = tl.reshape(groups * inv[:, :, None], (R, 128)).to(tl.float8e4nv)
    tl.store(Q + heads[:, None] * 128 + cols[None, :], q, heads[:, None] < M * 24)
    rows = heads // 24
    g = (heads % 24)[:, None] * 4 + cols[None, :] // 32
    offset = (g // 4) * 512 + (rows % 32)[:, None] * 16 + (rows // 32)[:, None] * 4 + g % 4
    codes = tl.reshape(((bits >> 23)[:, :, None] + tl.full((R, 4, 32), 0, tl.int32, groups.type.layout)), (R, 128))
    tl.store(S + offset, codes, (heads[:, None] < M * 24) & (cols[None, :] % 32 == 0))
    # Each CTA owns one disjoint stripe of the packed scale padding.
    pc = tl.arange(0, 128, layout=BlockedLayout([4], [32], [1], [0]))
    p = tl.program_id(0) * 128 + pc
    for base in range(0, tr.cdiv(128 * 96, tr.cdiv(M * 24, R) * 128)):
        idx = p + base * tr.cdiv(M * 24, R) * 128
        row = (idx % 512) // 16 + ((idx % 16) // 4) * 32
        tl.store(S + idx, 127, (idx < 128 * 96) & (row >= M))
    # All stores precede readiness; the PDL GEMM consumer waits before reads.
    tl.barrier()
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents; mov.u32 $0, 0;",
                             constraints="=r", args=[], dtype=tl.int32,
                             is_pure=False, pack=1)


def gated_norm_mxfp8(x, z, w, eps=1.e-6, *, rounded_out=None):
    """Exact Qwen TP2 [M,24,128] SiLU-gated RMS norm and packed MXFP8.

    The layout pins vLLM 0.29's FP32 reduction order and BF16 boundary.
    ``z`` may be the noncontiguous slice of an [M,8192] QKVZ projection.
    ``rounded_out`` is diagnostic only; normal inference does not write BF16.
    """
    m = x.shape[0]
    if (x.shape != (m, 24, 128) or m not in (1, 2, 4, 8, 16, 24, 32)
            or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous()
            or z.shape != x.shape or z.dtype != x.dtype or z.device != x.device
            or z.stride(-1) != 1 or z.stride(-2) != 128
            or w.shape != (128,) or not w.is_contiguous() or w.device != x.device
            or w.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError('GDN producer requires native TP2 BF16 [M,24,128] and BF16/FP32 [128] weight')
    if not 0 < eps < 1:
        raise ValueError('eps must be finite and in (0,1)')
    props = torch.cuda.get_device_properties(x.device)
    if (props.major, props.minor) != (12, 0):
        raise ValueError('GDN producer requires SM120')
    if rounded_out is not None and (rounded_out.shape != x.shape
            or rounded_out.dtype != x.dtype or rounded_out.device != x.device
            or not rounded_out.is_contiguous()):
        raise ValueError('rounded_out must match the contiguous BF16 input')
    q = torch.empty((m, 3072), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(128 * 96, dtype=torch.uint8, device=x.device)
    r = min(tr.next_power_of_2(tr.cdiv(m * 24, 2 * props.multi_processor_count)), 4)
    with torch.cuda.device(x.device):
        _producer[(tr.cdiv(m * 24, r),)](x, z, w, q, scales, rounded_out,
            m, z.stride(0), eps, r, rounded_out is not None, num_warps=1)
    return q.view(torch.uint8), scales
