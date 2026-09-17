"""Rounded BF16 attention gate and packed MXFP8 producer for SM120 TP2."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _gate_quant(X, G, Q, S, Y, M: tl.constexpr, GS: tl.constexpr,
                WRITE_ROUNDED: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * B + tl.arange(0, B)
    row, col = offsets // 3072, offsets % 3072
    x = tl.load(X + offsets, row < M, 0).to(tl.float32)
    g = tl.load(G + row * GS + col, row < M, 0).to(tl.float32)
    # Match eager ATen's expf/division and its materialized BF16 sigmoid.
    sigmoid = tl.div_rn(1., 1. + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    y = (x * sigmoid).to(tl.bfloat16).to(tl.float32)
    if WRITE_ROUNDED:
        tl.store(Y + offsets, y, row < M)
    groups = tl.reshape(y, (B // 32, 32))
    raw = tl.maximum(tl.max(tl.abs(groups), 1) / 448., 1.e-30)
    bits = (raw.to(tl.int32, bitcast=True) + 0x7fffff) & 0x7f800000
    q = tl.reshape(groups * (1. / bits.to(tl.float32, bitcast=True))[:, None], (B,)).to(tl.float8e4nv)
    tl.store(Q + offsets, q, row < M)
    group = block * (B // 32) + tl.arange(0, B // 32)
    r, c = group // 96, group % 96
    packed = (c // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + c % 4
    tl.store(S + packed, bits >> 23, r < M)
    for base in range(tr.cdiv(128 * 96, tr.cdiv(M * 3072, B) * B)):
        idx = offsets + base * tr.cdiv(M * 3072, B) * B
        pad_row = (idx % 512) // 16 + ((idx % 16) // 4) * 32
        tl.store(S + idx, 127, (idx < 128 * 96) & (pad_row >= M))
    tl.debug_barrier()
    tl.inline_asm_elementwise('griddepcontrol.launch_dependents; mov.u32 $0, 0;',
        constraints='=r', args=[], dtype=tl.int32, is_pure=False, pack=1)


def sigmoid_gate_mxfp8(x, gate, *, rounded_out=None):
    """Preserve sigmoid→BF16 and multiply→BF16 before block32 quantization."""
    if (x.ndim != 2 or x.shape[0] not in (1, 2, 4, 8, 16, 24, 32)
            or x.shape[1] != 3072 or not x.is_cuda
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or gate.shape != x.shape or gate.dtype != x.dtype
            or gate.device != x.device or gate.stride(1) != 1
            or gate.stride(0) < 3072):
        raise ValueError('Attention producer requires BF16 [M,3072] TP2 input and gate')
    props = torch.cuda.get_device_properties(x.device)
    if (props.major, props.minor) != (12, 0):
        raise ValueError('Attention producer requires SM120')
    if rounded_out is not None and (rounded_out.shape != x.shape
            or rounded_out.dtype != x.dtype or rounded_out.device != x.device
            or not rounded_out.is_contiguous()):
        raise ValueError('rounded_out must match the contiguous BF16 input')
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(128 * 96, device=x.device, dtype=torch.uint8)
    with torch.cuda.device(x.device):
        _gate_quant[(tr.cdiv(x.numel(), 128),)](x, gate, q, scales, rounded_out,
            x.shape[0], gate.stride(0), rounded_out is not None, 128,
            num_warps=4, enable_fp_fusion=False)
    return q.view(torch.uint8), scales
