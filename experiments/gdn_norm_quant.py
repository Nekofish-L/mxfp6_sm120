"""Experimental TP2 Qwen gated RMS norm + packed MXFP8 producer.

Kept outside the public API until exactness and full-model acceptance pass.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _producer(X, Z, W, Q, S, Y, M: tl.constexpr, ZS: tl.constexpr,
              EPS: tl.constexpr, R: tl.constexpr):
    heads = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, 128)
    x = tl.load(X + heads[:, None] * 128 + cols[None, :], heads[:, None] < M * 24, 0).to(tl.float32)
    z = tl.load(Z + (heads[:, None] // 24) * ZS + (heads[:, None] % 24) * 128 + cols[None, :], heads[:, None] < M * 24, 0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    var = tl.sum(x * x, 1) / 128
    y = ((x * tl.rsqrt(var + EPS)[:, None]) * w[None, :]) * (z * tl.sigmoid(z))
    y = y.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + heads[:, None] * 128 + cols[None, :], y, heads[:, None] < M * 24)
    groups = tl.reshape(y, (R, 4, 32))
    maximum = tl.max(tl.abs(groups), 2)
    raw = tl.maximum(maximum / 448., 1.e-30)
    bits = (raw.to(tl.int32, bitcast=True) + 0x7fffff) & 0x7f800000
    inv = 1. / bits.to(tl.float32, bitcast=True)
    q = tl.reshape(groups * inv[:, :, None], (R, 128)).to(tl.float8e4nv)
    tl.store(Q + heads[:, None] * 128 + cols[None, :], q, heads[:, None] < M * 24)
    rows = heads // 24
    g = (heads % 24)[:, None] * 4 + tl.arange(0, 4)[None, :]
    offset = (g // 4) * 512 + (rows % 32)[:, None] * 16 + (rows // 32)[:, None] * 4 + g % 4
    tl.store(S + offset, bits >> 23, heads[:, None] < M * 24)
    # Each CTA owns one disjoint stripe of the packed scale padding.
    p = tl.program_id(0) * 128 + cols
    for base in range(0, tr.cdiv(128 * 96, tr.cdiv(M * 24, R) * 128)):
        idx = p + base * tr.cdiv(M * 24, R) * 128
        row = (idx % 512) // 16 + ((idx % 16) // 4) * 32
        tl.store(S + idx, 127, (idx < 128 * 96) & (row >= M))


def produce(x, z, w, eps=1.e-6):
    m = x.shape[0]
    assert x.shape == (m, 24, 128) and m in (1, 2, 4, 8, 16, 24, 32)
    assert x.dtype == z.dtype == torch.bfloat16 and x.is_contiguous()
    assert z.shape == x.shape and z.stride(-1) == 1 and z.stride(-2) == 128
    assert w.shape == (128,) and w.is_contiguous()
    q = torch.empty((m, 3072), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(128 * 96, dtype=torch.uint8, device=x.device)
    rounded = torch.empty_like(x)
    r = min(tr.next_power_of_2(tr.cdiv(m * 24, 340)), 4)
    _producer[(tr.cdiv(m * 24, r),)](x, z, w, q, scales, rounded, m, z.stride(0), eps, r, num_warps=1)
    return q.view(torch.uint8), scales, rounded
