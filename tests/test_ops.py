#!/usr/bin/env python3
"""End-to-end tests for the packaged CUDA conversion and GEMM APIs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def decode_e3m2(codes: torch.Tensor) -> torch.Tensor:
    raw = codes.to(torch.int32)
    exponent = (raw >> 2) & 0x07
    mantissa = raw & 0x03
    subnormal = mantissa.float() * (2.0**-4)
    normal = torch.ldexp(1.0 + mantissa.float() / 4.0, exponent - 3)
    value = torch.where(exponent == 0, subnormal, normal)
    return torch.where((raw & 0x20) != 0, -value, value)


def decode_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32),
        scales.to(torch.int32) - 127,
    )


def cpu_pack_fp6(codes: torch.Tensor) -> torch.Tensor:
    q = (codes.cpu().contiguous().view(-1) & 0x3F).view(-1, 4)
    output = torch.empty((q.shape[0], 3), dtype=torch.uint8)
    output[:, 0] = q[:, 0] | ((q[:, 1] & 0x03) << 6)
    output[:, 1] = (q[:, 1] >> 2) | ((q[:, 2] & 0x0F) << 4)
    output[:, 2] = (q[:, 2] >> 4) | (q[:, 3] << 2)
    return output.flatten()


def test_fp6_tools(mxfp6) -> None:
    for rows, k in ((1, 128), (16, 128), (32, 256), (129, 128), (2048, 128)):
        codes = torch.randint(0, 256, (rows, k), device="cuda", dtype=torch.uint8)
        packed = mxfp6.pack_fp6(codes)
        torch.testing.assert_close(packed.cpu(), cpu_pack_fp6(codes))
        unpacked = mxfp6.unpack_fp6(packed, rows, k)
        torch.testing.assert_close(unpacked, codes & 0x3F)
    print("PASS FP6 CUDA pack/unpack and CPU bitstream compatibility")


def test_scale_tools(mxfp6) -> None:
    for rows, k in (
        (1, 128),
        (16, 128),
        (32, 256),
        (128, 128),
        (129, 256),
        (2048, 128),
    ):
        logical = torch.randint(
            120, 135, (rows, k // 32), device="cuda", dtype=torch.uint8
        )
        packed = mxfp6.pack_scales(logical)
        assert packed.numel() == ((rows + 127) // 128) * 128 * k // 32
        restored = mxfp6.unpack_scales(packed, rows, k)
        torch.testing.assert_close(restored, logical)
    print("PASS UE8M0 CUDA layout pack/unpack with padded rows")


def make_problem(m: int, n: int, k: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a_codes = torch.randint(
        0, 12, (m, k), generator=generator, device="cuda", dtype=torch.uint8
    )
    b_codes = torch.randint(
        0, 12, (n, k), generator=generator, device="cuda", dtype=torch.uint8
    )
    a_codes |= torch.randint(
        0, 2, (m, k), generator=generator, device="cuda", dtype=torch.uint8
    ) << 5
    b_codes |= torch.randint(
        0, 2, (n, k), generator=generator, device="cuda", dtype=torch.uint8
    ) << 5
    sfa = torch.randint(
        125, 129, (m, k // 32), generator=generator,
        device="cuda", dtype=torch.uint8,
    )
    sfb = torch.randint(
        125, 129, (n, k // 32), generator=generator,
        device="cuda", dtype=torch.uint8,
    )
    return a_codes, b_codes, sfa, sfb


def reference_gemm(
    a_codes: torch.Tensor,
    b_codes: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
) -> torch.Tensor:
    a = decode_e3m2(a_codes) * decode_ue8m0(sfa).repeat_interleave(32, dim=1)
    b = decode_e3m2(b_codes) * decode_ue8m0(sfb).repeat_interleave(32, dim=1)
    return (a @ b.t()).half()


def test_random_scale_gemm(mxfp6) -> None:
    shapes = (
        (1, 8, 128),
        (16, 128, 128),
        (17, 136, 128),
        (32, 128, 128),
        (48, 136, 128),
        (64, 128, 256),
        (96, 136, 128),
        (97, 128, 128),
        (112, 136, 128),
        (127, 128, 128),
        (128, 136, 128),
        (129, 128, 128),
        (255, 136, 128),
        (512, 128, 128),
        (2048, 128, 128),
    )
    for m, n, k in shapes:
        a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 1000 + m)
        a = mxfp6.pack_operand(a_codes, sfa)
        b = mxfp6.pack_operand(b_codes, sfb)
        actual = mxfp6.gemm(a, b)
        reference = reference_gemm(a_codes, b_codes, sfa, sfb)
        torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.25)
        max_abs = (actual - reference).abs().max().item()
        print(f"PASS random-scale GEMM {m}x{n}x{k}: max_abs={max_abs:g}")


def test_nondefault_stream(mxfp6) -> None:
    m, n, k = 16, 128, 128
    a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 2026)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = mxfp6.gemm_from_codes(a_codes, b_codes, sfa, sfb)
    stream.synchronize()
    reference = reference_gemm(a_codes, b_codes, sfa, sfb)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.25)
    print("PASS CUDA non-default current-stream semantics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    options = parser.parse_args()
    if options.library is not None:
        os.environ["MXFP6_LIBRARY_PATH"] = str(options.library.resolve())
    sys.path.insert(0, str(ROOT / "python"))
    import mxfp6

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 is required")

    print(f"Library: {mxfp6.load_library()}")
    test_fp6_tools(mxfp6)
    test_scale_tools(mxfp6)
    test_random_scale_gemm(mxfp6)
    test_nondefault_stream(mxfp6)
    print("All MXFP6 tool tests passed")


if __name__ == "__main__":
    main()
