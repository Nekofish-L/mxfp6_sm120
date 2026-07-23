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


def test_fp6_to_fp8(mxfp6) -> None:
    codes = torch.arange(64, dtype=torch.uint8, device="cuda").repeat(4, 2)
    packed = mxfp6.pack_fp6(codes)
    expanded = mxfp6.expand_fp6_to_fp8(packed, *codes.shape)
    torch.testing.assert_close(expanded.float(), decode_e3m2(codes))
    print("PASS lossless E3M2-to-E4M3 expansion for all 64 encodings")


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
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    a = decode_e3m2(a_codes) * decode_ue8m0(sfa).repeat_interleave(32, dim=1)
    b = decode_e3m2(b_codes) * decode_ue8m0(sfb).repeat_interleave(32, dim=1)
    return (a @ b.t()).to(out_dtype)


def decode_mxfp8(operand, mxfp6) -> torch.Tensor:
    values = operand.values.view(operand.rows, operand.k).view(
        torch.float8_e4m3fn
    ).float()
    scales = mxfp6.unpack_scales(
        operand.scales, operand.rows, operand.k
    )
    return values * decode_ue8m0(scales).repeat_interleave(32, dim=1)


def test_dynamic_quantization(mxfp6) -> None:
    """Verify the real 16->8 and 16->6 activation mappings."""
    for dtype in (torch.float16, torch.bfloat16):
        for rows, k in ((1, 128), (7, 256), (129, 128)):
            generator = torch.Generator(device="cuda").manual_seed(rows + k)
            source = (
                torch.randn(
                    (rows, k), generator=generator, device="cuda",
                    dtype=torch.float32,
                )
                * 6.0
            ).to(dtype)
            quantized = mxfp6.quantize_mxfp8(source)
            logical_scales = mxfp6.unpack_scales(
                quantized.scales, rows, k
            )
            scales = decode_ue8m0(logical_scales).repeat_interleave(32, dim=1)
            expected = (source.float() / scales).to(
                torch.float8_e4m3fn
            ).view(torch.uint8)
            actual = quantized.values.view(rows, k)
            torch.testing.assert_close(actual, expected)

        # Every group contains +28, so its exact UE8M0 scale is one. All E3M2
        # values then round-trip through the native 16->6 conversion.
        rows, k = 17, 256
        codes = torch.arange(32, dtype=torch.uint8, device="cuda").repeat(
            rows, k // 32
        )
        source = decode_e3m2(codes).to(dtype)
        quantized6 = mxfp6.quantize_mxfp6(source)
        restored_codes, restored_scales = mxfp6.unpack_operand(quantized6)
        torch.testing.assert_close(restored_codes, codes)
        torch.testing.assert_close(
            restored_scales,
            torch.full_like(restored_scales, 0x7F),
        )
    print("PASS native FP16/BF16 16->8 and 16->6 MX quantization")


def test_w6a8_candidate_registry(mxfp6) -> None:
    """Keep the persistent-cache IDs synchronized with the native registry."""
    m, n, k = 17, 136, 128
    a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 6817)
    a6 = mxfp6.pack_operand(a_codes, sfa)
    b = mxfp6.pack_operand(b_codes, sfb)
    a8 = torch.ops.mxfp6.expand_fp6_to_fp8(a6.values, m, k)
    assert torch.ops.mxfp6.w6a8_config_abi(a8) == "native-w6a8-29-v4"
    for out_dtype in (torch.float16, torch.bfloat16):
        torch.ops.mxfp6.set_w6a8_config(
            a8, m, n, k, -1, 1, 0, out_dtype
        )
        reference = torch.ops.mxfp6.gemm_w6a8(
            a8, b.values, a6.scales, b.scales, m, n, k, 1.0, out_dtype
        )
        assert reference.dtype == out_dtype
        for config_id in range(29):
            actual = torch.ops.mxfp6.gemm_w6a8_config(
                a8,
                b.values,
                a6.scales,
                b.scales,
                m,
                n,
                k,
                1.0,
                config_id,
                1,
                0,
                out_dtype,
            )
            torch.testing.assert_close(
                actual, reference, rtol=2e-3, atol=0.5
            )

    try:
        assert torch.ops.mxfp6.set_w6a8_config(
            a8, m, n, k, 5, 2, 1, torch.bfloat16
        )
        overridden = torch.ops.mxfp6.gemm_w6a8(
            a8, b.values, a6.scales, b.scales, m, n, k,
            1.0, torch.bfloat16
        )
        expected = torch.ops.mxfp6.gemm_w6a8_config(
            a8, b.values, a6.scales, b.scales, m, n, k,
            1.0, 5, 2, 1, torch.bfloat16
        )
        torch.testing.assert_close(overridden, expected, rtol=0, atol=0)
    finally:
        torch.ops.mxfp6.set_w6a8_config(
            a8, m, n, k, -1, 1, 0, torch.bfloat16
        )
    print("PASS FP16/BF16 W6A8 candidate ABI and C++ override registry")


def test_float_w6a8_gemm(mxfp6) -> None:
    """Compare fused and prequantized native W6A8 against an FP32 reference."""
    for index, (m, n, k, dtype) in enumerate(
        (
            (1, 128, 128, torch.float16),
            (17, 136, 128, torch.bfloat16),
            (129, 128, 256, torch.float16),
        )
    ):
        generator = torch.Generator(device="cuda").manual_seed(6800 + index)
        source = torch.randn(
            (m, k), generator=generator, device="cuda", dtype=dtype
        )
        _, b_codes, _, sfb = make_problem(m, n, k, 6900 + index)
        b = mxfp6.pack_operand(b_codes, sfb)
        quantized = mxfp6.quantize_activation(source)
        b_values = decode_e3m2(b_codes) * decode_ue8m0(sfb).repeat_interleave(
            32, dim=1
        )
        reference_fp32 = decode_mxfp8(quantized, mxfp6) @ b_values.t()
        for out_dtype in (torch.float16, torch.bfloat16):
            direct = mxfp6.gemm_w6a8(
                quantized, b, out_dtype=out_dtype
            )
            fused = mxfp6.gemm(source, b, out_dtype=out_dtype)
            reference = reference_fp32.to(out_dtype)
            assert direct.dtype == out_dtype
            torch.testing.assert_close(
                direct, reference, rtol=2e-3, atol=0.5
            )
            torch.testing.assert_close(fused, direct, rtol=0, atol=0)
        print(
            "PASS native FP16/BF16 output for "
            f"{str(dtype).removeprefix('torch.')} W6A8 {m}x{n}x{k}"
        )


def test_warmup_api(mxfp6) -> None:
    m, n, k = 17, 136, 128
    source = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = mxfp6.quantize_mxfp6(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    result = mxfp6.warmup_w6a8(
        source,
        weight,
        out_dtype=torch.bfloat16,
        iterations=2,
        autotune=False,
    )
    assert result is None
    result = mxfp6.warmup_w6a8(
        mxfp6.quantize_activation(source),
        weight,
        out_dtype=torch.float16,
        iterations=1,
        autotune=False,
    )
    assert result is None

    try:
        mxfp6.gemm(source, weight, out_dtype=torch.float32)
    except TypeError:
        pass
    else:
        raise AssertionError("float32 output must be rejected")
    try:
        mxfp6.warmup_w6a8(source, weight, iterations=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero warmup iterations must be rejected")
    print("PASS public FP16/BF16 W6A8 warmup API and validation")


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
        actual_bf16 = mxfp6.gemm(a, b, out_dtype=torch.bfloat16)
        reference_bf16 = reference_gemm(
            a_codes, b_codes, sfa, sfb, torch.bfloat16
        )
        assert actual_bf16.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual_bf16, reference_bf16, rtol=2e-3, atol=0.5
        )
        max_abs = (actual - reference).abs().max().item()
        print(f"PASS random-scale GEMM {m}x{n}x{k}: max_abs={max_abs:g}")


def test_large_m_mixed_gemm(mxfp6) -> None:
    """Exercise the profiler-selected E4M3-by-E3M2 large-M dispatch."""
    m, n, k = 2048, 5120, 3072
    a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 1208)
    a = mxfp6.pack_operand(a_codes, sfa)
    b = mxfp6.pack_operand(b_codes, sfb)
    assert a.values.numel() == m * k * 6 // 8
    assert b.values.numel() == n * k * 6 // 8
    actual = mxfp6.gemm(a, b)
    reference = reference_gemm(a_codes, b_codes, sfa, sfb)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.25)
    max_abs = (actual - reference).abs().max().item()
    print(
        "PASS packed-W6 large-M mixed GEMM "
        f"{m}x{n}x{k}: max_abs={max_abs:g}"
    )


def test_small_w6a8_dispatch(mxfp6) -> None:
    """Exercise each retained exact small-batch W6A8 kernel family."""
    shapes = (
        (1, 5120, 3072),    # 128x8, five-stage cooperative
        (1, 5120, 8704),    # 128x8 Stream-K
        (16, 8192, 5120),   # 64x16x256 ping-pong
        (16, 17408, 5120),  # 128x16 cooperative
        (16, 5120, 8704),   # deep-K exact 64x16x256 policy
    )
    for index, (m, n, k) in enumerate(shapes):
        a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 1800 + index)
        a6 = mxfp6.pack_operand(a_codes, sfa)
        b = mxfp6.pack_operand(b_codes, sfb)
        a8 = mxfp6.MXFP8Tensor(
            torch.ops.mxfp6.expand_fp6_to_fp8(a6.values, m, k),
            a6.scales,
            m,
            k,
        )
        actual = mxfp6.gemm_w6a8(a8, b)
        expected = mxfp6.gemm(a6, b)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=0.25)
        max_abs = (actual - expected).abs().max().item()
        print(f"PASS small W6A8 dispatch {m}x{n}x{k}: max_abs={max_abs:g}")


def test_tuned_transition_gemm(mxfp6) -> None:
    """Exercise native and mixed exact dispatch at M=32, 64, and 96."""
    shapes = (
        (32, 8192, 5120),
        (64, 5120, 3072),
        (64, 5120, 8704),
        (96, 7168, 5120),
        (96, 17408, 5120),
    )
    for index, (m, n, k) in enumerate(shapes):
        a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 3200 + index)
        actual = mxfp6.gemm(
            mxfp6.pack_operand(a_codes, sfa),
            mxfp6.pack_operand(b_codes, sfb),
        )
        reference = reference_gemm(a_codes, b_codes, sfa, sfb)
        torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.25)
        max_abs = (actual - reference).abs().max().item()
        print(f"PASS tuned transition GEMM {m}x{n}x{k}: max_abs={max_abs:g}")


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


def test_float_nondefault_stream(mxfp6) -> None:
    m, n, k = 16, 128, 128
    source = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    _, b_codes, _, sfb = make_problem(m, n, k, 2027)
    b = mxfp6.pack_operand(b_codes, sfb)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = mxfp6.gemm(source, b, out_dtype=torch.bfloat16)
        expected = mxfp6.gemm_w6a8(
            mxfp6.quantize_activation(source),
            b,
            out_dtype=torch.bfloat16,
        )
    stream.synchronize()
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("PASS fused W6A8 non-default current-stream semantics")


def test_persistent_workspace(mxfp6) -> None:
    """Exercise resettable Stream-K lanes through CUDA Graph replay."""
    m, n, k = 1, 128, 8704
    a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 2028)
    a6 = mxfp6.pack_operand(a_codes, sfa)
    b = mxfp6.pack_operand(b_codes, sfb)
    a8 = torch.ops.mxfp6.expand_fp6_to_fp8(a6.values, m, k)
    torch.ops.mxfp6.set_w6a8_config(
        a8, m, n, k, 3, 1, 0, torch.bfloat16
    )
    arguments = (
        a8,
        b.values,
        a6.scales,
        b.scales,
        m,
        n,
        k,
        1.0,
        torch.bfloat16,
    )
    split_k_arguments = (
        a6.values,
        b.values,
        a6.scales,
        b.scales,
        m,
        n,
        k,
        1.0,
        torch.float16,
    )

    mxfp6.begin_workspace_planning()
    previous_collection = torch.ops.mxfp6._set_workspace_collection(
        a8, False
    )
    assert previous_collection
    torch.ops.mxfp6.gemm_w6a8(*arguments)
    torch.ops.mxfp6._set_workspace_collection(a8, previous_collection)
    assert mxfp6.workspace_stats()["layouts"] == 0
    reference = torch.ops.mxfp6.gemm_w6a8(*arguments)
    split_k_reference = torch.ops.mxfp6.gemm(*split_k_arguments)
    torch.cuda.synchronize()
    planning_stats = mxfp6.workspace_stats()
    assert planning_stats["layouts"] >= 1
    assert planning_stats["arena_bytes"] > 0

    finalized_stats = mxfp6.finalize_workspace_planning()
    assert finalized_stats["frozen"] == 1
    assert finalized_stats["lanes"] == 1
    torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        eager = torch.ops.mxfp6.gemm_w6a8(*arguments)
        split_k_eager = torch.ops.mxfp6.gemm(*split_k_arguments)
    capture_stream.synchronize()
    torch.testing.assert_close(eager, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        split_k_eager, split_k_reference, rtol=0, atol=0
    )
    assert mxfp6.workspace_stats()["lanes"] == 2

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = torch.ops.mxfp6.gemm_w6a8(*arguments)
    stress_iterations = int(
        os.environ.get("MXFP6_PERSISTENT_STRESS_ITERATIONS", "100")
    )
    if stress_iterations <= 0:
        raise ValueError("MXFP6_PERSISTENT_STRESS_ITERATIONS must be positive")
    for _ in range(stress_iterations):
        graph.replay()
        torch.ops.mxfp6.gemm(*split_k_arguments)
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, reference, rtol=0, atol=0)
    assert mxfp6.workspace_barriers_zero()
    assert mxfp6.workspace_stats()["fallback_launches"] == 0
    print("PASS persistent Stream-K workspace and CUDA Graph stream lanes")


def test_humming_backend(mxfp6) -> None:
    m, n, k = 512, 256, 128
    a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 1206)
    a = mxfp6.pack_operand(a_codes, sfa)
    b = mxfp6.pack_operand(b_codes, sfb)
    humming_b = mxfp6.prepare_humming_weight(b)
    assert (
        humming_b.values.numel() * humming_b.values.element_size()
        == n * k * 6 // 8
    )
    for out_dtype in (torch.float16, torch.bfloat16):
        actual = mxfp6.gemm(a, humming_b, out_dtype=out_dtype)
        expected = mxfp6.gemm(a, b, out_dtype=out_dtype)
        assert actual.dtype == out_dtype
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=0.5)
    print(
        "PASS FP16/BF16 Humming W6A8 bridge correctness and six-bit "
        "weight storage"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument(
        "--humming", action="store_true", help="also JIT and test the Humming backend"
    )
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
    test_fp6_to_fp8(mxfp6)
    test_scale_tools(mxfp6)
    test_dynamic_quantization(mxfp6)
    test_w6a8_candidate_registry(mxfp6)
    test_float_w6a8_gemm(mxfp6)
    test_warmup_api(mxfp6)
    test_random_scale_gemm(mxfp6)
    test_small_w6a8_dispatch(mxfp6)
    test_tuned_transition_gemm(mxfp6)
    test_large_m_mixed_gemm(mxfp6)
    test_nondefault_stream(mxfp6)
    test_float_nondefault_stream(mxfp6)
    test_persistent_workspace(mxfp6)
    if options.humming:
        test_humming_backend(mxfp6)
    print("All MXFP6 tool tests passed")


if __name__ == "__main__":
    main()
