#!/usr/bin/env python3
"""End-to-end tests for the packaged CUDA conversion and GEMM APIs."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


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
        (1, 64),
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
        packed_k_blocks = (k // 32 + 3) // 4 * 4
        assert packed.numel() == ((rows + 127) // 128 * 128 * packed_k_blocks)
        restored = mxfp6.unpack_scales(packed, rows, k)
        torch.testing.assert_close(restored, logical)
    print("PASS UE8M0 CUDA layout pack/unpack with padded rows")


def test_grouped_w6a8(mxfp6) -> None:
    """Verify routed scale packing and native grouped W6A8."""
    counts = (1, 0, 3, 129, 2)
    num_experts = len(counts)
    m, n, k = sum(counts), 128, 128
    expert_offsets = torch.tensor(
        (0, *torch.tensor(counts).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int64,
    )
    generator = torch.Generator(device="cuda").manual_seed(12035)
    activation_source = torch.randn(
        (m, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_source = torch.randn(
        (num_experts * n, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    activation = mxfp6.quantize_mxfp8(activation_source)
    activation_logical_scales = mxfp6.unpack_scales(activation.scales, m, k)
    activation_scales, scale_offsets = mxfp6.pack_grouped_scales(
        activation_logical_scales, expert_offsets
    )
    assert scale_offsets.cpu().tolist() == [0, 128, 128, 256, 512, 640]
    for expert, rows in enumerate(counts):
        if rows == 0:
            continue
        first = int(scale_offsets[expert].item())
        packed_rows = (rows + 127) // 128 * 128
        restored = mxfp6.unpack_scales(
            activation_scales[first * (k // 32) : (first + packed_rows) * (k // 32)],
            rows,
            k,
        )
        row_start = int(expert_offsets[expert].item())
        torch.testing.assert_close(
            restored,
            activation_logical_scales[row_start : row_start + rows],
        )

    weight = mxfp6.quantize_mxfp6(weight_source)
    weight_values = weight.values.view(num_experts, n, k * 3 // 4)
    for out_dtype in (torch.float16, torch.bfloat16):
        actual = mxfp6.grouped_gemm_w6a8(
            activation.values.view(torch.float8_e4m3fn),
            activation_scales,
            weight_values,
            weight.scales,
            expert_offsets,
            scale_offsets,
            out_dtype=out_dtype,
        )
        activation_values = (
            activation.values.view(m, k).view(torch.float8_e4m3fn).float()
        )
        activation_dequant = activation_values * decode_ue8m0(
            activation_logical_scales
        ).repeat_interleave(32, dim=1)
        weight_codes = mxfp6.unpack_fp6(weight.values, num_experts * n, k)
        weight_logical_scales = mxfp6.unpack_scales(weight.scales, num_experts * n, k)
        weight_dequant = decode_e3m2(weight_codes) * decode_ue8m0(
            weight_logical_scales
        ).repeat_interleave(32, dim=1)
        reference = torch.empty_like(actual)
        for expert, rows in enumerate(counts):
            row_start = int(expert_offsets[expert].item())
            row_end = row_start + rows
            reference[row_start:row_end] = (
                activation_dequant[row_start:row_end]
                @ weight_dequant[expert * n : (expert + 1) * n].t()
            ).to(out_dtype)
        torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.5)
    print("PASS grouped scale packing and native SM120 W6A8 MoE GEMM")


def test_grouped_w6a8_k64_transposed(mxfp6) -> None:
    """Verify compact K64 weights in the transposed FP6 TMA layout."""
    counts = (1, 3, 129)
    num_experts = len(counts)
    m, n, k = sum(counts), 128, 64
    expert_offsets = torch.tensor(
        (0, *torch.tensor(counts).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int64,
    )
    generator = torch.Generator(device="cuda").manual_seed(12064)
    activation_source = torch.randn(
        (m, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_source = torch.randn(
        (num_experts * n, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    activation = mxfp6.quantize_mxfp8(activation_source)
    activation_logical_scales = mxfp6.unpack_scales(activation.scales, m, k)
    activation_scales, scale_offsets = mxfp6.pack_grouped_scales(
        activation_logical_scales, expert_offsets
    )
    weight = mxfp6.quantize_mxfp6(weight_source)
    weight_codes = mxfp6.unpack_fp6(weight.values, num_experts * n, k).view(
        num_experts, n, k
    )
    weight_values = mxfp6.to_mma_k64_weight(
        weight.values.view(num_experts, n, k * 3 // 4)
    )

    actual = mxfp6.grouped_gemm_w6a8(
        activation.values.view(torch.float8_e4m3fn),
        activation_scales,
        weight_values,
        weight.scales,
        expert_offsets,
        scale_offsets,
        out_dtype=torch.bfloat16,
    )
    activation_dequant = activation.values.view(m, k).view(
        torch.float8_e4m3fn
    ).float() * decode_ue8m0(activation_logical_scales).repeat_interleave(32, dim=1)
    weight_logical_scales = mxfp6.unpack_scales(weight.scales, num_experts * n, k)
    weight_dequant = decode_e3m2(weight_codes.view(-1, k))
    weight_dequant *= decode_ue8m0(weight_logical_scales).repeat_interleave(32, dim=1)
    reference = torch.empty_like(actual)
    for expert, rows in enumerate(counts):
        row_start = int(expert_offsets[expert].item())
        row_end = row_start + rows
        reference[row_start:row_end] = (
            activation_dequant[row_start:row_end]
            @ weight_dequant[expert * n : (expert + 1) * n].t()
        ).to(torch.bfloat16)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.5)
    print("PASS compact transposed K64 grouped W6A8")


def test_array_w6a8(mxfp6) -> None:
    """Verify equal-shape pointer-array W6A8 in unsorted route order."""
    tokens, topk, num_experts, n, k = 4, 2, 5, 128, 128
    generator = torch.Generator(device="cuda").manual_seed(12036)
    source = torch.randn(
        (tokens, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_source = torch.randn(
        (num_experts * n, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor(
        [[4, 0], [2, 3], [1, 4], [0, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    activation, logical_scales = mxfp6.quantize_mxfp8_logical(source)
    weight = mxfp6.quantize_mxfp6(weight_source)
    weight_values = weight.values.view(num_experts, n, k * 3 // 4)

    activation_dequant = activation.float() * decode_ue8m0(
        logical_scales
    ).repeat_interleave(32, dim=1)
    weight_codes = mxfp6.unpack_fp6(weight.values, num_experts * n, k)
    weight_logical_scales = mxfp6.unpack_scales(weight.scales, num_experts * n, k)
    weight_dequant = decode_e3m2(weight_codes) * decode_ue8m0(
        weight_logical_scales
    ).repeat_interleave(32, dim=1)

    routes = tokens * topk
    for out_dtype in (torch.float16, torch.bfloat16):
        actual = torch.empty((routes, n), device="cuda", dtype=out_dtype)
        mxfp6.array_gemm_w6a8_out(
            actual,
            activation,
            logical_scales,
            weight_values,
            weight.scales,
            topk_ids,
        )
        reference = torch.stack(
            [
                activation_dequant[route // topk]
                @ weight_dequant[
                    int(topk_ids.flatten()[route]) * n : (
                        int(topk_ids.flatten()[route]) + 1
                    )
                    * n
                ].t()
                for route in range(routes)
            ]
        ).to(out_dtype)
        torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.5)

    combined_routes = tokens * (topk + 1)
    combined = torch.empty(
        (combined_routes, n),
        device="cuda",
        dtype=torch.bfloat16,
    )
    mxfp6.array_gemm_w6a8_out(
        combined,
        activation,
        logical_scales,
        weight_values,
        weight.scales,
        topk_ids,
        include_shared=True,
    )
    combined_reference = torch.stack(
        [
            activation_dequant[route // (topk + 1)]
            @ weight_dequant[
                (
                    num_experts - 1
                    if route % (topk + 1) == topk
                    else int(
                        topk_ids[
                            route // (topk + 1),
                            route % (topk + 1),
                        ]
                    )
                )
                * n : (
                    num_experts
                    if route % (topk + 1) == topk
                    else int(
                        topk_ids[
                            route // (topk + 1),
                            route % (topk + 1),
                        ]
                    )
                    + 1
                )
                * n
            ].t()
            for route in range(combined_routes)
        ]
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        combined,
        combined_reference,
        rtol=2e-3,
        atol=0.5,
    )

    topk_weights = torch.rand(
        (tokens, topk),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    shared_gate = torch.randn(
        tokens,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    reduced = torch.empty(
        (tokens, n),
        device="cuda",
        dtype=torch.bfloat16,
    )

    def reduce_array() -> None:
        mxfp6.moe_reduce_array_out(
            reduced,
            combined,
            topk_weights,
            shared_gate,
        )

    reduce_array()
    combined_view = combined.view(tokens, topk + 1, n).float()
    reduce_reference = (
        (combined_view[:, :topk] * topk_weights[:, :, None]).sum(dim=1)
        + torch.sigmoid(shared_gate.float())[:, None] * combined_view[:, topk]
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        reduced,
        reduce_reference,
        rtol=0,
        atol=0,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reduce_array()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        reduced,
        reduce_reference,
        rtol=0,
        atol=0,
    )
    print("PASS SM120 pointer-array routed/shared W6A8 and fused reduce")


def test_fused_array_silu_mxfp8(mxfp6, k: int = 2048) -> None:
    """Verify fused W1 against the unfused BF16 gate/up path."""
    tokens, topk, num_experts, intermediate = 1, 2, 5, 256
    gate_up = intermediate * 2
    routes = tokens * (topk + 1)
    generator = torch.Generator(device="cuda").manual_seed(12037)
    source = torch.randn(
        (tokens, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_source = torch.randn(
        (num_experts * gate_up, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor(
        [[3, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    activation, logical_scales = mxfp6.quantize_mxfp8_logical(source)
    weight = mxfp6.quantize_mxfp6(weight_source)
    weight_values = weight.values.view(num_experts, gate_up, k * 3 // 4)

    gate_up_output = torch.empty(
        (routes, gate_up),
        device="cuda",
        dtype=torch.bfloat16,
    )
    mxfp6.array_gemm_w6a8_out(
        gate_up_output,
        activation,
        logical_scales,
        weight_values,
        weight.scales,
        topk_ids,
        include_shared=True,
    )
    reference, reference_scales = mxfp6.silu_and_mul_mxfp8_logical(gate_up_output)
    actual = torch.empty(
        reference.shape,
        device="cuda",
        dtype=torch.uint8,
    )
    actual_scales = torch.empty_like(reference_scales)
    mxfp6.array_gemm_w6a8_silu_mxfp8_out(
        actual,
        actual_scales,
        activation,
        logical_scales,
        weight_values,
        weight.scales,
        topk_ids,
        include_shared=True,
    )
    torch.testing.assert_close(actual_scales, reference_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        actual,
        reference.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    padded_weight = mxfp6.pad_fp6(weight_values)
    padded_actual = torch.empty_like(actual)
    padded_scales = torch.empty_like(actual_scales)
    mxfp6.array_gemm_w6a8_silu_mxfp8_out(
        padded_actual,
        padded_scales,
        activation,
        logical_scales,
        padded_weight,
        weight.scales,
        topk_ids,
        include_shared=True,
    )
    torch.testing.assert_close(padded_scales, reference_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        padded_actual,
        reference.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    print("PASS fused swapAB W1 SiLU/mul MXFP8")


def test_fused_array_w2_reduce(mxfp6) -> None:
    """Verify top-8 routed/shared W2 and reduction fusion."""
    tokens, topk, num_experts, n, k = 1, 8, 10, 128, 256
    routes = tokens * (topk + 1)
    generator = torch.Generator(device="cuda").manual_seed(12038)
    source = torch.randn(
        (routes, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_source = torch.randn(
        (num_experts * n, k),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor(
        [[7, 2, 0, 5, 3, 8, 1, 6]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.rand(
        (tokens, topk),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    shared_gate = torch.randn(
        (tokens, num_experts),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    activation, logical_scales = mxfp6.quantize_mxfp8_logical(source)
    weight = mxfp6.quantize_mxfp6(weight_source)
    weight_values = weight.values.view(num_experts, n, k * 3 // 4)

    combined = torch.empty(
        (routes, n),
        device="cuda",
        dtype=torch.bfloat16,
    )
    mxfp6.array_gemm_w6a8_out(
        combined,
        activation,
        logical_scales,
        weight_values,
        weight.scales,
        topk_ids,
        include_shared=True,
    )
    reference = torch.empty(
        (tokens, n),
        device="cuda",
        dtype=torch.bfloat16,
    )
    mxfp6.moe_reduce_array_out(
        reference,
        combined,
        topk_weights,
        shared_gate,
    )
    actual = torch.empty_like(reference)
    mxfp6.array_gemm_w6a8_reduce_out(
        actual,
        activation,
        logical_scales,
        weight_values,
        weight.scales,
        topk_ids,
        topk_weights,
        shared_gate,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    padded_weight = mxfp6.pad_fp6(weight_values)
    padded_actual = torch.empty_like(actual)
    mxfp6.array_gemm_w6a8_reduce_out(
        padded_actual,
        activation,
        logical_scales,
        padded_weight,
        weight.scales,
        topk_ids,
        topk_weights,
        shared_gate,
    )
    torch.testing.assert_close(padded_actual, reference, rtol=0, atol=0)

    external_actual = torch.empty_like(actual)
    mxfp6.array_gemm_w6a8_reduce_out(
        external_actual,
        activation[:topk].contiguous(),
        logical_scales[:topk].contiguous(),
        weight_values[: num_experts - 1].contiguous(),
        weight.scales,
        topk_ids,
        topk_weights,
        shared_gate,
        combined[topk:].contiguous(),
    )
    torch.testing.assert_close(external_actual, reference, rtol=0, atol=0)
    print("PASS fused top-8 W2 reduce with internal/external shared expert")


def test_qwen35_b1_specialization(mxfp6) -> None:
    """Verify Qwen3.5 TP2 B=1..8 W1 and the graph-captured B=1 layer."""
    generator = torch.Generator(device="cuda").manual_seed(12039)
    hidden = torch.randn(
        (1, 2048),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate_weight = (
        torch.randn(
            (257, 2048),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)
    quantized = torch.empty_like(hidden, dtype=torch.uint8)
    input_scales = torch.empty((1, 64), device="cuda", dtype=torch.uint8)
    routed_logits = torch.empty((1, 256), device="cuda", dtype=torch.bfloat16)
    topk_weights = torch.empty((1, 8), device="cuda", dtype=torch.float32)
    topk_ids = torch.empty((1, 8), device="cuda", dtype=torch.int32)
    shared_gate = torch.empty((1,), device="cuda", dtype=torch.bfloat16)
    w1_partial = torch.empty((576, 64), device="cuda", dtype=torch.float32)
    mxfp6.qwen35_router_quant_out(
        quantized,
        input_scales,
        routed_logits,
        topk_weights,
        topk_ids,
        shared_gate,
        hidden,
        gate_weight,
    )
    separate_quantized = torch.empty_like(quantized)
    separate_input_scales = torch.empty_like(input_scales)
    separate_routed_logits = torch.empty_like(routed_logits)
    separate_topk_weights = torch.empty_like(topk_weights)
    separate_topk_ids = torch.empty_like(topk_ids)
    separate_shared_gate = torch.empty_like(shared_gate)
    mxfp6.qwen35_router_quant_out(
        separate_quantized,
        separate_input_scales,
        separate_routed_logits,
        separate_topk_weights,
        separate_topk_ids,
        separate_shared_gate,
        hidden,
        gate_weight[:256],
        shared_gate_weight=gate_weight[256:],
    )
    for separate, combined in (
        (separate_quantized, quantized),
        (separate_input_scales, input_scales),
        (separate_routed_logits, routed_logits),
        (separate_topk_weights, topk_weights),
        (separate_topk_ids, topk_ids),
        (separate_shared_gate, shared_gate),
    ):
        torch.testing.assert_close(separate, combined, rtol=0, atol=0)

    expected_quantized, expected_scales = mxfp6.quantize_mxfp8_logical(hidden)
    expected_logits = F.linear(hidden, gate_weight[:256])
    expected_shared_gate = F.linear(hidden, gate_weight[256:]).flatten()
    expected_weights, expected_ids = torch.sort(
        torch.softmax(expected_logits.float(), dim=-1),
        dim=-1,
        descending=True,
        stable=True,
    )
    expected_weights = expected_weights[:, :8]
    expected_ids = expected_ids[:, :8]
    torch.testing.assert_close(quantized, expected_quantized.view(torch.uint8))
    torch.testing.assert_close(input_scales, expected_scales)
    torch.testing.assert_close(routed_logits, expected_logits)
    torch.testing.assert_close(shared_gate, expected_shared_gate)
    torch.testing.assert_close(topk_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(
        topk_weights,
        expected_weights,
        rtol=1e-6,
        atol=1e-8,
    )
    mxfp6.qwen35_router_quant_out(
        quantized,
        input_scales,
        routed_logits,
        topk_weights,
        topk_ids,
        shared_gate,
        hidden,
        gate_weight,
        renormalize=True,
    )
    expected_renormalized = expected_weights / expected_weights.sum(
        dim=-1,
        keepdim=True,
    )
    torch.testing.assert_close(
        topk_weights,
        expected_renormalized,
        rtol=1e-6,
        atol=1e-8,
    )
    mxfp6.qwen35_router_quant_out(
        quantized,
        input_scales,
        routed_logits,
        topk_weights,
        topk_ids,
        shared_gate,
        hidden,
        gate_weight,
    )

    for batch in (2, 4, 8):
        hidden_small = torch.randn(
            (batch, 2048),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        quantized_small = torch.empty_like(hidden_small, dtype=torch.uint8)
        scales_small = torch.empty(
            (batch, 64), device="cuda", dtype=torch.uint8
        )
        logits_small = torch.empty(
            (batch, 256), device="cuda", dtype=torch.bfloat16
        )
        weights_small = torch.empty(
            (batch, 8), device="cuda", dtype=torch.float32
        )
        ids_small = torch.empty(
            (batch, 8), device="cuda", dtype=torch.int32
        )
        shared_gate_small = torch.empty(
            (batch,), device="cuda", dtype=torch.bfloat16
        )
        mxfp6.qwen35_router_quant_out(
            quantized_small,
            scales_small,
            logits_small,
            weights_small,
            ids_small,
            shared_gate_small,
            hidden_small,
            gate_weight,
        )
        expected_quantized_small, expected_scales_small = (
            mxfp6.quantize_mxfp8_logical(hidden_small)
        )
        expected_logits_small = F.linear(hidden_small, gate_weight[:256])
        expected_shared_small = F.linear(
            hidden_small, gate_weight[256:]
        ).flatten()
        router_weights_small, router_ids_small = torch.sort(
            torch.softmax(logits_small.float(), dim=-1),
            dim=-1,
            descending=True,
            stable=True,
        )
        torch.testing.assert_close(
            quantized_small,
            expected_quantized_small.view(torch.uint8),
        )
        torch.testing.assert_close(scales_small, expected_scales_small)
        # The batched router uses BF16 accumulation and can differ from cuBLAS
        # by one representable BF16 step. Validate its numerical envelope
        # against F.linear, then require top-k to match the logits it emitted.
        torch.testing.assert_close(
            logits_small,
            expected_logits_small,
            rtol=2e-2,
            atol=6.25e-2,
        )
        torch.testing.assert_close(
            shared_gate_small,
            expected_shared_small,
            rtol=2e-2,
            atol=6.25e-2,
        )
        torch.testing.assert_close(
            ids_small,
            router_ids_small[:, :8].to(torch.int32),
        )
        torch.testing.assert_close(
            weights_small,
            router_weights_small[:, :8],
            rtol=1e-6,
            atol=1e-8,
        )

    w1 = torch.randint(
        0,
        256,
        (257, 512, 1536),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    w1_scales = torch.full(
        (257 * 512 * 64,),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    reference_activation = torch.empty((9, 256), device="cuda", dtype=torch.uint8)
    reference_activation_scales = torch.empty((9, 8), device="cuda", dtype=torch.uint8)
    mxfp6.array_gemm_w6a8_silu_mxfp8_out(
        reference_activation,
        reference_activation_scales,
        quantized,
        input_scales,
        w1,
        w1_scales,
        topk_ids,
        include_shared=True,
    )
    activation = torch.empty_like(reference_activation)
    activation_scales = torch.empty_like(reference_activation_scales)
    mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
        activation,
        activation_scales,
        w1_partial,
        quantized,
        input_scales,
        w1,
        w1_scales,
        topk_ids,
    )
    torch.testing.assert_close(activation, reference_activation)
    torch.testing.assert_close(activation_scales, reference_activation_scales)
    combined_w1_partial = w1_partial.clone()
    separate_activation = torch.empty_like(activation)
    separate_activation_scales = torch.empty_like(activation_scales)
    separate_w1_partial = torch.empty_like(w1_partial)
    mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
        separate_activation,
        separate_activation_scales,
        separate_w1_partial,
        quantized,
        input_scales,
        w1[:256],
        w1_scales[: 256 * 512 * 64],
        topk_ids,
        shared_weight=w1[256:],
        shared_weight_scales=w1_scales[256 * 512 * 64 :],
    )
    torch.testing.assert_close(
        separate_activation, activation, rtol=0, atol=0
    )
    torch.testing.assert_close(
        separate_activation_scales, activation_scales, rtol=0, atol=0
    )
    torch.testing.assert_close(
        separate_w1_partial.flatten()[: 144 * 128],
        combined_w1_partial.flatten()[: 144 * 128],
        rtol=0,
        atol=0,
    )

    base_ids = torch.tensor(
        [0, 7, 15, 31, 63, 95, 127, 255],
        device="cuda",
        dtype=torch.int32,
    )
    for batch in (2, 4, 8):
        hidden_small = torch.randn(
            (batch, 2048),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        quantized_small, input_scales_small = mxfp6.quantize_mxfp8_logical(hidden_small)
        topk_ids_small = (
            base_ids[None, :]
            + torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
        ) % 256
        reference_activation_small = torch.empty(
            (batch * 9, 256),
            device="cuda",
            dtype=torch.uint8,
        )
        reference_scales_small = torch.empty(
            (batch * 9, 8),
            device="cuda",
            dtype=torch.uint8,
        )
        mxfp6.array_gemm_w6a8_silu_mxfp8_out(
            reference_activation_small,
            reference_scales_small,
            quantized_small,
            input_scales_small,
            w1,
            w1_scales,
            topk_ids_small,
            include_shared=True,
        )
        activation_small = torch.empty_like(reference_activation_small)
        activation_scales_small = torch.empty_like(reference_scales_small)
        mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
            activation_small,
            activation_scales_small,
            w1_partial,
            quantized_small,
            input_scales_small,
            w1,
            w1_scales,
            topk_ids_small,
        )
        torch.testing.assert_close(
            activation_small,
            reference_activation_small,
        )
        torch.testing.assert_close(
            activation_scales_small,
            reference_scales_small,
        )
        if batch <= 4:
            separate_activation_small = torch.empty_like(
                activation_small
            )
            separate_scales_small = torch.empty_like(
                activation_scales_small
            )
            mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
                separate_activation_small,
                separate_scales_small,
                w1_partial,
                quantized_small,
                input_scales_small,
                w1[:256],
                w1_scales[: 256 * 512 * 64],
                topk_ids_small,
                shared_weight=w1[256:],
                shared_weight_scales=w1_scales[256 * 512 * 64 :],
            )
            torch.testing.assert_close(
                separate_activation_small,
                activation_small,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                separate_scales_small,
                activation_scales_small,
                rtol=0,
                atol=0,
            )

    routed_w1 = w1[:256]
    routed_w1_scales = w1_scales[: 256 * 512 * 64]
    for batch in (1, 2, 4, 8):
        hidden_small = torch.randn(
            (batch, 2048),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        quantized_small, input_scales_small = mxfp6.quantize_mxfp8_logical(
            hidden_small
        )
        topk_ids_small = (
            base_ids[None, :]
            + torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
        ) % 256
        reference_routed = torch.empty(
            (batch * 8, 256), device="cuda", dtype=torch.uint8
        )
        reference_routed_scales = torch.empty(
            (batch * 8, 8), device="cuda", dtype=torch.uint8
        )
        mxfp6.array_gemm_w6a8_silu_mxfp8_out(
            reference_routed,
            reference_routed_scales,
            quantized_small,
            input_scales_small,
            routed_w1,
            routed_w1_scales,
            topk_ids_small,
            include_shared=False,
        )
        routed = torch.empty_like(reference_routed)
        routed_scales = torch.empty_like(reference_routed_scales)
        mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
            routed,
            routed_scales,
            w1_partial,
            quantized_small,
            input_scales_small,
            routed_w1,
            routed_w1_scales,
            topk_ids_small,
            include_shared=False,
        )
        torch.testing.assert_close(routed, reference_routed)
        torch.testing.assert_close(routed_scales, reference_routed_scales)

    w2 = torch.randint(
        0,
        256,
        (257, 2048, 192),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    w2_scales = torch.full(
        (257 * 2048 * 8,),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    for batch in (2, 4):
        activation_small = torch.randint(
            0,
            120,
            (batch * 9, 256),
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )
        activation_scales_small = torch.full(
            (batch * 9, 8),
            127,
            device="cuda",
            dtype=torch.uint8,
        )
        topk_ids_small = (
            base_ids[None, :]
            + torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
        ) % 256
        topk_weights_small = torch.rand(
            (batch, 8),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        topk_weights_small /= topk_weights_small.sum(dim=1, keepdim=True)
        shared_gate_small = torch.randn(
            (batch,),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        combined_output_small = torch.empty(
            (batch, 2048), device="cuda", dtype=torch.bfloat16
        )
        separate_output_small = torch.empty_like(combined_output_small)
        mxfp6.array_gemm_w6a8_reduce_out(
            combined_output_small,
            activation_small,
            activation_scales_small,
            w2,
            w2_scales,
            topk_ids_small,
            topk_weights_small,
            shared_gate_small,
        )
        mxfp6.array_gemm_w6a8_reduce_out(
            separate_output_small,
            activation_small,
            activation_scales_small,
            w2[:256],
            w2_scales[: 256 * 2048 * 8],
            topk_ids_small,
            topk_weights_small,
            shared_gate_small,
            shared_weight=w2[256:],
            shared_weight_scales=w2_scales[256 * 2048 * 8 :],
        )
        torch.testing.assert_close(
            separate_output_small,
            combined_output_small,
            rtol=0,
            atol=0,
        )
        if batch == 4:
            vector_output = torch.empty_like(combined_output_small)
            mxfp6.array_gemm_w6a8_reduce_out(
                vector_output,
                activation_small,
                activation_scales_small,
                w2[:256],
                w2_scales[: 256 * 2048 * 8],
                topk_ids_small,
                topk_weights_small,
                shared_gate_small,
                shared_weight=w2[256:],
                shared_weight_scales=w2_scales[256 * 2048 * 8 :],
                use_packed_vector_loads=True,
            )
            torch.testing.assert_close(
                vector_output,
                separate_output_small,
                rtol=0,
                atol=0,
            )

            shared_weight = w2[256:]
            misaligned_storage = torch.empty(
                shared_weight.numel() + 4,
                device="cuda",
                dtype=torch.uint8,
            )
            misaligned_shared_weight = misaligned_storage[4:].view_as(
                shared_weight
            )
            misaligned_shared_weight.copy_(shared_weight)
            fallback_output = torch.empty_like(combined_output_small)
            mxfp6.array_gemm_w6a8_reduce_out(
                fallback_output,
                activation_small,
                activation_scales_small,
                w2[:256],
                w2_scales[: 256 * 2048 * 8],
                topk_ids_small,
                topk_weights_small,
                shared_gate_small,
                shared_weight=misaligned_shared_weight,
                shared_weight_scales=w2_scales[256 * 2048 * 8 :],
                use_packed_vector_loads=True,
            )
            torch.testing.assert_close(
                fallback_output,
                separate_output_small,
                rtol=0,
                atol=0,
            )
    reference = torch.empty((1, 2048), device="cuda", dtype=torch.bfloat16)
    mxfp6.array_gemm_w6a8_reduce_out(
        reference,
        activation,
        activation_scales,
        w2,
        w2_scales,
        topk_ids,
        topk_weights,
        shared_gate,
    )
    output = torch.empty_like(reference)
    w2_partial = torch.empty((256, 9, 16), device="cuda", dtype=torch.float32)
    workspace = mxfp6.Qwen35MoeB1Workspace(
        quantized=quantized,
        input_scales=input_scales,
        routed_logits=routed_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_gate=shared_gate,
        w1_partial=w1_partial,
        activation=activation,
        activation_scales=activation_scales,
        w2_partial=w2_partial,
        output=output,
    )

    def run_specialized() -> None:
        mxfp6.qwen35_moe_b1_out(
            workspace,
            hidden,
            gate_weight,
            w1,
            w1_scales,
            w2,
            w2_scales,
        )

    run_specialized()
    torch.testing.assert_close(output, reference)
    combined_w2_partial = w2_partial.clone()
    separate_output = torch.empty_like(output)
    separate_w2_partial = torch.empty_like(w2_partial)
    mxfp6.qwen35_w2_splitk_reduce_out(
        separate_output,
        separate_w2_partial,
        activation,
        activation_scales,
        w2[:256],
        w2_scales[: 256 * 2048 * 8],
        topk_ids,
        topk_weights,
        shared_gate,
        shared_weight=w2[256:],
        shared_weight_scales=w2_scales[256 * 2048 * 8 :],
    )
    torch.testing.assert_close(separate_output, output, rtol=0, atol=0)
    torch.testing.assert_close(
        separate_w2_partial, combined_w2_partial, rtol=0, atol=0
    )

    zero_topk_weights = torch.zeros_like(topk_weights)
    gate_open = torch.full_like(shared_gate, 20)
    gate_half = torch.zeros_like(shared_gate)
    shared_open = torch.empty_like(output)
    shared_half = torch.empty_like(output)
    for gate, gated_output in (
        (gate_open, shared_open),
        (gate_half, shared_half),
    ):
        mxfp6.qwen35_w2_splitk_reduce_out(
            gated_output,
            separate_w2_partial,
            activation,
            activation_scales,
            w2[:256],
            w2_scales[: 256 * 2048 * 8],
            topk_ids,
            zero_topk_weights,
            gate,
            shared_weight=w2[256:],
            shared_weight_scales=w2_scales[256 * 2048 * 8 :],
        )
    torch.testing.assert_close(
        shared_half,
        (shared_open.float() * 0.5).to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        shared_half,
        (shared_open.float() * 0.25).to(torch.bfloat16),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_specialized()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference)
    print("PASS Qwen3.5 TP2 B=1 fused router/W1/W2 CUDA Graph")


def test_route_mxfp8(mxfp6) -> None:
    """Verify fused routing, inverse indices, and grouped scale packing."""
    tokens, topk, k = 5, 3, 128
    source = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16)
    values, logical_scales = mxfp6.quantize_mxfp8_logical(source)
    cases = (
        (
            torch.tensor(
                [[0, 1, 3], [2, 0, 1], [3, 3, 0], [1, 2, 3], [2, 1, 0]],
                device="cuda",
                dtype=torch.int64,
            ),
            None,
            4,
        ),
        (
            torch.tensor(
                [[0, 1, 5], [2, 4, 1], [5, 3, 0], [1, 2, 5], [4, 3, 0]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [0, -1, 1, 2, -1, 3],
                device="cuda",
                dtype=torch.int32,
            ),
            4,
        ),
    )
    for topk_ids, expert_map, local_experts in cases:
        routes = tokens * topk
        permuted = torch.empty((routes, k), device="cuda", dtype=torch.float8_e4m3fn)
        packed, offsets, scale_offsets, inverse = mxfp6.route_mxfp8_out(
            values,
            logical_scales,
            topk_ids,
            expert_map,
            local_experts,
            permuted,
        )
        mapped = topk_ids.to(torch.int64)
        if expert_map is not None:
            mapped = expert_map[mapped].to(torch.int64)
        valid = (mapped >= 0) & (mapped < local_experts)
        counts = torch.stack(
            [(mapped[valid] == expert).sum() for expert in range(local_experts)]
        )
        expected_offsets = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int64),
                counts.cumsum(0),
            )
        )
        torch.testing.assert_close(offsets, expected_offsets)
        valid_rows = int(offsets[-1].item())

        flat_mapped = mapped.flatten()
        flat_inverse = inverse.to(torch.int64)
        for route in range(routes):
            expert = int(flat_mapped[route].item())
            destination = int(flat_inverse[route].item())
            if expert < 0 or expert >= local_experts:
                assert destination >= valid_rows
                continue
            assert int(offsets[expert].item()) <= destination
            assert destination < int(offsets[expert + 1].item())
            torch.testing.assert_close(permuted[destination], values[route // topk])

        for expert in range(local_experts):
            rows = int(counts[expert].item())
            if rows == 0:
                continue
            first = int(scale_offsets[expert].item())
            padded_rows = (rows + 127) // 128 * 128
            restored = mxfp6.unpack_scales(
                packed[first * (k // 32) : (first + padded_rows) * (k // 32)],
                rows,
                k,
            )
            for route in range(routes):
                if int(flat_mapped[route].item()) != expert:
                    continue
                destination = int(flat_inverse[route].item())
                local_row = destination - int(offsets[expert].item())
                torch.testing.assert_close(
                    restored[local_row], logical_scales[route // topk]
                )
    print("PASS fused MXFP8 routing, inverse map, and packed scales")


def test_shared_group_and_reduce(mxfp6) -> None:
    """Verify the always-hit shared group and package-owned final reduce."""
    tokens, topk, hidden_size = 5, 3, 128
    routed_experts = 4
    source = torch.randn((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    values, logical_scales = mxfp6.quantize_mxfp8_logical(source)
    topk_ids = torch.tensor(
        [[3, 1, 0], [2, 3, 1], [1, 0, 2], [3, 2, 0], [0, 1, 3]],
        device="cuda",
        dtype=torch.int32,
    )
    routes = tokens * topk
    permuted = torch.empty(
        (routes + tokens, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    (
        packed_scales,
        expert_offsets,
        scale_offsets,
        inverse,
    ) = mxfp6.route_mxfp8_out(
        values,
        logical_scales,
        topk_ids,
        None,
        routed_experts + 1,
        permuted,
        include_shared=True,
    )
    del packed_scales, scale_offsets
    shared_start = int(expert_offsets[routed_experts].item())
    assert shared_start == routes
    assert int(expert_offsets[-1].item()) == routes + tokens
    torch.testing.assert_close(
        permuted[shared_start : shared_start + tokens],
        values,
    )

    routed_output = torch.randn(
        (routes + tokens, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_weights = torch.rand((tokens, topk), device="cuda", dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    shared_gate = torch.randn((tokens, 1), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)

    def run_reduce() -> None:
        mxfp6.moe_reduce_out(
            output,
            routed_output,
            topk_weights,
            inverse,
            routed_output[shared_start : shared_start + tokens],
            shared_gate,
        )

    run_reduce()
    selected = routed_output[inverse.view(tokens, topk).long()].float()
    expected = (
        (selected * topk_weights[..., None]).sum(dim=1)
        + torch.sigmoid(shared_gate.float())
        * routed_output[shared_start : shared_start + tokens].float()
    ).to(source.dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_reduce()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print("PASS shared grouped routing and independent fused MoE reduce")


def make_problem(m: int, n: int, k: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a_codes = torch.randint(
        0, 12, (m, k), generator=generator, device="cuda", dtype=torch.uint8
    )
    b_codes = torch.randint(
        0, 12, (n, k), generator=generator, device="cuda", dtype=torch.uint8
    )
    a_codes |= (
        torch.randint(
            0, 2, (m, k), generator=generator, device="cuda", dtype=torch.uint8
        )
        << 5
    )
    b_codes |= (
        torch.randint(
            0, 2, (n, k), generator=generator, device="cuda", dtype=torch.uint8
        )
        << 5
    )
    sfa = torch.randint(
        125,
        129,
        (m, k // 32),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    sfb = torch.randint(
        125,
        129,
        (n, k // 32),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
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
    values = (
        operand.values.view(operand.rows, operand.k).view(torch.float8_e4m3fn).float()
    )
    scales = mxfp6.unpack_scales(operand.scales, operand.rows, operand.k)
    return values * decode_ue8m0(scales).repeat_interleave(32, dim=1)


def test_dynamic_quantization(mxfp6) -> None:
    """Verify the real 16->8 and 16->6 activation mappings."""
    for dtype in (torch.float16, torch.bfloat16):
        for rows, k in ((1, 128), (7, 256), (129, 128)):
            generator = torch.Generator(device="cuda").manual_seed(rows + k)
            source = (
                torch.randn(
                    (rows, k),
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                * 6.0
            ).to(dtype)
            quantized = mxfp6.quantize_mxfp8(source)
            logical_scales = mxfp6.unpack_scales(quantized.scales, rows, k)
            scales = decode_ue8m0(logical_scales).repeat_interleave(32, dim=1)
            expected = (
                (source.float() / scales).to(torch.float8_e4m3fn).view(torch.uint8)
            )
            actual = quantized.values.view(rows, k)
            torch.testing.assert_close(actual, expected)
            dual_values, dual_logical, dual_packed = (
                mxfp6.quantize_mxfp8_dual(source)
            )
            torch.testing.assert_close(
                dual_values.view(torch.uint8),
                quantized.values.view(rows, k),
            )
            torch.testing.assert_close(dual_logical, logical_scales)
            torch.testing.assert_close(dual_packed, quantized.scales)
            dual_out = torch.empty_like(actual)
            logical_out = torch.empty_like(logical_scales)
            packed_out = torch.full_like(quantized.scales, 0x7F)
            mxfp6.quantize_mxfp8_dual_out(
                dual_out,
                logical_out,
                packed_out,
                source,
            )
            torch.testing.assert_close(dual_out, actual)
            torch.testing.assert_close(logical_out, logical_scales)
            torch.testing.assert_close(packed_out, quantized.scales)

            silu_values, silu_logical = (
                mxfp6.silu_and_mul_mxfp8_logical(source)
            )
            silu_packed = mxfp6.pack_scales(silu_logical)
            silu_out = torch.empty_like(silu_values, dtype=torch.uint8)
            silu_packed_out = torch.full_like(silu_packed, 0x7F)
            mxfp6.silu_and_mul_mxfp8_packed_out(
                silu_out,
                silu_packed_out,
                source,
            )
            torch.testing.assert_close(
                silu_out,
                silu_values.view(torch.uint8),
            )
            torch.testing.assert_close(silu_packed_out, silu_packed)

            expert_offsets = torch.tensor(
                [0, rows // 2, rows],
                device="cuda",
                dtype=torch.int64,
            )
            (
                grouped_values,
                grouped_scales,
                grouped_scale_offsets,
            ) = mxfp6.silu_and_mul_mxfp8_grouped(
                source,
                expert_offsets,
            )
            grouped_out = torch.empty_like(
                grouped_values,
                dtype=torch.uint8,
            )
            grouped_scales_out = torch.full_like(
                grouped_scales,
                0x7F,
            )
            mxfp6.silu_and_mul_mxfp8_grouped_out(
                grouped_out,
                grouped_scales_out,
                source,
                expert_offsets,
                grouped_scale_offsets,
            )
            torch.testing.assert_close(
                grouped_out,
                grouped_values.view(torch.uint8),
            )
            torch.testing.assert_close(
                grouped_scales_out,
                grouped_scales,
            )

        # Every group contains +28, so its exact UE8M0 scale is one. All E3M2
        # values then round-trip through the native 16->6 conversion.
        rows, k = 17, 256
        codes = torch.arange(32, dtype=torch.uint8, device="cuda").repeat(rows, k // 32)
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
    assert torch.ops.mxfp6.w6a8_config_abi(a8) == "native-w6a8-30-v5"
    for out_dtype in (torch.float16, torch.bfloat16):
        torch.ops.mxfp6.set_w6a8_config(a8, m, n, k, -1, 1, 0, out_dtype)
        reference = torch.ops.mxfp6.gemm_w6a8(
            a8, b.values, a6.scales, b.scales, m, n, k, 1.0, out_dtype
        )
        assert reference.dtype == out_dtype
        for config_id in range(30):
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
            torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.5)

    try:
        assert torch.ops.mxfp6.set_w6a8_config(a8, m, n, k, 5, 2, 1, torch.bfloat16)
        overridden = torch.ops.mxfp6.gemm_w6a8(
            a8, b.values, a6.scales, b.scales, m, n, k, 1.0, torch.bfloat16
        )
        expected = torch.ops.mxfp6.gemm_w6a8_config(
            a8, b.values, a6.scales, b.scales, m, n, k, 1.0, 5, 2, 1, torch.bfloat16
        )
        torch.testing.assert_close(overridden, expected, rtol=0, atol=0)
    finally:
        torch.ops.mxfp6.set_w6a8_config(a8, m, n, k, -1, 1, 0, torch.bfloat16)
    print("PASS FP16/BF16 W6A8 candidate ABI and C++ override registry")


def test_stream_k_autotune_filter(mxfp6) -> None:
    """Keep Stream-K out of candidate search when integration disables it."""
    autotune = importlib.import_module("mxfp6.autotune")
    kernel_ids = autotune._kernel_ids(512, 5120)
    stream_k_ids = autotune.STREAM_K_CONFIG_IDS
    if mxfp6.is_stream_k_enabled():
        assert stream_k_ids.intersection(kernel_ids)
    else:
        assert stream_k_ids.isdisjoint(kernel_ids)
    print("PASS Stream-K runtime-autotune candidate filtering")


def test_hybrid_autotune_candidate_policy(mxfp6) -> None:
    """Keep the validated all-family/safe-large-M boundary intentional."""
    autotune = importlib.import_module("mxfp6.autotune")
    with mock.patch.dict(
        os.environ,
        {"MXFP6_AUTOTUNE_ALL_FAMILIES_MAX_M": "512"},
        clear=False,
    ):
        expected_small = set(range(len(autotune.KERNEL_NAMES)))
        expected_large = set(range(6, 17))
        if not mxfp6.is_stream_k_enabled():
            expected_small.difference_update(autotune.STREAM_K_CONFIG_IDS)
            expected_large.difference_update(autotune.STREAM_K_CONFIG_IDS)
        assert set(autotune._kernel_ids(512, 5120)) == expected_small
        assert autotune._kernel_ids(513, 5120) == tuple(sorted(expected_large))
    print("PASS hybrid runtime-autotune candidate policy")


def test_autotune_graph_timer(mxfp6) -> None:
    """Exercise graph-internal timing with and without explicit L2 flush."""
    autotune = importlib.import_module("mxfp6.autotune")
    m, n, k = 17, 136, 128
    a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 6818)
    a6 = mxfp6.pack_operand(a_codes, sfa)
    b = mxfp6.pack_operand(b_codes, sfb)
    a8 = torch.ops.mxfp6.expand_fp6_to_fp8(a6.values, m, k)
    config = autotune.W6A8Config(5, 1, 0)
    allocated_before = torch.cuda.memory_allocated()
    warm_latency = autotune._measure_config(
        config,
        a8,
        b.values,
        a6.scales,
        b.scales,
        m,
        n,
        k,
        torch.bfloat16,
        2,
        5,
        2,
        None,
    )
    flush = torch.empty(250_000, dtype=torch.int32, device="cuda")
    cold_latency = autotune._measure_config(
        config,
        a8,
        b.values,
        a6.scales,
        b.scales,
        m,
        n,
        k,
        torch.bfloat16,
        2,
        5,
        2,
        flush,
    )
    del flush
    torch.cuda.synchronize()
    assert math.isfinite(warm_latency) and warm_latency > 0
    assert math.isfinite(cold_latency) and cold_latency > 0
    assert torch.cuda.memory_allocated() == allocated_before
    print("PASS CUDA Graph autotune timing and graph storage release")


def test_autotune_cache_only(mxfp6) -> None:
    """A deployment cache miss must not profile, while a hit installs."""
    autotune = importlib.import_module("mxfp6.autotune")
    ops = importlib.import_module("mxfp6.ops")
    m, n, k = 17, 136, 128
    a_codes, b_codes, sfa, sfb = make_problem(m, n, k, 6820)
    a6 = mxfp6.pack_operand(a_codes, sfa)
    b = mxfp6.pack_operand(b_codes, sfb)
    a8 = mxfp6.MXFP8Tensor(
        torch.ops.mxfp6.expand_fp6_to_fp8(a6.values, m, k),
        a6.scales,
        m,
        k,
    )
    out_dtype = torch.float16
    device_index = a8.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    descriptor = autotune._descriptor(device_index, m, n, k, out_dtype)
    cache_key = autotune._cache_key(descriptor)
    decision_key = (device_index, cache_key)
    ready_key = ops._autotune_process_key(a8.device, m, n, k, out_dtype)

    def clear_process_state() -> None:
        with autotune._STATE_LOCK:
            autotune._DECISIONS.pop(decision_key, None)
        with ops._AUTOTUNE_STATE_LOCK:
            ops._AUTOTUNE_READY.discard(ready_key)

    torch.ops.mxfp6.set_w6a8_config(
        a8.values, m, n, k, -1, 1, 0, out_dtype
    )
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "MXFP6_AUTOTUNE": "cache_only",
                    "MXFP6_AUTOTUNE_CACHE_DIR": cache_dir,
                },
                clear=False,
            ):
                assert mxfp6.is_autotune_enabled()
                assert mxfp6.is_autotune_cache_only()
                clear_process_state()
                with mock.patch.object(
                    autotune,
                    "_autotune",
                    side_effect=AssertionError("cache_only profiled a miss"),
                ) as profiler:
                    mxfp6.gemm_w6a8(a8, b, out_dtype=out_dtype)
                    mxfp6.gemm_w6a8(a8, b, out_dtype=out_dtype)
                    assert mxfp6.autotune_w6a8(
                        a8, b, out_dtype=out_dtype, force=True
                    ) is None
                    assert not profiler.called
                with ops._AUTOTUNE_STATE_LOCK:
                    assert ready_key in ops._AUTOTUNE_READY
                assert not list(Path(cache_dir).glob("*.json"))

                expected_config = autotune.W6A8Config(5, 2, 1)
                autotune._write_entry(
                    cache_key,
                    descriptor,
                    autotune.AutotuneResult(
                        expected_config, 1.0, 2.0, 1.5, 3
                    ),
                )
                clear_process_state()
                with mock.patch.object(
                    ops, "is_tuned_shape", return_value=True
                ), mock.patch.object(
                    autotune,
                    "_autotune",
                    side_effect=AssertionError("cache_only profiled a hit"),
                ) as profiler:
                    # Cache-only must also check a cache entry that overrides
                    # a shape otherwise covered by the built-in static table.
                    actual_config = mxfp6.autotune_w6a8(
                        a8, b, out_dtype=out_dtype
                    )
                    assert actual_config == expected_config
                    assert not profiler.called
                actual = mxfp6.gemm_w6a8(a8, b, out_dtype=out_dtype)
                expected = torch.ops.mxfp6.gemm_w6a8_config(
                    a8.values,
                    b.values,
                    a8.scales,
                    b.scales,
                    m,
                    n,
                    k,
                    1.0,
                    expected_config.config_id,
                    expected_config.swizzle,
                    expected_config.raster_order,
                    out_dtype,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        clear_process_state()
        torch.ops.mxfp6.set_w6a8_config(
            a8.values, m, n, k, -1, 1, 0, out_dtype
        )
    print("PASS cache-only autotune cache hit/miss behavior")


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
        source = torch.randn((m, k), generator=generator, device="cuda", dtype=dtype)
        _, b_codes, _, sfb = make_problem(m, n, k, 6900 + index)
        b = mxfp6.pack_operand(b_codes, sfb)
        quantized = mxfp6.quantize_activation(source)
        b_values = decode_e3m2(b_codes) * decode_ue8m0(sfb).repeat_interleave(32, dim=1)
        reference_fp32 = decode_mxfp8(quantized, mxfp6) @ b_values.t()
        for out_dtype in (torch.float16, torch.bfloat16):
            direct = mxfp6.gemm_w6a8(quantized, b, out_dtype=out_dtype)
            fused = mxfp6.gemm(source, b, out_dtype=out_dtype)
            reference = reference_fp32.to(out_dtype)
            assert direct.dtype == out_dtype
            torch.testing.assert_close(direct, reference, rtol=2e-3, atol=0.5)
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
        reference_bf16 = reference_gemm(a_codes, b_codes, sfa, sfb, torch.bfloat16)
        assert actual_bf16.dtype == torch.bfloat16
        torch.testing.assert_close(actual_bf16, reference_bf16, rtol=2e-3, atol=0.5)
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
    print(f"PASS packed-W6 large-M mixed GEMM {m}x{n}x{k}: max_abs={max_abs:g}")


def test_eager_prefill_static_heuristics(mxfp6) -> None:
    """Exercise the cache-miss large-prefill fallback windows."""
    shapes = (
        (1025, 5120, 3072),  # 64x128 wave-balance choice
        (1537, 7168, 5120),  # deep-K QKV short-tail Stream-K
        (1281, 8192, 5120),  # deep-K GDN-in short-tail Stream-K
    )
    with mock.patch.dict(os.environ, {"MXFP6_AUTOTUNE": "0"}, clear=False):
        for index, (m, n, k) in enumerate(shapes):
            assert not mxfp6.is_tuned_shape(m, n, k)
            a_codes, b_codes, sfa, sfb = make_problem(
                m, n, k, 1210 + index
            )
            a = mxfp6.pack_operand(a_codes, sfa)
            b = mxfp6.pack_operand(b_codes, sfb)
            actual = mxfp6.gemm(a, b)
            reference = reference_gemm(a_codes, b_codes, sfa, sfb)
            torch.testing.assert_close(actual, reference, rtol=2e-3, atol=0.25)
            max_abs = (actual - reference).abs().max().item()
            print(
                "PASS eager-prefill static heuristic "
                f"{m}x{n}x{k}: max_abs={max_abs:g}"
            )


def test_small_w6a8_dispatch(mxfp6) -> None:
    """Exercise each retained exact small-batch W6A8 kernel family."""
    shapes = (
        (1, 5120, 3072),    # 128x8, five-stage cooperative
        (1, 5120, 8704),    # 128x8 Stream-K
        (16, 8192, 5120),   # 64x16x128 stage-3 ping-pong
        (16, 5120, 3072),   # 64x16x128 stage-3 ping-pong
        (16, 7168, 5120),   # 64x16x128 stage-3 ping-pong
        (16, 17408, 5120),  # 128x16 cooperative
        (16, 5120, 8704),   # deep-K 64x16x128 stage-3 ping-pong
        (24, 8192, 5120),   # 128x32 cooperative
        (24, 5120, 3072),   # 64x32x128 ping-pong
        (24, 7168, 5120),   # 128x32 cooperative
        (24, 17408, 5120),  # 128x32 cooperative, N-major raster
        (24, 5120, 8704),   # deep-K 64x32x128 ping-pong
    )
    for index, (m, n, k) in enumerate(shapes):
        assert mxfp6.is_tuned_shape(m, n, k)
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
    """Exercise native and mixed exact dispatch around batch transitions."""
    shapes = (
        (32, 8192, 5120),
        (40, 5120, 3072),
        (48, 5120, 3072),
        (64, 5120, 3072),
        (64, 5120, 8704),
        (96, 7168, 5120),
        (96, 17408, 5120),
    )
    for index, (m, n, k) in enumerate(shapes):
        assert mxfp6.is_tuned_shape(m, n, k)
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
    torch.ops.mxfp6.set_w6a8_config(a8, m, n, k, 3, 1, 0, torch.bfloat16)
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

    if not mxfp6.is_stream_k_enabled():
        mxfp6.begin_workspace_planning()
        actual = torch.ops.mxfp6.gemm_w6a8(*arguments)
        split_k_actual = torch.ops.mxfp6.gemm(*split_k_arguments)
        torch.cuda.synchronize()
        planning_stats = mxfp6.workspace_stats()
        assert planning_stats["layouts"] == 0
        assert planning_stats["arena_bytes"] == 0
        finalized_stats = mxfp6.finalize_workspace_planning()
        assert finalized_stats["frozen"] == 1
        assert finalized_stats["lanes"] == 0
        expected = reference_gemm(a_codes, b_codes, sfa, sfb, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=0.5)
        torch.testing.assert_close(
            split_k_actual,
            reference_gemm(a_codes, b_codes, sfa, sfb),
            rtol=2e-3,
            atol=0.5,
        )
        print("PASS MXFP6_STREAM_K disables Stream-K and workspace layouts")
        return

    mxfp6.begin_workspace_planning()
    previous_collection = torch.ops.mxfp6._set_workspace_collection(a8, False)
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
    torch.testing.assert_close(split_k_eager, split_k_reference, rtol=0, atol=0)
    assert mxfp6.workspace_stats()["lanes"] == 2

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = torch.ops.mxfp6.gemm_w6a8(*arguments)
    stress_iterations = int(os.environ.get("MXFP6_PERSISTENT_STRESS_ITERATIONS", "100"))
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
    test_fp6_to_fp8(mxfp6)
    test_scale_tools(mxfp6)
    test_grouped_w6a8(mxfp6)
    test_grouped_w6a8_k64_transposed(mxfp6)
    test_array_w6a8(mxfp6)
    test_fused_array_silu_mxfp8(mxfp6)
    test_fused_array_w2_reduce(mxfp6)
    test_qwen35_b1_specialization(mxfp6)
    test_route_mxfp8(mxfp6)
    test_shared_group_and_reduce(mxfp6)
    test_dynamic_quantization(mxfp6)
    test_w6a8_candidate_registry(mxfp6)
    test_stream_k_autotune_filter(mxfp6)
    test_hybrid_autotune_candidate_policy(mxfp6)
    test_autotune_graph_timer(mxfp6)
    test_autotune_cache_only(mxfp6)
    test_float_w6a8_gemm(mxfp6)
    test_warmup_api(mxfp6)
    test_random_scale_gemm(mxfp6)
    test_small_w6a8_dispatch(mxfp6)
    test_tuned_transition_gemm(mxfp6)
    test_large_m_mixed_gemm(mxfp6)
    test_eager_prefill_static_heuristics(mxfp6)
    test_nondefault_stream(mxfp6)
    test_float_nondefault_stream(mxfp6)
    test_persistent_workspace(mxfp6)
    print("All MXFP6 tool tests passed")


if __name__ == "__main__":
    main()
