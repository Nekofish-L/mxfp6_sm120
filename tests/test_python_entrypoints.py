"""Compatibility checks for the split Python entry points (no CUDA required)."""

import importlib

import torch
import mxfp6
from mxfp6 import ops, quantization, moe


def test_existing_import_paths_share_implementations():
    for name in (
        "pack_fp6",
        "pack_scales",
        "quantize_mxfp8",
        "quantize_mxfp6",
        "quantize_mxfp8_dual",
        "quantize_mxfp8_dual_out",
        "MXFP8Tensor",
        "PackedMXFP6Tensor",
        "silu_and_mul_mxfp8",
    ):
        assert getattr(mxfp6, name) is getattr(ops, name) is getattr(quantization, name)
    for name in (
        "grouped_gemm_w6a8_out",
        "route_mxfp8_out",
        "moe_reduce_out",
        "qwen35_router_quant_out",
        "qwen35_topk_quant_route_out",
    ):
        assert getattr(mxfp6, name) is getattr(ops, name) is getattr(moe, name)
    w8 = importlib.import_module("mxfp6.mxfp8")
    assert w8.MXFP8Tensor is mxfp6.MXFP8Tensor
    assert (
        w8.quantize_activation
        is mxfp6.quantize_activation
        is quantization.quantize_mxfp8
    )


def test_float8_view_keeps_storage_and_legacy_name():
    values = torch.arange(128, dtype=torch.uint8)
    operand = mxfp6.MXFP8Tensor(values, torch.zeros(4, dtype=torch.uint8), 1, 128)
    view = operand.as_float8()
    assert view.shape == (1, 128)
    assert view.dtype == torch.float8_e4m3fn
    assert view.data_ptr() == values.data_ptr()
    assert torch.equal(
        view.view(torch.uint8), operand.dequantized_values().view(torch.uint8)
    )
