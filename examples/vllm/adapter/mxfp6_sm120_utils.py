# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the optional ``mxfp6-sm120`` W6A8 backend."""

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import TypeAlias

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_W6A8Problem = tuple[int, int, torch.Tensor, torch.Tensor]
_W6A8Models: TypeAlias = torch.nn.Module | Iterable[torch.nn.Module]

_REQUIRED_API = (
    "MXFP8Tensor",
    "PackedMXFP6Tensor",
    "autotune_w6a8",
    "begin_workspace_planning",
    "finalize_workspace_planning",
    "gemm_w6a8",
    "is_available",
    "load_library",
    "pack_scales",
    "workspace_stats",
)

_MOE_REQUIRED_API = (
    "Qwen35GroupedWorkspace",
    "array_gemm_w6a8_reduce_out",
    "grouped_gemm_w6a8_out",
    "moe_reduce_out",
    "qwen35_grouped_gemm_out",
    "qwen35_grouped_moe_out",
    "qwen35_grouped_reduce_out",
    "qwen35_grouped_workspace_shapes",
    "qwen35_router_quant_out",
    "qwen35_w1_splitk_silu_mxfp8_out",
    "qwen35_w2_splitk_reduce_out",
    "quantize_mxfp8_logical",
    "route_mxfp8_out",
    "silu_and_mul_mxfp8_grouped",
    "to_mma_k64_weight",
)


def _mxfp6_sm120_qwen35_grouped_moe_impl(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    mxfp6 = _import_mxfp6()
    workspace = mxfp6.Qwen35GroupedWorkspace.from_storage(
        output,
        workspace13,
        workspace2,
    )
    mxfp6.qwen35_grouped_moe_out(
        workspace,
        output,
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
    )


def _mxfp6_sm120_qwen35_grouped_moe_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    del output, hidden_states, w1, w1_scale, w2, w2_scale
    del topk_weights, topk_ids, workspace13, workspace2


direct_register_custom_op(
    op_name="mxfp6_sm120_qwen35_grouped_moe",
    op_func=_mxfp6_sm120_qwen35_grouped_moe_impl,
    mutates_args=["output", "workspace13", "workspace2"],
    fake_impl=_mxfp6_sm120_qwen35_grouped_moe_fake,
)


def mxfp6_sm120_qwen35_grouped_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    """Run the package-owned Qwen3.5 grouped routed schedule."""
    torch.ops.vllm.mxfp6_sm120_qwen35_grouped_moe(
        output,
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        workspace13,
        workspace2,
    )


def _import_mxfp6() -> ModuleType:
    return importlib.import_module("mxfp6")


def is_mxfp6_sm120_available() -> bool:
    """Return whether the native extension can run on the current device."""
    if not current_platform.is_cuda() or not current_platform.is_device_capability(120):
        return False

    try:
        mxfp6 = _import_mxfp6()
        if not all(hasattr(mxfp6, name) for name in _REQUIRED_API):
            logger.debug("mxfp6-sm120 does not provide the required W6A8 API")
            return False
        mxfp6.load_library()
        return True
    except Exception:
        logger.debug("mxfp6-sm120 is unavailable", exc_info=True)
        return False


def is_mxfp6_sm120_moe_available() -> bool:
    """Return whether the optional backend provides its native MoE API."""
    if not is_mxfp6_sm120_available():
        return False
    try:
        mxfp6 = _import_mxfp6()
        return all(hasattr(mxfp6, name) for name in _MOE_REQUIRED_API)
    except Exception:
        logger.debug("mxfp6-sm120 MoE API is unavailable", exc_info=True)
        return False


def pack_mxfp6_sm120_scales(logical_scales: torch.Tensor) -> torch.Tensor:
    """Convert logical UE8M0 scales to the CUTLASS SM120 layout."""
    return _import_mxfp6().pack_scales(logical_scales.contiguous())


def pack_mxfp6_sm120_k64_weight(weight: torch.Tensor) -> torch.Tensor:
    """Convert packed ``[E,N,64]`` weights to the compact SM120 TMA layout."""
    return _import_mxfp6().to_mma_k64_weight(weight)


def _mxfp6_sm120_quantize_mxfp8_impl(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _import_mxfp6().quantize_mxfp8_logical(input)


def _mxfp6_sm120_quantize_mxfp8_fake(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (input.shape[0], input.shape[1] // 32),
        dtype=torch.uint8,
        device=input.device,
    )
    return values, scales


direct_register_custom_op(
    op_name="mxfp6_sm120_quantize_mxfp8",
    op_func=_mxfp6_sm120_quantize_mxfp8_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_quantize_mxfp8_fake,
)


def mxfp6_sm120_quantize_mxfp8(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation with row-major per-32 UE8M0 scales."""
    return torch.ops.vllm.mxfp6_sm120_quantize_mxfp8(input)


def _mxfp6_sm120_route_impl(
    permuted_activation: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _import_mxfp6().route_mxfp8_out(
        activation,
        logical_scales,
        topk_ids,
        expert_map,
        local_experts,
        permuted_activation,
    )


def _mxfp6_sm120_route_fake(
    permuted_activation: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del permuted_activation, logical_scales, expert_map
    routes = activation.shape[0] * topk_ids.shape[1]
    scale_rows = routes + local_experts * 127
    packed_scales = torch.empty(
        scale_rows * (activation.shape[1] // 32),
        dtype=torch.uint8,
        device=activation.device,
    )
    expert_offsets = torch.empty(
        local_experts + 1,
        dtype=torch.int64,
        device=activation.device,
    )
    scale_offsets = torch.empty_like(expert_offsets)
    inverse_permutation = torch.empty(
        routes,
        dtype=torch.int32,
        device=activation.device,
    )
    return (
        packed_scales,
        expert_offsets,
        scale_offsets,
        inverse_permutation,
    )


direct_register_custom_op(
    op_name="mxfp6_sm120_route",
    op_func=_mxfp6_sm120_route_impl,
    mutates_args=["permuted_activation"],
    fake_impl=_mxfp6_sm120_route_fake,
)


def mxfp6_sm120_route(
    permuted_activation: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route MXFP8 rows and directly produce grouped CUTLASS scales."""
    return torch.ops.vllm.mxfp6_sm120_route(
        permuted_activation,
        activation,
        logical_scales,
        topk_ids,
        expert_map,
        local_experts,
    )


def _mxfp6_sm120_quantize_mxfp8_packed_impl(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    operand = _import_mxfp6().quantize_mxfp8(input)
    return operand.values.view(torch.float8_e4m3fn), operand.scales


def _mxfp6_sm120_quantize_mxfp8_packed_fake(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    padded_rows = (input.shape[0] + 127) // 128 * 128
    scales = torch.empty(
        padded_rows * (input.shape[1] // 32),
        dtype=torch.uint8,
        device=input.device,
    )
    return values, scales


direct_register_custom_op(
    op_name="mxfp6_sm120_quantize_mxfp8_packed",
    op_func=_mxfp6_sm120_quantize_mxfp8_packed_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_quantize_mxfp8_packed_fake,
)


def mxfp6_sm120_quantize_mxfp8_packed(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation directly to the SM120 CUTLASS scale layout."""
    return torch.ops.vllm.mxfp6_sm120_quantize_mxfp8_packed(input)


def _mxfp6_sm120_grouped_gemm_impl(
    output: torch.Tensor,
    activation: torch.Tensor,
    packed_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    mxfp6 = _import_mxfp6()
    mxfp6.grouped_gemm_w6a8_out(
        output,
        activation,
        packed_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_grouped_gemm_fake(
    output: torch.Tensor,
    activation: torch.Tensor,
    packed_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    del output, activation, packed_scales, weight, weight_scales
    del expert_offsets, scale_offsets


direct_register_custom_op(
    op_name="mxfp6_sm120_grouped_gemm",
    op_func=_mxfp6_sm120_grouped_gemm_impl,
    mutates_args=["output"],
    fake_impl=_mxfp6_sm120_grouped_gemm_fake,
)


def mxfp6_sm120_grouped_gemm(
    output: torch.Tensor,
    activation: torch.Tensor,
    packed_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    """Run a native grouped W6A8 GEMM with routed packed scales."""
    torch.ops.vllm.mxfp6_sm120_grouped_gemm(
        output,
        activation,
        packed_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_silu_grouped_gemm_impl(
    output: torch.Tensor,
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    mxfp6 = _import_mxfp6()
    activation, scales, scale_offsets = mxfp6.silu_and_mul_mxfp8_grouped(
        gate_up, expert_offsets, scale_offsets
    )
    mxfp6.grouped_gemm_w6a8_out(
        output,
        activation,
        scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_silu_grouped_gemm_fake(
    output: torch.Tensor,
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    del output, gate_up, weight, weight_scales
    del expert_offsets, scale_offsets


direct_register_custom_op(
    op_name="mxfp6_sm120_silu_grouped_gemm",
    op_func=_mxfp6_sm120_silu_grouped_gemm_impl,
    mutates_args=["output"],
    fake_impl=_mxfp6_sm120_silu_grouped_gemm_fake,
)


def mxfp6_sm120_silu_grouped_gemm(
    output: torch.Tensor,
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    """Fuse SiLU-and-mul quantization with the second grouped W6A8 GEMM."""
    torch.ops.vllm.mxfp6_sm120_silu_grouped_gemm(
        output,
        gate_up,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_reduce_impl(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> None:
    _import_mxfp6().moe_reduce_out(
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
    )


def _mxfp6_sm120_reduce_fake(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> None:
    del output, routed_output, topk_weights, inverse_permutation


direct_register_custom_op(
    op_name="mxfp6_sm120_reduce",
    op_func=_mxfp6_sm120_reduce_impl,
    mutates_args=["output"],
    fake_impl=_mxfp6_sm120_reduce_fake,
)


def mxfp6_sm120_reduce(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> None:
    """Reduce routed rows using only the independent mxfp6-sm120 package."""
    torch.ops.vllm.mxfp6_sm120_reduce(
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
    )
