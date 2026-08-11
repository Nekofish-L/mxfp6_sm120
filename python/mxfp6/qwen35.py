"""Allocation-free Qwen3.5 MXFP6 MoE schedules for SM120."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ops import (
    grouped_gemm_w6a8_out,
    moe_reduce_out,
    quantize_mxfp8_logical,
    qwen35_grouped_w1_silu_mxfp8_out,
    qwen35_route_mxfp8_out,
    qwen35_router_quant_out,
    qwen35_topk_quant_route_out,
    qwen35_w1_splitk_silu_mxfp8_out,
    qwen35_w2_splitk_reduce_out,
    route_mxfp8_out,
    silu_and_mul_mxfp8_grouped,
)

_HIDDEN_SIZE = 2048
_INTERMEDIATE_SIZE = 256
_NUM_EXPERTS = 256
_TOPK = 8
_SCALE_ROWS_PER_ATOM = 128


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _grouped_workspace_bytes(batch_size: int) -> tuple[int, int]:
    routes = batch_size * _TOPK
    active_experts = min(routes, _NUM_EXPERTS)
    padded_rows = routes + active_experts * (_SCALE_ROWS_PER_ATOM - 1)

    first_bytes = batch_size * _HIDDEN_SIZE * 2
    first_bytes = _align_up(first_bytes, 16) + batch_size * 64
    first_bytes = _align_up(first_bytes, 4) + batch_size * _TOPK * 4
    first_bytes = _align_up(first_bytes, 4) + batch_size * _TOPK * 4
    first_bytes = _align_up(first_bytes, 4) + routes * 4
    first_bytes = _align_up(first_bytes, 16) + routes * _INTERMEDIATE_SIZE
    first_bytes = _align_up(first_bytes, 16) + padded_rows * 8
    first_bytes = _align_up(first_bytes, 4) + _NUM_EXPERTS * 4
    first_bytes = _align_up(first_bytes, 8) + (_NUM_EXPERTS + 1) * 8
    first_bytes = _align_up(first_bytes, 8) + (_NUM_EXPERTS + 1) * 8
    first_bytes = _align_up(first_bytes, 4) + routes * 4

    second_bytes = routes * _HIDDEN_SIZE
    second_bytes = _align_up(second_bytes, 16) + padded_rows * 64
    second_bytes = _align_up(second_bytes, 16) + routes * _HIDDEN_SIZE * 2
    return first_bytes, second_bytes


def qwen35_grouped_workspace_shapes(
    batch_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return BF16 workspace shapes for the grouped Qwen3.5 schedule."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    routes = batch_size * _TOPK
    first_bytes, second_bytes = _grouped_workspace_bytes(batch_size)
    first_width = (_align_up(first_bytes, 2) // 2 + routes - 1) // routes
    second_width = (_align_up(second_bytes, 2) // 2 + routes - 1) // routes
    return (routes, first_width), (routes, second_width)


class _ByteArena:
    def __init__(self, storage: torch.Tensor, offset: int = 0) -> None:
        if not storage.is_contiguous():
            raise ValueError("workspace storage must be contiguous")
        self.storage = storage.view(torch.uint8).flatten()
        self.offset = offset

    def take(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        alignment: int,
    ) -> torch.Tensor:
        elements = 1
        for dimension in shape:
            elements *= dimension
        item_size = torch.empty((), dtype=dtype).element_size()
        size = elements * item_size
        self.offset = _align_up(self.offset, alignment)
        end = self.offset + size
        if end > self.storage.numel():
            raise ValueError("Qwen3.5 grouped workspace is too small")
        result = self.storage[self.offset : end].view(dtype).view(shape)
        self.offset = end
        return result


@dataclass(frozen=True)
class Qwen35GroupedWorkspace:
    """Views over graph-stable scratch storage for the grouped schedule."""

    quantized: torch.Tensor
    input_scales: torch.Tensor
    packed_input_scales: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    source_tokens: torch.Tensor
    permuted: torch.Tensor
    grouped_input_scales: torch.Tensor
    activated: torch.Tensor
    activated_scales: torch.Tensor
    expert_cursors: torch.Tensor
    expert_offsets: torch.Tensor
    scale_offsets: torch.Tensor
    inverse_permutation: torch.Tensor
    routed_output: torch.Tensor

    @classmethod
    def from_storage(
        cls,
        output: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> "Qwen35GroupedWorkspace":
        """Partition two BF16 scratch tensors without allocating CUDA memory."""
        if (
            output.dtype != torch.bfloat16
            or output.ndim != 2
            or output.shape[1] != _HIDDEN_SIZE
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous BF16 [B,2048]")
        batch_size = output.shape[0]
        routes = batch_size * _TOPK
        active_experts = min(routes, _NUM_EXPERTS)
        padded_rows = routes + active_experts * (_SCALE_ROWS_PER_ATOM - 1)
        required_first, required_second = _grouped_workspace_bytes(batch_size)
        if first.numel() * first.element_size() < required_first:
            raise ValueError("first Qwen3.5 grouped workspace is too small")
        if second.numel() * second.element_size() < required_second:
            raise ValueError("second Qwen3.5 grouped workspace is too small")

        first_arena = _ByteArena(first, output.numel() * output.element_size())
        input_scales = first_arena.take((batch_size, 64), torch.uint8, 16)
        topk_weights = first_arena.take(
            (batch_size, _TOPK), torch.float32, 4
        )
        topk_ids = first_arena.take((batch_size, _TOPK), torch.int32, 4)
        source_tokens = first_arena.take((routes,), torch.int32, 4)
        activated = first_arena.take(
            (routes, _INTERMEDIATE_SIZE), torch.uint8, 16
        )
        activated_scales = first_arena.take((padded_rows * 8,), torch.uint8, 16)
        expert_cursors = first_arena.take((_NUM_EXPERTS,), torch.int32, 4)
        expert_offsets = first_arena.take((_NUM_EXPERTS + 1,), torch.int64, 8)
        scale_offsets = first_arena.take((_NUM_EXPERTS + 1,), torch.int64, 8)
        inverse = first_arena.take((routes,), torch.int32, 4)

        second_arena = _ByteArena(second)
        permuted = second_arena.take(
            (routes, _HIDDEN_SIZE), torch.float8_e4m3fn, 16
        )
        grouped_input_scales = second_arena.take(
            (padded_rows * 64,), torch.uint8, 16
        )
        routed_output = second_arena.take(
            (routes, _HIDDEN_SIZE), torch.bfloat16, 16
        )
        quantized = permuted[:batch_size].view(torch.uint8)
        packed_rows = (batch_size + _SCALE_ROWS_PER_ATOM - 1) // (
            _SCALE_ROWS_PER_ATOM
        ) * _SCALE_ROWS_PER_ATOM
        packed_input_scales = grouped_input_scales[: packed_rows * 64]
        return cls(
            quantized=quantized,
            input_scales=input_scales,
            packed_input_scales=packed_input_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            source_tokens=source_tokens,
            permuted=permuted,
            grouped_input_scales=grouped_input_scales,
            activated=activated,
            activated_scales=activated_scales,
            expert_cursors=expert_cursors,
            expert_offsets=expert_offsets,
            scale_offsets=scale_offsets,
            inverse_permutation=inverse,
            routed_output=routed_output,
        )


def qwen35_grouped_gemm_out(
    workspace: Qwen35GroupedWorkspace,
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    renormalize: bool = True,
) -> None:
    """Queue the fused router and indirect routed GEMMs into ``workspace``."""
    qwen35_topk_quant_route_out(
        workspace.quantized,
        workspace.input_scales,
        workspace.packed_input_scales,
        workspace.topk_weights,
        workspace.topk_ids,
        workspace.permuted,
        workspace.grouped_input_scales,
        workspace.expert_cursors,
        workspace.expert_offsets,
        workspace.scale_offsets,
        workspace.inverse_permutation,
        hidden,
        router_logits,
        source_tokens=workspace.source_tokens,
        renormalize=renormalize,
    )
    qwen35_grouped_w1_silu_mxfp8_out(
        workspace.activated,
        workspace.activated_scales,
        workspace.quantized,
        workspace.packed_input_scales,
        w1,
        w1_scales,
        workspace.expert_offsets,
        workspace.scale_offsets,
        use_pdl=True,
        source_tokens=workspace.source_tokens,
    )
    grouped_gemm_w6a8_out(
        workspace.routed_output,
        workspace.activated,
        workspace.activated_scales,
        w2,
        w2_scales,
        workspace.expert_offsets,
        workspace.scale_offsets,
        tile_n=-11,
        use_pdl=True,
    )


def qwen35_grouped_reduce_out(
    workspace: Qwen35GroupedWorkspace,
    output: torch.Tensor,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce routed rows and optionally add an already-gated shared expert."""
    moe_reduce_out(
        output,
        workspace.routed_output,
        workspace.topk_weights,
        workspace.inverse_permutation,
        shared_output=shared_output,
        use_pdl=True,
    )
    return output


def qwen35_grouped_moe_logits_out(
    workspace: Qwen35GroupedWorkspace,
    output: torch.Tensor,
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    shared_output: torch.Tensor | None = None,
    renormalize: bool = True,
) -> torch.Tensor:
    """Run the allocation-free fused-router Qwen3.5 grouped schedule."""
    qwen35_grouped_gemm_out(
        workspace,
        hidden,
        router_logits,
        w1,
        w1_scales,
        w2,
        w2_scales,
        renormalize=renormalize,
    )
    return qwen35_grouped_reduce_out(workspace, output, shared_output)


def qwen35_grouped_moe_out(
    workspace: Qwen35GroupedWorkspace,
    output: torch.Tensor,
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the allocation-free routed Qwen3.5 B20--B96 schedule."""
    quantized, logical_scales = quantize_mxfp8_logical(hidden)
    (
        grouped_input_scales,
        expert_offsets,
        scale_offsets,
        inverse_permutation,
    ) = route_mxfp8_out(
        quantized,
        logical_scales,
        topk_ids,
        None,
        _NUM_EXPERTS,
        workspace.permuted,
    )
    qwen35_grouped_w1_silu_mxfp8_out(
        workspace.activated,
        workspace.activated_scales,
        workspace.permuted,
        grouped_input_scales,
        w1,
        w1_scales,
        expert_offsets,
        scale_offsets,
        use_pdl=False,
    )
    grouped_gemm_w6a8_out(
        workspace.routed_output,
        workspace.activated,
        workspace.activated_scales,
        w2,
        w2_scales,
        expert_offsets,
        scale_offsets,
        tile_n=-11,
        use_pdl=False,
    )
    moe_reduce_out(
        output,
        workspace.routed_output,
        topk_weights,
        inverse_permutation,
        use_pdl=False,
    )
    return output


@dataclass(frozen=True)
class Qwen35MoeB1Workspace:
    """Persistent buffers for the Qwen3.5-35B-A3B TP2 decode specialization."""

    quantized: torch.Tensor
    input_scales: torch.Tensor
    routed_logits: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    shared_gate: torch.Tensor
    w1_partial: torch.Tensor
    activation: torch.Tensor
    activation_scales: torch.Tensor
    w2_partial: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(
        cls,
        device: torch.device | str | int,
    ) -> "Qwen35MoeB1Workspace":
        """Allocate graph-stable workspace tensors on ``device``."""
        device = torch.device(device)
        return cls(
            quantized=torch.empty((1, 2048), device=device, dtype=torch.uint8),
            input_scales=torch.empty((1, 64), device=device, dtype=torch.uint8),
            routed_logits=torch.empty((1, 256), device=device, dtype=torch.bfloat16),
            topk_weights=torch.empty((1, 8), device=device, dtype=torch.float32),
            topk_ids=torch.empty((1, 8), device=device, dtype=torch.int32),
            shared_gate=torch.empty((1,), device=device, dtype=torch.bfloat16),
            w1_partial=torch.empty((288, 64), device=device, dtype=torch.float32),
            activation=torch.empty((9, 256), device=device, dtype=torch.uint8),
            activation_scales=torch.empty((9, 8), device=device, dtype=torch.uint8),
            w2_partial=torch.empty((256, 9, 16), device=device, dtype=torch.float32),
            output=torch.empty((1, 2048), device=device, dtype=torch.bfloat16),
        )


def qwen35_moe_b1_out(
    workspace: Qwen35MoeB1Workspace,
    hidden: torch.Tensor,
    combined_gate: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    renormalize: bool = False,
) -> torch.Tensor:
    """Run the three-kernel Qwen3.5 TP2 B=1 schedule into ``workspace``.

    ``combined_gate`` contains the 256 routed gate rows followed by the
    shared-expert gate. ``w1`` and ``w2`` contain the 256 routed experts
    followed by the shared expert in canonical packed-FP6 layout.
    """
    qwen35_router_quant_out(
        workspace.quantized,
        workspace.input_scales,
        workspace.routed_logits,
        workspace.topk_weights,
        workspace.topk_ids,
        workspace.shared_gate,
        hidden,
        combined_gate,
        renormalize=renormalize,
    )
    qwen35_w1_splitk_silu_mxfp8_out(
        workspace.activation,
        workspace.activation_scales,
        workspace.w1_partial,
        workspace.quantized,
        workspace.input_scales,
        w1,
        w1_scales,
        workspace.topk_ids,
    )
    qwen35_w2_splitk_reduce_out(
        workspace.output,
        workspace.w2_partial,
        workspace.activation,
        workspace.activation_scales,
        w2,
        w2_scales,
        workspace.topk_ids,
        workspace.topk_weights,
        workspace.shared_gate,
    )
    return workspace.output
