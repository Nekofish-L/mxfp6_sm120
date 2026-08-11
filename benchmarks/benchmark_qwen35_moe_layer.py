#!/usr/bin/env python3
"""Benchmark one Qwen3.5 MoE layer with real TP-sharded checkpoint weights.

The FP8 baseline intentionally follows vLLM's runtime choices on SM120:

* the routed experts use vLLM's Triton FP8-block MoE;
* the shared expert uses block-scaled CUTLASS GEMMs;
* the shared path overlaps the router/routed path on an auxiliary stream;
* SiLU-and-mul and the second shared activation quantization are fused.

The MXFP6 modes cover merged grouped GEMMs, route-array scheduling, and
auxiliary-stream shared-expert overlap.  The automatic large-batch schedule
uses route-array experts below batch 32 and expert-sorted grouped GEMMs from
batch 32 while overlapping the dense shared expert.  Both paths are captured
as complete CUDA Graphs.  With two torchrun workers, the final TP all-reduce
is captured too.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

import mxfp6


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
NUM_EXPERTS = 256
TOPK = 8
FP8_BLOCK = 128
MX_BLOCK = 32
LAYER_PREFIX = "model.language_model.layers.0.mlp"


@dataclass(frozen=True)
class Fp8LayerWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    shared_w1: torch.Tensor
    shared_w2: torch.Tensor
    shared_w1_scale: torch.Tensor
    shared_w2_scale: torch.Tensor
    router_gate: torch.Tensor
    shared_gate: torch.Tensor


@dataclass(frozen=True)
class MxLayerWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_padded: torch.Tensor
    w2_padded: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    combined_gate: torch.Tensor


@dataclass(frozen=True)
class CapturedLayer:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor


class RuntimeAllReduce:
    """vLLM's CUDA custom all-reduce with its graph-registration protocol."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.communicator = CustomAllreduce(group=group, device=device)
        if self.communicator.disabled:
            raise RuntimeError("vLLM custom all-reduce is disabled on this topology")

    @contextmanager
    def capture(self):
        with self.communicator.capture():
            yield

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        output = self.communicator.custom_all_reduce(value)
        if output is None:
            raise RuntimeError("vLLM custom all-reduce rejected the layer output")
        return output

    def fused_moe_reduce(
        self,
        output: torch.Tensor,
        routed_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_permutation: torch.Tensor,
        shared_output: torch.Tensor,
        shared_gate: torch.Tensor,
    ) -> torch.Tensor:
        mxfp6.moe_reduce_tp2_out(
            output,
            routed_output,
            topk_weights,
            inverse_permutation,
            shared_output,
            shared_gate,
            self.communicator.meta_ptrs,
            self.communicator.buffer_ptrs,
            self.communicator.rank,
        )
        return output

    def close(self) -> None:
        self.communicator.close()


def _tensor(
    handle,
    name: str,
    row_slice: slice | None = None,
    column_slice: slice | None = None,
) -> torch.Tensor:
    value = handle.get_tensor(name)
    if row_slice is not None:
        value = value[row_slice]
    if column_slice is not None:
        value = value[:, column_slice]
    return value.contiguous()


def _expert_name(expert: int, projection: str, suffix: str) -> str:
    return f"{LAYER_PREFIX}.experts.{expert}.{projection}.{suffix}"


def _shared_name(projection: str, suffix: str) -> str:
    return f"{LAYER_PREFIX}.shared_expert.{projection}.{suffix}"


def _to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, non_blocking=False)


def load_fp8_layer(
    model_path: Path,
    tp_rank: int,
    tp_size: int,
    device: torch.device,
) -> Fp8LayerWeights:
    if INTERMEDIATE_SIZE % tp_size:
        raise ValueError("intermediate size must be divisible by TP size")
    local_i = INTERMEDIATE_SIZE // tp_size
    row_slice = slice(tp_rank * local_i, (tp_rank + 1) * local_i)
    scale_rows = local_i // FP8_BLOCK
    scale_row_slice = slice(
        tp_rank * scale_rows,
        (tp_rank + 1) * scale_rows,
    )
    packed_columns = local_i
    column_slice = slice(
        tp_rank * packed_columns,
        (tp_rank + 1) * packed_columns,
    )
    scale_columns = local_i // FP8_BLOCK
    scale_column_slice = slice(
        tp_rank * scale_columns,
        (tp_rank + 1) * scale_columns,
    )

    file_w1 = model_path / "model.safetensors-00006-of-00014.safetensors"
    file_w2 = model_path / "model.safetensors-00012-of-00014.safetensors"
    file_shared = model_path / "model.safetensors-00014-of-00014.safetensors"
    with (
        safe_open(file_w1, framework="pt", device="cpu") as w1_file,
        safe_open(file_w2, framework="pt", device="cpu") as w2_file,
        safe_open(file_shared, framework="pt", device="cpu") as shared_file,
    ):
        gate = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(expert, "gate_proj", "weight"),
                    row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        up = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(expert, "up_proj", "weight"),
                    row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        w1 = _to_device(torch.cat((gate, up), dim=1), device)
        del gate, up

        gate_scale = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(
                        expert,
                        "gate_proj",
                        "weight_scale_inv",
                    ),
                    scale_row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        up_scale = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(
                        expert,
                        "up_proj",
                        "weight_scale_inv",
                    ),
                    scale_row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        w1_scale = _to_device(
            torch.cat((gate_scale, up_scale), dim=1).float(),
            device,
        )
        del gate_scale, up_scale

        w2 = _to_device(
            torch.stack(
                [
                    _tensor(
                        w2_file,
                        _expert_name(expert, "down_proj", "weight"),
                        column_slice=column_slice,
                    )
                    for expert in range(NUM_EXPERTS)
                ]
            ),
            device,
        )
        w2_scale = _to_device(
            torch.stack(
                [
                    _tensor(
                        w2_file,
                        _expert_name(
                            expert,
                            "down_proj",
                            "weight_scale_inv",
                        ),
                        column_slice=scale_column_slice,
                    )
                    for expert in range(NUM_EXPERTS)
                ]
            ).float(),
            device,
        )

        shared_gate_weight = _tensor(
            shared_file,
            _shared_name("gate_proj", "weight"),
            row_slice,
        )
        shared_up_weight = _tensor(
            shared_file,
            _shared_name("up_proj", "weight"),
            row_slice,
        )
        shared_w1 = _to_device(
            torch.cat((shared_gate_weight, shared_up_weight), dim=0),
            device,
        )
        shared_gate_scale = _tensor(
            shared_file,
            _shared_name("gate_proj", "weight_scale_inv"),
            scale_row_slice,
        )
        shared_up_scale = _tensor(
            shared_file,
            _shared_name("up_proj", "weight_scale_inv"),
            scale_row_slice,
        )
        shared_w1_scale = _to_device(
            torch.cat((shared_gate_scale, shared_up_scale), dim=0).float(),
            device,
        )
        shared_w2 = _to_device(
            _tensor(
                shared_file,
                _shared_name("down_proj", "weight"),
                column_slice=column_slice,
            ),
            device,
        )
        shared_w2_scale = _to_device(
            _tensor(
                shared_file,
                _shared_name("down_proj", "weight_scale_inv"),
                column_slice=scale_column_slice,
            ).float(),
            device,
        )
        router_gate = _to_device(
            _tensor(shared_file, f"{LAYER_PREFIX}.gate.weight"),
            device,
        )
        shared_gate = _to_device(
            _tensor(
                shared_file,
                f"{LAYER_PREFIX}.shared_expert_gate.weight",
            ),
            device,
        )

    gc.collect()
    return Fp8LayerWeights(
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_w1_scale=shared_w1_scale,
        shared_w2_scale=shared_w2_scale,
        router_gate=router_gate,
        shared_gate=shared_gate,
    )


def _pack_expert_scales(logical: torch.Tensor) -> torch.Tensor:
    experts = logical.shape[0]
    packed = mxfp6.pack_scales(logical.flatten(0, 1).contiguous())
    return packed.view(experts, -1)


def load_mx_layer(
    model_path: Path,
    tp_rank: int,
    tp_size: int,
    device: torch.device,
) -> MxLayerWeights:
    if INTERMEDIATE_SIZE % tp_size:
        raise ValueError("intermediate size must be divisible by TP size")
    local_i = INTERMEDIATE_SIZE // tp_size
    row_slice = slice(tp_rank * local_i, (tp_rank + 1) * local_i)
    scale_row_slice = row_slice
    packed_columns = local_i * 3 // 4
    column_slice = slice(
        tp_rank * packed_columns,
        (tp_rank + 1) * packed_columns,
    )
    scale_columns = local_i // MX_BLOCK
    scale_column_slice = slice(
        tp_rank * scale_columns,
        (tp_rank + 1) * scale_columns,
    )

    file_w1 = model_path / "model.safetensors-00006-of-00014.safetensors"
    file_w2 = model_path / "model.safetensors-00012-of-00014.safetensors"
    file_shared = model_path / "model.safetensors-00014-of-00014.safetensors"
    with (
        safe_open(file_w1, framework="pt", device="cpu") as w1_file,
        safe_open(file_w2, framework="pt", device="cpu") as w2_file,
        safe_open(file_shared, framework="pt", device="cpu") as shared_file,
    ):
        gate = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(expert, "gate_proj", "weight"),
                    row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        up = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(expert, "up_proj", "weight"),
                    row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        routed_w1 = torch.cat((gate, up), dim=1)
        del gate, up

        gate_scale = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(
                        expert,
                        "gate_proj",
                        "weight_scale",
                    ),
                    scale_row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        up_scale = torch.stack(
            [
                _tensor(
                    w1_file,
                    _expert_name(expert, "up_proj", "weight_scale"),
                    scale_row_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        routed_w1_scale = torch.cat((gate_scale, up_scale), dim=1)
        del gate_scale, up_scale

        routed_w2 = torch.stack(
            [
                _tensor(
                    w2_file,
                    _expert_name(expert, "down_proj", "weight"),
                    column_slice=column_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )
        routed_w2_scale = torch.stack(
            [
                _tensor(
                    w2_file,
                    _expert_name(expert, "down_proj", "weight_scale"),
                    column_slice=scale_column_slice,
                )
                for expert in range(NUM_EXPERTS)
            ]
        )

        shared_w1 = torch.cat(
            (
                _tensor(
                    shared_file,
                    _shared_name("gate_proj", "weight"),
                    row_slice,
                ),
                _tensor(
                    shared_file,
                    _shared_name("up_proj", "weight"),
                    row_slice,
                ),
            ),
            dim=0,
        ).unsqueeze(0)
        shared_w1_scale = torch.cat(
            (
                _tensor(
                    shared_file,
                    _shared_name("gate_proj", "weight_scale"),
                    scale_row_slice,
                ),
                _tensor(
                    shared_file,
                    _shared_name("up_proj", "weight_scale"),
                    scale_row_slice,
                ),
            ),
            dim=0,
        ).unsqueeze(0)
        shared_w2 = _tensor(
            shared_file,
            _shared_name("down_proj", "weight"),
            column_slice=column_slice,
        ).unsqueeze(0)
        shared_w2_scale = _tensor(
            shared_file,
            _shared_name("down_proj", "weight_scale"),
            column_slice=scale_column_slice,
        ).unsqueeze(0)

        w1 = _to_device(
            torch.cat((routed_w1, shared_w1), dim=0),
            device,
        )
        w2 = _to_device(
            torch.cat((routed_w2, shared_w2), dim=0),
            device,
        )
        logical_w1_scale = _to_device(
            torch.cat((routed_w1_scale, shared_w1_scale), dim=0),
            device,
        )
        logical_w2_scale = _to_device(
            torch.cat((routed_w2_scale, shared_w2_scale), dim=0),
            device,
        )
        router_gate = _tensor(
            shared_file,
            f"{LAYER_PREFIX}.gate.weight",
        )
        shared_gate = _tensor(
            shared_file,
            f"{LAYER_PREFIX}.shared_expert_gate.weight",
        )
        combined_gate = _to_device(
            torch.cat((router_gate, shared_gate), dim=0),
            device,
        )

    w1_scale = _pack_expert_scales(logical_w1_scale)
    w2_scale = _pack_expert_scales(logical_w2_scale)
    w1_padded = mxfp6.pad_fp6(w1)
    w2_padded = mxfp6.pad_fp6(w2)
    del logical_w1_scale, logical_w2_scale
    gc.collect()
    return MxLayerWeights(
        w1=w1,
        w2=w2,
        w1_padded=w1_padded,
        w2_padded=w2_padded,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        combined_gate=combined_gate,
    )


def _maybe_all_reduce(
    output: torch.Tensor,
    all_reduce: RuntimeAllReduce | None,
) -> torch.Tensor:
    if all_reduce is None:
        return output
    return all_reduce(output)


class Fp8Layer:
    def __init__(
        self,
        weights: Fp8LayerWeights,
        quant_config,
        all_reduce: RuntimeAllReduce | None,
    ) -> None:
        self.weights = weights
        self.quant_config = quant_config
        self.all_reduce = all_reduce
        self.shared_stream = torch.cuda.Stream()

    def _shared(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        weights = self.weights
        quantized, scales = per_token_group_quant_fp8(
            hidden_states,
            group_size=FP8_BLOCK,
            column_major_scales=True,
            dtype=torch.float8_e4m3fn,
            use_ue8m0=False,
        )
        gate_up = ops.cutlass_scaled_mm(
            quantized,
            weights.shared_w1.T,
            out_dtype=torch.bfloat16,
            scale_a=scales,
            scale_b=weights.shared_w1_scale.T,
        )
        activated, activated_scales = ops.silu_and_mul_per_block_quant(
            gate_up,
            FP8_BLOCK,
            torch.float8_e4m3fn,
            is_scale_transposed=True,
        )
        output = ops.cutlass_scaled_mm(
            activated,
            weights.shared_w2.T,
            out_dtype=torch.bfloat16,
            scale_a=activated_scales,
            scale_b=weights.shared_w2_scale.T,
        )
        gate = torch.sigmoid(F.linear(hidden_states, weights.shared_gate))
        return output * gate

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts,
        )
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            fused_topk,
        )

        current = torch.cuda.current_stream()
        self.shared_stream.wait_stream(current)
        with torch.cuda.stream(self.shared_stream):
            shared_output = self._shared(hidden_states)

        router_logits = F.linear(
            hidden_states,
            self.weights.router_gate,
        )
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            router_logits,
            TOPK,
            renormalize=False,
        )
        routed_output = fused_experts(
            hidden_states,
            self.weights.w1,
            self.weights.w2,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            quant_config=self.quant_config,
        )
        current.wait_stream(self.shared_stream)
        output = routed_output + shared_output
        return _maybe_all_reduce(output, self.all_reduce)


class MxLayer:
    def __init__(
        self,
        weights: MxLayerWeights,
        batch_size: int,
        all_reduce: RuntimeAllReduce | None,
        tile_n: int,
    ) -> None:
        self.weights = weights
        self.all_reduce = all_reduce
        self.tile_n = tile_n
        rows = batch_size * (TOPK + 1)
        device = weights.w1.device
        self.permuted = torch.empty(
            (rows, HIDDEN_SIZE),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self.gemm1 = torch.empty(
            (rows, weights.w1.shape[1]),
            device=device,
            dtype=torch.bfloat16,
        )
        self.gemm2 = torch.empty(
            (rows, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
        )
        self.output = torch.empty(
            (batch_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            fused_topk,
        )

        batch_size = hidden_states.shape[0]
        routes = batch_size * TOPK
        gate_logits = F.linear(
            hidden_states,
            self.weights.combined_gate,
        )
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            gate_logits[:, :NUM_EXPERTS],
            TOPK,
            renormalize=False,
        )
        quantized, logical_scales = mxfp6.quantize_mxfp8_logical(hidden_states)
        (
            packed_scales,
            expert_offsets,
            scale_offsets,
            inverse_permutation,
        ) = mxfp6.route_mxfp8_out(
            quantized,
            logical_scales,
            topk_ids,
            None,
            NUM_EXPERTS + 1,
            self.permuted,
            include_shared=True,
        )
        mxfp6.grouped_gemm_w6a8_out(
            self.gemm1,
            self.permuted,
            packed_scales,
            self.weights.w1,
            self.weights.w1_scale,
            expert_offsets,
            scale_offsets,
            tile_n=self.tile_n,
        )
        activated, activated_scales, activated_scale_offsets = (
            mxfp6.silu_and_mul_mxfp8_grouped(
                self.gemm1,
                expert_offsets,
                scale_offsets,
            )
        )
        mxfp6.grouped_gemm_w6a8_out(
            self.gemm2,
            activated,
            activated_scales,
            self.weights.w2,
            self.weights.w2_scale,
            expert_offsets,
            activated_scale_offsets,
            tile_n=self.tile_n,
        )
        mxfp6.moe_reduce_out(
            self.output,
            self.gemm2,
            topk_weights,
            inverse_permutation,
            self.gemm2[routes:],
            gate_logits,
        )
        return _maybe_all_reduce(self.output, self.all_reduce)


class MxArrayLayer:
    """Small-batch swapAB path that keeps rows in token/route order."""

    def __init__(
        self,
        weights: MxLayerWeights,
        batch_size: int,
        all_reduce: RuntimeAllReduce | None,
    ) -> None:
        self.weights = weights
        self.all_reduce = all_reduce
        self.fused_router = batch_size in (1, 2, 4, 8)
        self.splitk_w1 = batch_size in (1, 2, 4, 8)
        rows = batch_size * (TOPK + 1)
        device = weights.w1.device
        intermediate_size = weights.w1_padded.shape[1] // 2
        self.quantized = torch.empty(
            (batch_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.uint8,
        )
        self.input_scales = torch.empty(
            (batch_size, HIDDEN_SIZE // MX_BLOCK),
            device=device,
            dtype=torch.uint8,
        )
        self.router_logits = torch.empty(
            (batch_size, NUM_EXPERTS),
            device=device,
            dtype=torch.bfloat16,
        )
        self.topk_weights = torch.empty(
            (batch_size, TOPK),
            device=device,
            dtype=torch.float32,
        )
        self.topk_ids = torch.empty(
            (batch_size, TOPK),
            device=device,
            dtype=torch.int32,
        )
        self.shared_gate = torch.empty(
            (batch_size,),
            device=device,
            dtype=torch.bfloat16,
        )
        self.w1_partial = torch.empty(
            (
                576 if batch_size in (4, 8) else 288,
                64,
            ),
            device=device,
            dtype=torch.float32,
        )
        self.w2_partial = torch.empty(
            (256, 9, 16),
            device=device,
            dtype=torch.float32,
        )
        self.activated = torch.empty(
            (rows, intermediate_size),
            device=device,
            dtype=torch.uint8,
        )
        self.activated_scales = torch.empty(
            (rows, intermediate_size // 32),
            device=device,
            dtype=torch.uint8,
        )
        self.output = torch.empty(
            (batch_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
        )
        self.b1_workspace = (
            mxfp6.Qwen35MoeB1Workspace(
                quantized=self.quantized,
                input_scales=self.input_scales,
                routed_logits=self.router_logits,
                topk_weights=self.topk_weights,
                topk_ids=self.topk_ids,
                shared_gate=self.shared_gate,
                w1_partial=self.w1_partial,
                activation=self.activated,
                activation_scales=self.activated_scales,
                w2_partial=self.w2_partial,
                output=self.output,
            )
            if batch_size == 1
            else None
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.b1_workspace is not None:
            output = mxfp6.qwen35_moe_b1_out(
                self.b1_workspace,
                hidden_states,
                self.weights.combined_gate,
                self.weights.w1,
                self.weights.w1_scale,
                self.weights.w2,
                self.weights.w2_scale,
            )
            return _maybe_all_reduce(output, self.all_reduce)

        if self.fused_router:
            mxfp6.qwen35_router_quant_out(
                self.quantized,
                self.input_scales,
                self.router_logits,
                self.topk_weights,
                self.topk_ids,
                self.shared_gate,
                hidden_states,
                self.weights.combined_gate,
            )
            quantized = self.quantized
            logical_scales = self.input_scales
            topk_weights = self.topk_weights
            topk_ids = self.topk_ids
            shared_gate = self.shared_gate
        else:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
                fused_topk,
            )

            gate_logits = F.linear(
                hidden_states,
                self.weights.combined_gate,
            )
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states,
                gate_logits[:, :NUM_EXPERTS],
                TOPK,
                renormalize=False,
            )
            quantized, logical_scales = mxfp6.quantize_mxfp8_logical(hidden_states)
            shared_gate = gate_logits
        if self.splitk_w1:
            mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
                self.activated,
                self.activated_scales,
                self.w1_partial,
                quantized,
                logical_scales,
                self.weights.w1,
                self.weights.w1_scale,
                topk_ids,
            )
        else:
            mxfp6.array_gemm_w6a8_silu_mxfp8_out(
                self.activated,
                self.activated_scales,
                quantized,
                logical_scales,
                self.weights.w1,
                self.weights.w1_scale,
                topk_ids,
                include_shared=True,
            )
        mxfp6.array_gemm_w6a8_reduce_out(
            self.output,
            self.activated,
            self.activated_scales,
            self.weights.w2,
            self.weights.w2_scale,
            topk_ids,
            topk_weights,
            shared_gate,
        )
        return _maybe_all_reduce(self.output, self.all_reduce)


class MxSplitLayer:
    """Route-array experts with a manually overlapped dense shared expert."""

    def __init__(
        self,
        weights: MxLayerWeights,
        batch_size: int,
        all_reduce: RuntimeAllReduce | None,
        fused_w2: bool | None = None,
        fused_w1: bool | None = None,
    ) -> None:
        self.weights = weights
        self.all_reduce = all_reduce
        self.shared_stream = torch.cuda.Stream()
        self.fused_router = False
        self.fused_w1 = (
            batch_size in (16, 20, 24, 28)
            if fused_w1 is None
            else fused_w1
        )
        self.fused_w2 = False if fused_w2 is None else fused_w2
        rows = batch_size * TOPK
        device = weights.w1.device
        self.quantized = torch.empty(
            (batch_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.uint8,
        )
        self.input_scales = torch.empty(
            (batch_size, HIDDEN_SIZE // MX_BLOCK),
            device=device,
            dtype=torch.uint8,
        )
        self.packed_input_scales = torch.full(
            (
                ((batch_size + 127) // 128 * 128)
                * (HIDDEN_SIZE // MX_BLOCK),
            ),
            0x7F,
            device=device,
            dtype=torch.uint8,
        )
        self.router_logits = torch.empty(
            (batch_size, NUM_EXPERTS),
            device=device,
            dtype=torch.bfloat16,
        )
        self.topk_weights = torch.empty(
            (batch_size, TOPK),
            device=device,
            dtype=torch.float32,
        )
        self.topk_ids = torch.empty(
            (batch_size, TOPK),
            device=device,
            dtype=torch.int32,
        )
        self.shared_gate = torch.empty(
            (batch_size,),
            device=device,
            dtype=torch.bfloat16,
        )
        intermediate_size = weights.w1.shape[1] // 2
        self.gemm1 = torch.empty(
            (rows, weights.w1.shape[1]),
            device=device,
            dtype=torch.bfloat16,
        )
        self.activated = torch.empty(
            (rows, intermediate_size),
            device=device,
            dtype=torch.uint8,
        )
        self.activated_scales = torch.empty(
            (rows, intermediate_size // MX_BLOCK),
            device=device,
            dtype=torch.uint8,
        )
        self.shared_activated = torch.empty(
            (batch_size, intermediate_size),
            device=device,
            dtype=torch.uint8,
        )
        self.shared_activated_scales = torch.full(
            (
                ((batch_size + 127) // 128 * 128)
                * (intermediate_size // MX_BLOCK),
            ),
            0x7F,
            device=device,
            dtype=torch.uint8,
        )
        self.gemm2 = torch.empty(
            (rows, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
        )
        self.output = torch.empty(
            (batch_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
        )
        self.inverse = torch.arange(
            rows,
            device=device,
            dtype=torch.int32,
        )
        self.shared_offsets = torch.tensor(
            [0, batch_size],
            device=device,
            dtype=torch.int64,
        )
        self.shared_w1 = mxfp6.PackedMXFP6Tensor(
            values=weights.w1[-1],
            scales=weights.w1_scale[-1],
            rows=weights.w1.shape[1],
            k=HIDDEN_SIZE,
        )
        self.shared_w2 = mxfp6.PackedMXFP6Tensor(
            values=weights.w2[-1],
            scales=weights.w2_scale[-1],
            rows=HIDDEN_SIZE,
            k=weights.w1.shape[1] // 2,
        )

    def _shared(
        self,
        quantized: mxfp6.MXFP8Tensor,
    ) -> torch.Tensor:
        gate_up = mxfp6.gemm_w6a8(
            quantized,
            self.shared_w1,
            out_dtype=torch.bfloat16,
        )
        mxfp6.silu_and_mul_mxfp8_packed_out(
            self.shared_activated,
            self.shared_activated_scales,
            gate_up,
        )
        activated_operand = mxfp6.MXFP8Tensor(
            values=self.shared_activated,
            scales=self.shared_activated_scales,
            rows=quantized.rows,
            k=gate_up.shape[1] // 2,
        )
        output = mxfp6.gemm_w6a8(
            activated_operand,
            self.shared_w2,
            out_dtype=torch.bfloat16,
        )
        return output

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current = torch.cuda.current_stream()
        self.shared_stream.wait_stream(current)
        with torch.cuda.stream(self.shared_stream):
            shared_gate = F.linear(
                hidden_states,
                self.weights.combined_gate[NUM_EXPERTS:],
            )
        router_logits = F.linear(
            hidden_states,
            self.weights.combined_gate[:NUM_EXPERTS],
        )
        mxfp6.qwen35_topk_quant_out(
            self.quantized,
            self.input_scales,
            self.packed_input_scales,
            self.topk_weights,
            self.topk_ids,
            hidden_states,
            router_logits,
        )
        quantized = self.quantized.view(torch.float8_e4m3fn)
        logical_scales = self.input_scales
        topk_weights = self.topk_weights
        topk_ids = self.topk_ids
        shared_input = mxfp6.MXFP8Tensor(
            values=self.quantized,
            scales=self.packed_input_scales,
            rows=hidden_states.shape[0],
            k=hidden_states.shape[1],
        )
        self.shared_stream.wait_stream(current)
        with torch.cuda.stream(self.shared_stream):
            shared_output = self._shared(shared_input)
        if self.fused_w1:
            mxfp6.array_gemm_w6a8_silu_mxfp8_out(
                self.activated,
                self.activated_scales,
                quantized,
                logical_scales,
                self.weights.w1[:NUM_EXPERTS],
                self.weights.w1_scale[:NUM_EXPERTS],
                topk_ids,
                include_shared=False,
            )
            activated = self.activated
            activated_scales = self.activated_scales
        else:
            mxfp6.array_gemm_w6a8_out(
                self.gemm1,
                quantized,
                logical_scales,
                self.weights.w1,
                self.weights.w1_scale,
                topk_ids,
            )
            activated, activated_scales = mxfp6.silu_and_mul_mxfp8_logical(
                self.gemm1
            )
        if self.fused_w2:
            current.wait_stream(self.shared_stream)
            mxfp6.array_gemm_w6a8_reduce_out(
                self.output,
                activated,
                activated_scales,
                self.weights.w2[:NUM_EXPERTS],
                self.weights.w2_scale[:NUM_EXPERTS],
                topk_ids,
                topk_weights,
                shared_gate,
                shared_output,
            )
        else:
            mxfp6.array_gemm_w6a8_out(
                self.gemm2,
                activated,
                activated_scales,
                self.weights.w2,
                self.weights.w2_scale,
                topk_ids,
            )
            current.wait_stream(self.shared_stream)
            mxfp6.moe_reduce_out(
                self.output,
                self.gemm2,
                topk_weights,
                self.inverse,
                shared_output,
                shared_gate,
            )
        return _maybe_all_reduce(self.output, self.all_reduce)


class MxGroupedSplitLayer(MxSplitLayer):
    """Expert-sorted routed GEMMs overlapped with a dense shared expert."""

    def __init__(
        self,
        weights: MxLayerWeights,
        batch_size: int,
        all_reduce: RuntimeAllReduce | None,
        tile_w1: int,
        tile_w2: int,
        use_pdl: bool,
        merge_shared: bool,
        fused_grouped_w1: bool,
        fused_reduce_ar: bool,
        indirect_route: bool,
    ) -> None:
        super().__init__(weights, batch_size, all_reduce)
        if fused_reduce_ar and all_reduce is None:
            raise ValueError("fused reduce/all-reduce requires TP2")
        if fused_reduce_ar and merge_shared:
            raise ValueError("fused reduce/all-reduce requires separate shared expert")
        self.tile_w1 = tile_w1
        self.tile_w2 = tile_w2
        self.use_pdl = use_pdl
        self.merge_shared = merge_shared
        self.fused_grouped_w1 = fused_grouped_w1
        self.fused_reduce_ar = fused_reduce_ar
        self.indirect_route = indirect_route
        self.fuse_w1_metadata = (
            not merge_shared
            and batch_size * TOPK >= NUM_EXPERTS
            and tile_w1 == -8
            and not fused_grouped_w1
        )
        self.fuse_w2_metadata = (
            not merge_shared
            and batch_size * TOPK >= NUM_EXPERTS
            and tile_w2 == -8
            and not fused_grouped_w1
        )
        routed_rows = batch_size * TOPK
        total_rows = routed_rows + (batch_size if merge_shared else 0)
        local_experts = NUM_EXPERTS + int(merge_shared)
        self.source_tokens = (
            None
            if merge_shared or not indirect_route
            else torch.empty(
                (routed_rows,),
                device=weights.w1.device,
                dtype=torch.int32,
            )
        )
        self._permuted_storage = torch.empty(
            (total_rows + 7, HIDDEN_SIZE),
            device=weights.w1.device,
            dtype=torch.float8_e4m3fn,
        )
        self.permuted = self._permuted_storage[:total_rows]
        active_experts = local_experts
        input_packed_rows = total_rows + active_experts * 127
        activation_packed_rows = (
            total_rows + min(total_rows, local_experts) * 127
        )
        intermediate_size = weights.w1.shape[1] // 2
        if merge_shared:
            self.gemm1 = torch.empty(
                (total_rows, weights.w1.shape[1]),
                device=weights.w1.device,
                dtype=torch.bfloat16,
            )
            self.activated = torch.empty(
                (total_rows, intermediate_size),
                device=weights.w1.device,
                dtype=torch.uint8,
            )
            self.gemm2 = torch.empty(
                (total_rows, HIDDEN_SIZE),
                device=weights.w1.device,
                dtype=torch.bfloat16,
            )
            self.merged_shared_output = self.gemm2[routed_rows:]
        else:
            self.merged_shared_output = None
            self._activated_storage = torch.empty(
                (routed_rows + 7, intermediate_size),
                device=weights.w1.device,
                dtype=torch.uint8,
            )
            self.activated = self._activated_storage[:routed_rows]
        self.grouped_input_scales = torch.full(
            (input_packed_rows * (HIDDEN_SIZE // MX_BLOCK),),
            0x7F,
            device=weights.w1.device,
            dtype=torch.uint8,
        )
        self.expert_cursors = torch.empty(
            (local_experts,),
            device=weights.w1.device,
            dtype=torch.int32,
        )
        self.expert_offsets = torch.empty(
            (local_experts + 1,),
            device=weights.w1.device,
            dtype=torch.int64,
        )
        self.scale_offsets = torch.empty_like(
            self.expert_offsets,
        )
        packed_groups = (
            (intermediate_size // MX_BLOCK + 3) // 4 * 4
        )
        self.grouped_activated_scales = torch.full(
            (activation_packed_rows * packed_groups,),
            0x7F,
            device=weights.w1.device,
            dtype=torch.uint8,
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current = torch.cuda.current_stream()
        self.shared_stream.wait_stream(current)
        with torch.cuda.stream(self.shared_stream):
            shared_gate = F.linear(
                hidden_states,
                self.weights.combined_gate[NUM_EXPERTS:],
            )
        router_logits = F.linear(
            hidden_states,
            self.weights.combined_gate[:NUM_EXPERTS],
        )
        if self.merge_shared:
            mxfp6.qwen35_topk_quant_out(
                self.quantized,
                self.input_scales,
                self.packed_input_scales,
                self.topk_weights,
                self.topk_ids,
                hidden_states,
                router_logits,
            )
        else:
            mxfp6.qwen35_topk_quant_route_out(
                self.quantized,
                self.input_scales,
                self.packed_input_scales,
                self.topk_weights,
                self.topk_ids,
                self.permuted,
                self.grouped_input_scales,
                self.expert_cursors,
                self.expert_offsets,
                self.scale_offsets,
                self.inverse,
                hidden_states,
                router_logits,
                self.gemm1 if self.fuse_w1_metadata else None,
                (
                    self.weights.w1[:NUM_EXPERTS]
                    if self.fuse_w1_metadata
                    else None
                ),
                (
                    self.weights.w1_scale[:NUM_EXPERTS]
                    if self.fuse_w1_metadata
                    else None
                ),
                self.source_tokens,
            )
        topk_weights = self.topk_weights
        quantized = self.quantized.view(torch.float8_e4m3fn)
        shared_input = mxfp6.MXFP8Tensor(
            values=self.quantized,
            scales=self.packed_input_scales,
            rows=hidden_states.shape[0],
            k=hidden_states.shape[1],
        )
        if self.merge_shared:
            shared_output = self.merged_shared_output
        else:
            self.shared_stream.wait_stream(current)
            with torch.cuda.stream(self.shared_stream):
                shared_output = self._shared(shared_input)
        if self.merge_shared:
            mxfp6.qwen35_route_mxfp8_out(
                quantized,
                self.input_scales,
                self.topk_ids,
                self.permuted,
                self.grouped_input_scales,
                self.expert_cursors,
                self.expert_offsets,
                self.scale_offsets,
                self.inverse,
                include_shared=True,
            )
        expert_weights = (
            self.weights.w1
            if self.merge_shared
            else self.weights.w1[:NUM_EXPERTS]
        )
        expert_weight_scales = (
            self.weights.w1_scale
            if self.merge_shared
            else self.weights.w1_scale[:NUM_EXPERTS]
        )
        if self.fused_grouped_w1:
            mxfp6.qwen35_grouped_w1_silu_mxfp8_out(
                self.activated,
                self.grouped_activated_scales,
                (
                    self.quantized
                    if self.source_tokens is not None
                    else self.permuted
                ),
                (
                    self.packed_input_scales
                    if self.source_tokens is not None
                    else self.grouped_input_scales
                ),
                expert_weights,
                expert_weight_scales,
                self.expert_offsets,
                self.scale_offsets,
                use_pdl=self.use_pdl,
                source_tokens=self.source_tokens,
            )
        elif not self.fuse_w1_metadata:
            mxfp6.grouped_gemm_w6a8_out(
                self.gemm1,
                self.permuted,
                self.grouped_input_scales,
                expert_weights,
                expert_weight_scales,
                self.expert_offsets,
                self.scale_offsets,
                tile_n=self.tile_w1,
                use_pdl=self.use_pdl,
            )
        if not self.fused_grouped_w1:
            mxfp6.silu_and_mul_mxfp8_grouped_out(
                self.activated,
                self.grouped_activated_scales,
                self.gemm1,
                self.expert_offsets,
                self.scale_offsets,
                use_pdl=self.use_pdl,
                grouped_output=(
                    self.gemm2 if self.fuse_w2_metadata else None
                ),
                weight=(
                    self.weights.w2[:NUM_EXPERTS]
                    if self.fuse_w2_metadata
                    else None
                ),
                weight_scales=(
                    self.weights.w2_scale[:NUM_EXPERTS]
                    if self.fuse_w2_metadata
                    else None
                ),
            )
        if not self.fuse_w2_metadata:
            mxfp6.grouped_gemm_w6a8_out(
                self.gemm2,
                self.activated,
                self.grouped_activated_scales,
                (
                    self.weights.w2
                    if self.merge_shared
                    else self.weights.w2[:NUM_EXPERTS]
                ),
                (
                    self.weights.w2_scale
                    if self.merge_shared
                    else self.weights.w2_scale[:NUM_EXPERTS]
                ),
                self.expert_offsets,
                self.scale_offsets,
                tile_n=self.tile_w2,
                use_pdl=self.use_pdl,
            )
        current.wait_stream(self.shared_stream)
        if self.fused_reduce_ar:
            assert self.all_reduce is not None
            return self.all_reduce.fused_moe_reduce(
                self.output,
                self.gemm2,
                topk_weights,
                self.inverse,
                shared_output,
                shared_gate,
            )
        mxfp6.moe_reduce_out(
            self.output,
            self.gemm2,
            topk_weights,
            self.inverse,
            shared_output,
            shared_gate,
            use_pdl=self.use_pdl,
        )
        return _maybe_all_reduce(self.output, self.all_reduce)

def _capture(
    function: Callable[[], torch.Tensor],
    warmup: int,
    distributed: bool,
    all_reduce: RuntimeAllReduce | None,
) -> CapturedLayer:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    if distributed:
        dist.barrier()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    if distributed:
        dist.barrier()
    capture_context = all_reduce.capture() if all_reduce is not None else nullcontext()
    with capture_context:
        with torch.cuda.graph(graph, stream=capture_stream):
            output = function()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    if distributed:
        dist.barrier()
    return CapturedLayer(graph=graph, output=output)


def _bench_graph(
    captured: CapturedLayer,
    iterations: int,
    repeats: int,
    distributed: bool,
) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        if distributed:
            dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            captured.graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    latency = statistics.median(samples)
    if distributed:
        value = torch.tensor(latency, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        latency = float(value)
    return latency


def _error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    relative_rms = float(
        difference.square().mean().sqrt()
        / reference_float.square().mean().sqrt().clamp_min(1e-12)
    )
    cosine = float(
        F.cosine_similarity(
            reference_float.flatten(),
            candidate_float.flatten(),
            dim=0,
        )
    )
    return relative_rms, cosine


def _initialize_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
        rank = dist.get_rank()
    else:
        torch.cuda.set_device(0)
        rank = 0
    return rank, world_size, torch.device("cuda", local_rank)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp8-model",
        type=Path,
        default=Path("/data/models/Qwen3.5-35B-A3B-FP8"),
    )
    parser.add_argument(
        "--mx-model",
        type=Path,
        default=Path("/data/models/Qwen3.5-35B-A3B-MXFP6"),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 128],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--tile-n",
        type=int,
        choices=[-14, -13, -11, -10, -9, -8,
                 -6, -4, -3, -1, 0, 8, 16, 32, 64, 128],
        default=0,
    )
    parser.add_argument(
        "--w1-tile-n",
        type=int,
        choices=[-14, -13, -11, -10, -9, -8,
                 -6, -4, -3, -1, 0, 8, 16, 32, 64, 128],
        help="Override the grouped W1 tile selector.",
    )
    parser.add_argument(
        "--w2-tile-n",
        type=int,
        choices=[-14, -13, -11, -10, -9, -8,
                 -6, -4, -3, -1, 0, 8, 16, 32, 64, 128],
        help="Override the grouped W2 tile selector.",
    )
    parser.add_argument(
        "--mx-mode",
        choices=["auto", "array", "grouped", "split", "grouped-split"],
        default="auto",
        help=(
            "Select route-array, grouped, shared-overlap, or the measured "
            "batch-dependent schedule."
        ),
    )
    parser.add_argument(
        "--split-w2",
        choices=["auto", "fused", "unfused"],
        default="auto",
        help="Override the routed W2/reduce schedule in split mode.",
    )
    parser.add_argument(
        "--split-w1",
        choices=["auto", "fused", "unfused"],
        default="auto",
        help="Override the routed W1/activation schedule in split mode.",
    )
    parser.add_argument(
        "--grouped-pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use programmatic dependent launch for the grouped MoE chain.",
    )
    parser.add_argument(
        "--grouped-merge-shared",
        action="store_true",
        help="Schedule the shared expert inside both grouped GEMMs.",
    )
    parser.add_argument(
        "--grouped-fused-w1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override fused grouped Qwen3.5 W1, SiLU, and MXFP8 "
            "quantization."
        ),
    )
    parser.add_argument(
        "--grouped-indirect-route",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let fused W1 gather source tokens without materializing routes.",
    )
    parser.add_argument(
        "--fused-reduce-ar",
        action="store_true",
        help="Fuse grouped local reduction with vLLM's TP2 custom all-reduce.",
    )
    parser.add_argument(
        "--emulated-tp-size",
        type=int,
        default=2,
        help="TP shard size for a single-process diagnostic run.",
    )
    parser.add_argument(
        "--route-stats",
        action="store_true",
        help="Print the routed-expert occupancy used by each measured batch.",
    )
    args = parser.parse_args()

    rank, world_size, device = _initialize_distributed()
    tp_size = world_size if world_size > 1 else args.emulated_tp_size
    tp_rank = rank if world_size > 1 else 0
    distributed = world_size > 1

    import vllm._moe_C_stable_libtorch  # noqa: F401
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(device)
    mxfp6.load_library()
    torch.manual_seed(20260730)
    cpu_group = None
    all_reduce = None
    if distributed:
        cpu_group = dist.new_group(backend="gloo")
        all_reduce = RuntimeAllReduce(cpu_group, device)

    if rank == 0:
        mode = (
            f"TP{tp_size} vLLM custom AR" if distributed else f"TP{tp_size} local-rank0"
        )
        print(f"Loading real layer-0 weights ({mode})...", flush=True)
    fp8_weights = load_fp8_layer(
        args.fp8_model,
        tp_rank,
        tp_size,
        device,
    )
    mx_weights = load_mx_layer(
        args.mx_model,
        tp_rank,
        tp_size,
        device,
    )
    fp8_quant = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=fp8_weights.w1_scale,
        w2_scale=fp8_weights.w2_scale,
        block_shape=[FP8_BLOCK, FP8_BLOCK],
    )
    if distributed:
        dist.barrier()

    if rank == 0:
        print(
            "FP8 baseline: Triton routed MoE + CUTLASS shared expert "
            "(aux stream, fused act+quant)",
            flush=True,
        )
        print(
            "batch  fp8_layer_ms  mxfp6_layer_ms  speedup  rel_rms  cosine",
            flush=True,
        )

    split_fused_w2 = (
        None if args.split_w2 == "auto" else args.split_w2 == "fused"
    )
    split_fused_w1 = (
        None if args.split_w1 == "auto" else args.split_w1 == "fused"
    )
    grouped_w1_tile = (
        (-8 if args.tile_n == 0 else args.tile_n)
        if args.w1_tile_n is None
        else args.w1_tile_n
    )
    grouped_w2_tile = (
        (-11 if args.tile_n == 0 else args.tile_n)
        if args.w2_tile_n is None
        else args.w2_tile_n
    )
    for batch_size in args.batch_sizes:
        torch.manual_seed(20260730 + batch_size)
        grouped_fused_w1 = (
            batch_size <= 96
            if args.grouped_fused_w1 is None
            else args.grouped_fused_w1
        )
        hidden_states = torch.randn(
            (batch_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
        )
        if distributed:
            dist.broadcast(hidden_states, src=0)

        fp8_layer = Fp8Layer(fp8_weights, fp8_quant, all_reduce)
        if args.mx_mode == "auto":
            if batch_size < 20:
                mx_layer = MxSplitLayer(
                    mx_weights,
                    batch_size,
                    all_reduce,
                    split_fused_w2,
                    split_fused_w1,
                )
            else:
                mx_layer = MxGroupedSplitLayer(
                    mx_weights,
                    batch_size,
                    all_reduce,
                    grouped_w1_tile,
                    grouped_w2_tile,
                    args.grouped_pdl,
                    args.grouped_merge_shared,
                    grouped_fused_w1,
                    args.fused_reduce_ar,
                    args.grouped_indirect_route,
                )
        elif args.mx_mode == "array":
            mx_layer = MxArrayLayer(
                mx_weights,
                batch_size,
                all_reduce,
            )
        elif args.mx_mode == "split":
            mx_layer = MxSplitLayer(
                mx_weights,
                batch_size,
                all_reduce,
                split_fused_w2,
                split_fused_w1,
            )
        elif args.mx_mode == "grouped-split":
            mx_layer = MxGroupedSplitLayer(
                mx_weights,
                batch_size,
                all_reduce,
                grouped_w1_tile,
                grouped_w2_tile,
                args.grouped_pdl,
                args.grouped_merge_shared,
                grouped_fused_w1,
                args.fused_reduce_ar,
                args.grouped_indirect_route,
            )
        else:
            mx_layer = MxLayer(
                mx_weights,
                batch_size,
                all_reduce,
                args.tile_n,
            )
        fp8_graph = _capture(
            lambda layer=fp8_layer, states=hidden_states: layer(states),
            args.warmup,
            distributed,
            all_reduce,
        )
        mx_graph = _capture(
            lambda layer=mx_layer, states=hidden_states: layer(states),
            args.warmup,
            distributed,
            (
                None
                if getattr(mx_layer, "fused_reduce_ar", False)
                else all_reduce
            ),
        )
        fp8_graph.graph.replay()
        mx_graph.graph.replay()
        torch.cuda.synchronize()
        relative_rms, cosine = _error(
            fp8_graph.output,
            mx_graph.output,
        )
        if rank == 0 and args.route_stats:
            counts = torch.bincount(
                mx_layer.topk_ids.flatten().to(torch.int64),
                minlength=NUM_EXPERTS,
            )
            active = counts[counts != 0]
            print(
                "route_stats "
                f"active={active.numel()} max={active.max().item()} "
                f"mean={active.float().mean().item():.3f} "
                f"gt8={(active > 8).sum().item()} "
                f"gt16={(active > 16).sum().item()}",
                flush=True,
            )
        fp8_ms = _bench_graph(
            fp8_graph,
            args.iterations,
            args.repeats,
            distributed,
        )
        mx_ms = _bench_graph(
            mx_graph,
            args.iterations,
            args.repeats,
            distributed,
        )
        if rank == 0:
            print(
                f"{batch_size:5d}  {fp8_ms:12.4f}  {mx_ms:14.4f}  "
                f"{fp8_ms / mx_ms:7.3f}x  {relative_rms:7.4f}  "
                f"{cosine:7.4f}",
                flush=True,
            )

        del fp8_graph, mx_graph, fp8_layer, mx_layer, hidden_states
        gc.collect()
        torch.cuda.empty_cache()
        if distributed:
            dist.barrier()

    if distributed:
        assert all_reduce is not None
        all_reduce.close()
        assert cpu_group is not None
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
