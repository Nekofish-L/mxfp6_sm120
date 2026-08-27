# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MXFP6-weight MoE experts for NVIDIA SM120."""

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
    _moe_forward_fake,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from mxfp6_sm120_utils import (
    is_mxfp6_sm120_moe_available,
    mxfp6_sm120_grouped_gemm,
    mxfp6_sm120_quantize_mxfp8,
    mxfp6_sm120_qwen35_grouped_moe,
    mxfp6_sm120_reduce,
    mxfp6_sm120_route,
    mxfp6_sm120_silu_grouped_gemm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.utils import replace_parameter
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_QWEN35_MOE_GRAPH_CACHE_VERSION = "qwen35_batch_dispatch_v4_b4_vector_cluster_gate"
_QWEN35_SMALL_BATCH_MAX_TOKENS = 4
_QWEN35_GROUPED_MIN_TOKENS = 5
_QWEN35_GROUPED_MAX_TOKENS = 96


def _qwen35_moe_schedule(
    num_tokens: int,
) -> Literal["small_batch", "grouped", "generic"]:
    if 1 <= num_tokens <= _QWEN35_SMALL_BATCH_MAX_TOKENS:
        return "small_batch"
    if _QWEN35_GROUPED_MIN_TOKENS <= num_tokens <= _QWEN35_GROUPED_MAX_TOKENS:
        return "grouped"
    return "generic"


direct_register_custom_op(
    op_name="mxfp6_sm120_moe_forward",
    op_func=_moe_forward,
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


@dataclass(frozen=True)
class _Qwen35MoeSmallBatchWorkspace:
    padded_hidden_states: torch.Tensor
    quantized: torch.Tensor
    input_scales: torch.Tensor
    routed_logits: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    shared_gate: torch.Tensor
    w1_partial: torch.Tensor
    activation: torch.Tensor
    activation_scales: torch.Tensor
    w2_partial: torch.Tensor | None
    output: torch.Tensor


@dataclass(frozen=True)
class _Qwen35MoeSmallBatchState:
    combined_gate: torch.Tensor
    w1: torch.Tensor
    w1_scales: torch.Tensor
    w2: torch.Tensor
    w2_scales: torch.Tensor
    workspaces: dict[int, _Qwen35MoeSmallBatchWorkspace]


def _qwen35_moe_small_batch(
    state: _Qwen35MoeSmallBatchState,
    hidden_states: torch.Tensor,
):
    import mxfp6

    tokens = hidden_states.shape[0]
    padded_tokens = 1 << (tokens - 1).bit_length()
    workspace = state.workspaces[padded_tokens]
    if tokens == padded_tokens:
        kernel_input = hidden_states
    else:
        workspace.padded_hidden_states[:tokens].copy_(hidden_states)
        kernel_input = workspace.padded_hidden_states

    logger.info_once("Executing Qwen3.5-35B TP2 MXFP6 fused small-batch schedule")
    mxfp6.qwen35_router_quant_out(
        workspace.quantized,
        workspace.input_scales,
        workspace.routed_logits,
        workspace.topk_weights,
        workspace.topk_ids,
        workspace.shared_gate,
        kernel_input,
        state.combined_gate,
        renormalize=True,
    )
    mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
        workspace.activation,
        workspace.activation_scales,
        workspace.w1_partial,
        workspace.quantized,
        workspace.input_scales,
        state.w1,
        state.w1_scales,
        workspace.topk_ids,
    )
    if padded_tokens == 1:
        assert workspace.w2_partial is not None
        mxfp6.qwen35_w2_splitk_reduce_out(
            workspace.output,
            workspace.w2_partial,
            workspace.activation,
            workspace.activation_scales,
            state.w2,
            state.w2_scales,
            workspace.topk_ids,
            workspace.topk_weights,
            workspace.shared_gate,
        )
    else:
        mxfp6.array_gemm_w6a8_reduce_out(
            workspace.output,
            workspace.activation,
            workspace.activation_scales,
            state.w2,
            state.w2_scales,
            workspace.topk_ids,
            workspace.topk_weights,
            workspace.shared_gate,
            use_packed_vector_loads=padded_tokens == 4,
        )
    return workspace.output[:tokens]


def _qwen35_grouped_workspace(hidden_states: torch.Tensor):
    import mxfp6

    tokens, hidden_size = hidden_states.shape
    workspace13_shape, workspace2_shape = mxfp6.qwen35_grouped_workspace_shapes(tokens)
    common_size = max(prod(workspace13_shape), tokens * hidden_size)
    common, workspace2 = current_workspace_manager().get_simultaneous(
        ((common_size,), torch.bfloat16),
        (workspace2_shape, torch.bfloat16),
    )
    output = _resize_cache(common, (tokens, hidden_size))
    workspace13 = _resize_cache(common, workspace13_shape)
    workspace = mxfp6.Qwen35GroupedWorkspace.from_storage(
        output,
        workspace13,
        workspace2,
    )
    return workspace, output


@MoERunner.register_oot(name="MoERunner")
class Mxfp6Sm120MoERunner(MoERunner):
    """MoE runner with optional package-owned Qwen3.5 fast paths."""

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = getattr(self, "_mxfp6_sm120_qwen35_small_batch_state", None)
        schedule = _qwen35_moe_schedule(hidden_states.shape[0])
        if (
            state is not None
            and hidden_states.shape[1] == 2048
            and schedule == "small_batch"
        ):
            return _qwen35_moe_small_batch(state, hidden_states)
        if (
            state is not None
            and hidden_states.shape[1] == 2048
            and schedule == "grouped"
            and shared_experts_input is not None
            and self.shared_experts is not None
        ):
            import mxfp6

            logger.info_once(
                "Executing Qwen3.5-35B TP2 MXFP6 indirect grouped schedule"
            )
            self.routed_experts._ensure_moe_quant_config_init()
            self._maybe_sync_shared_experts_stream(shared_experts_input)
            if self.gate is not None:
                router_logits, _ = self.gate(hidden_states)

            with self._sequence_parallel_context():
                hidden_states, router_logits = self._maybe_dispatch(
                    hidden_states,
                    router_logits,
                )
                self._maybe_apply_shared_experts(
                    shared_experts_input,
                    SharedExpertsOrder.NO_OVERLAP,
                )
                workspace, output = _qwen35_grouped_workspace(hidden_states)
                mxfp6.qwen35_grouped_gemm_out(
                    workspace,
                    hidden_states,
                    router_logits,
                    state.w1[:256],
                    state.w1_scales[:256],
                    state.w2[:256],
                    state.w2_scales[:256],
                    renormalize=True,
                )
                self._maybe_apply_shared_experts(
                    shared_experts_input,
                    SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
                )
                shared_output = self.shared_experts.output
                mxfp6.qwen35_grouped_reduce_out(
                    workspace,
                    output,
                    shared_output,
                )
                return output
        result = super()._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )
        if state is None:
            return result
        shared_output, fused_output = result
        return shared_output + fused_output


def _append_shared_parameter(
    routed_module: torch.nn.Module,
    routed_name: str,
    shared_module: torch.nn.Module,
    shared_name: str,
) -> torch.Tensor:
    routed = getattr(routed_module, routed_name)
    shared = getattr(shared_module, shared_name)
    combined = torch.cat((routed.data, shared.data.unsqueeze(0)), dim=0)
    replace_parameter(routed_module, routed_name, combined)
    combined_parameter = getattr(routed_module, routed_name)
    replace_parameter(shared_module, shared_name, combined_parameter.data[-1])
    return combined_parameter


def try_enable_qwen35_moe_small_batch(layer: torch.nn.Module) -> bool:
    """Install the Qwen3.5-35B TP2 small-batch schedule when compatible."""
    moe = layer.moe_config
    vllm_config = get_current_vllm_config()
    if (
        moe.hidden_dim != 2048
        or moe.intermediate_size_per_partition != 256
        or moe.num_experts != 256
        or moe.num_local_experts != 256
        or moe.experts_per_token != 8
        or moe.tp_size != 2
        or moe.ep_size != 1
        or moe.dp_size != 1
        or moe.pcp_size != 1
        or moe.is_sequence_parallel
        or moe.skip_final_all_reduce
        or moe.in_dtype != torch.bfloat16
        or layer.expert_map is not None
        or not layer.renormalize
        or layer.scoring_func != "softmax"
        or layer.custom_routing_function is not None
        or layer.routed_scaling_factor != 1.0
        or not isinstance(vllm_config.additional_config, dict)
    ):
        return False

    runner = vllm_config.compilation_config.static_forward_context.get(layer.layer_name)
    if (
        not isinstance(runner, Mxfp6Sm120MoERunner)
        or runner.gate is None
        or runner.shared_experts is None
        or runner.enable_dbo
    ):
        return False
    shared_layer = runner.shared_experts._layer
    if (
        not hasattr(shared_layer, "gate_up_proj")
        or not hasattr(shared_layer, "down_proj")
        or shared_layer.expert_gate is None
    ):
        return False

    expected_shapes = (
        (layer.w13_weight, (256, 512, 1536)),
        (shared_layer.gate_up_proj.weight, (512, 1536)),
        (layer.w2_weight, (256, 2048, 192)),
        (shared_layer.down_proj.weight, (2048, 192)),
        (layer.w13_weight_scale, (256, 512 * 64)),
        (shared_layer.gate_up_proj.weight_scale, (512 * 64,)),
        (layer.w2_weight_scale, (256, 2048 * 8)),
        (shared_layer.down_proj.weight_scale, (2048 * 8,)),
        (runner.gate.weight, (256, 2048)),
        (shared_layer.expert_gate.weight, (1, 2048)),
    )
    if any(tuple(tensor.shape) != shape for tensor, shape in expected_shapes):
        logger.debug(
            "Skipping Qwen3.5 MXFP6 small-batch specialization for %s due "
            "to weight shapes",
            layer.layer_name,
        )
        return False

    w1 = _append_shared_parameter(
        layer,
        "w13_weight",
        shared_layer.gate_up_proj,
        "weight",
    )
    w2 = _append_shared_parameter(
        layer,
        "w2_weight",
        shared_layer.down_proj,
        "weight",
    )
    w1_scales = _append_shared_parameter(
        layer,
        "w13_weight_scale",
        shared_layer.gate_up_proj,
        "weight_scale",
    )
    w2_scales = _append_shared_parameter(
        layer,
        "w2_weight_scale",
        shared_layer.down_proj,
        "weight_scale",
    )
    combined_gate = torch.cat(
        (runner.gate.weight.data, shared_layer.expert_gate.weight.data),
        dim=0,
    )

    device = layer.w13_weight.device
    workspaces: dict[int, _Qwen35MoeSmallBatchWorkspace] = {}
    for tokens in (1, 2, 4, 8):
        routes = tokens * 9
        workspaces[tokens] = _Qwen35MoeSmallBatchWorkspace(
            padded_hidden_states=torch.zeros(
                (tokens, 2048), device=device, dtype=torch.bfloat16
            ),
            quantized=torch.empty((tokens, 2048), device=device, dtype=torch.uint8),
            input_scales=torch.empty((tokens, 64), device=device, dtype=torch.uint8),
            routed_logits=torch.empty(
                (tokens, 256), device=device, dtype=torch.bfloat16
            ),
            topk_weights=torch.empty((tokens, 8), device=device, dtype=torch.float32),
            topk_ids=torch.empty((tokens, 8), device=device, dtype=torch.int32),
            shared_gate=torch.empty((tokens,), device=device, dtype=torch.bfloat16),
            w1_partial=torch.empty(
                (576 if tokens in (4, 8) else 288, 64),
                device=device,
                dtype=torch.float32,
            ),
            activation=torch.empty((routes, 256), device=device, dtype=torch.uint8),
            activation_scales=torch.empty(
                (routes, 8), device=device, dtype=torch.uint8
            ),
            w2_partial=(
                torch.empty((256, 9, 16), device=device, dtype=torch.float32)
                if tokens == 1
                else None
            ),
            output=torch.empty((tokens, 2048), device=device, dtype=torch.bfloat16),
        )
    runner._mxfp6_sm120_qwen35_small_batch_state = _Qwen35MoeSmallBatchState(
        combined_gate=combined_gate,
        w1=w1,
        w1_scales=w1_scales,
        w2=w2,
        w2_scales=w2_scales,
        workspaces=workspaces,
    )
    runner._forward_entry = torch.ops.vllm.mxfp6_sm120_moe_forward
    # This post-load op selection is not part of Dynamo's traced-file cache key.
    vllm_config.additional_config["mxfp6_sm120_moe_graph"] = (
        _QWEN35_MOE_GRAPH_CACHE_VERSION
    )
    logger.info_once(
        "Enabled Qwen3.5-35B TP2 MXFP6 fused small-batch router/shared-expert schedule"
    )
    return True


class Mxfp6Sm120Experts(mk.FusedMoEExpertsModular):
    """SM120 grouped MMA with MXFP8 activations and packed MXFP6 weights."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        if quant_config.w1_scale is None or quant_config.w2_scale is None:
            raise ValueError("MXFP6 SM120 MoE requires packed weight scales")
        if quant_config.w1_bias is not None or quant_config.w2_bias is not None:
            raise ValueError("MXFP6 SM120 MoE does not support expert bias")

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_mxfp6_sm120_moe_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        del weight_key, activation_key
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_batched_activation_format

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % 128 == 0

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del expert_tokens_meta
        if (
            20 <= M <= 96
            and N == 512
            and K == 2048
            and topk == 8
            and global_num_experts == 256
            and local_num_experts == 256
            and activation == MoEActivation.SILU
        ):
            import mxfp6

            workspace13, workspace2 = mxfp6.qwen35_grouped_workspace_shapes(M)
            return workspace13, workspace2, (M, K)
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(activation_out_dim, K))
        output = (M, K)
        return workspace13, workspace2, output

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del a1q_scale, a2_scale, apply_router_weight_on_input
        if expert_tokens_meta is not None:
            raise ValueError("MXFP6 SM120 MoE requires standard routed activations")
        if activation != MoEActivation.SILU:
            raise ValueError("MXFP6 SM120 MoE currently supports only SiLU")

        tokens, hidden_size = hidden_states.shape
        topk = topk_ids.shape[1]
        w1_scale = self.w1_scale
        w2_scale = self.w2_scale
        assert w1_scale is not None and w2_scale is not None
        if (
            expert_map is None
            and w1.shape[0] == global_num_experts + 1
            and w2.shape[0] == global_num_experts + 1
        ):
            w1 = w1[:global_num_experts]
            w2 = w2[:global_num_experts]
            w1_scale = w1_scale[:global_num_experts]
            w2_scale = w2_scale[:global_num_experts]
        local_experts = w1.shape[0]
        routed_rows = tokens * topk
        if w1.shape[2] * 4 != hidden_size * 3:
            raise ValueError("w1 packed K does not match the activation")
        intermediate_size = w2.shape[2] * 4 // 3
        if w1.shape[1] != 2 * intermediate_size or w2.shape[1] != hidden_size:
            raise ValueError("MXFP6 MoE weight shapes are inconsistent")

        use_qwen35_grouped = (
            20 <= tokens <= 96
            and hidden_states.dtype == torch.bfloat16
            and hidden_size == 2048
            and topk == 8
            and global_num_experts == 256
            and local_experts == 256
            and intermediate_size == 256
            and expert_map is None
            and topk_ids.dtype == torch.int32
            and tuple(w1.shape) == (256, 512, 1536)
            and tuple(w2.shape) == (256, 2048, 192)
        )
        if use_qwen35_grouped:
            mxfp6_sm120_qwen35_grouped_moe(
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
            return

        quantized, logical_scales = mxfp6_sm120_quantize_mxfp8(hidden_states)
        permuted_values = _resize_cache(
            workspace2.view(torch.float8_e4m3fn),
            (routed_rows, hidden_size),
        )
        if expert_map is None and global_num_experts != local_experts:
            raise ValueError("global and local expert counts require an expert_map")
        (
            packed_scales,
            expert_offsets,
            scale_offsets,
            inverse_permutation,
        ) = mxfp6_sm120_route(
            permuted_values,
            quantized,
            logical_scales,
            topk_ids,
            expert_map,
            local_experts,
        )

        gemm1_output = _resize_cache(
            workspace13,
            (routed_rows, w1.shape[1]),
        )
        mxfp6_sm120_grouped_gemm(
            gemm1_output,
            permuted_values,
            packed_scales,
            w1,
            w1_scale,
            expert_offsets,
            scale_offsets,
        )

        gemm2_output = _resize_cache(
            workspace2,
            (routed_rows, hidden_size),
        )
        mxfp6_sm120_silu_grouped_gemm(
            gemm2_output,
            gemm1_output,
            w2,
            w2_scale,
            expert_offsets,
            scale_offsets,
        )
        mxfp6_sm120_reduce(
            output,
            gemm2_output,
            topk_weights,
            inverse_permutation,
        )


def make_mxfp6_sm120_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> mk.FusedMoEKernel:
    """Construct the standard prepare/experts/finalize SM120 MoE chain."""
    prepare_finalize = maybe_make_prepare_finalize(
        moe_config,
        moe_quant_config,
        routing_tables,
        allow_new_interface=True,
    )
    if (
        prepare_finalize is None
        or prepare_finalize.activation_format != mk.FusedMoEActivationFormat.Standard
    ):
        raise ValueError("MXFP6 SM120 MoE requires the standard activation format")
    experts = Mxfp6Sm120Experts(moe_config, moe_quant_config)
    return mk.FusedMoEKernel(prepare_finalize, experts)
