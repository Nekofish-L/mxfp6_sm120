from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch


MXFP6_TP2_ADAPTER_VERSION = 3
_EXPECTED_SCHEME = "w_mxfp6_e3m2_a_fp8"


@dataclass(frozen=True)
class _Mxfp6SmallWorkspace:
    quantized: torch.Tensor
    input_scales: torch.Tensor
    routed_logits: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    shared_gate: torch.Tensor
    w1_partial: torch.Tensor
    activation: torch.Tensor
    activation_scales: torch.Tensor
    output: torch.Tensor


def _pack_expert_scales(logical: torch.Tensor) -> torch.Tensor:
    import mxfp6

    experts = logical.shape[0]
    packed = mxfp6.pack_scales(logical.flatten(0, 1).contiguous())
    return packed.view(experts, -1)


def _install_quark_weight_finalizer() -> None:
    from vllm.model_executor.layers.quantization.quark import quark_moe

    QuarkOCP_MX_MoEMethod = quark_moe.QuarkOCP_MX_MoEMethod

    if not getattr(QuarkOCP_MX_MoEMethod, "_mxfp6_tp2_init_installed", False):
        original_init = QuarkOCP_MX_MoEMethod.__init__

        def init(self, *args, **kwargs):
            try:
                original_init(self, *args, **kwargs)
                return
            except NotImplementedError as exc:
                expected_failure = (
                    os.environ.get("QWEN35_MXFP6_TP2") == "1"
                    and getattr(self, "ocp_mx_scheme", None) == _EXPECTED_SCHEME
                    and "dynamic input scales is currently not implemented"
                    in str(exc)
                )
                if not expected_failure:
                    raise

            # The upstream constructor rejects this exact scheme after it has
            # fully parsed the Quark config but before installing its emulation
            # fallback. Complete only those trailing fields; our runner never
            # executes the emulation kernel.
            self.model_type = getattr(
                quark_moe.get_current_vllm_config().model_config.hf_config,
                "model_type",
                None,
            )
            if self.mxfp4_backend is quark_moe.Mxfp4MoeBackend.NONE:
                self.mxfp4_backend = quark_moe.Mxfp4MoeBackend.EMULATION
            self.experts_cls = quark_moe.backend_to_kernel_cls(
                self.mxfp4_backend
            )[0]

        QuarkOCP_MX_MoEMethod.__init__ = init
        QuarkOCP_MX_MoEMethod._mxfp6_tp2_init_installed = True

    if getattr(QuarkOCP_MX_MoEMethod, "_mxfp6_tp2_installed", False):
        return
    original = QuarkOCP_MX_MoEMethod.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        if (
            os.environ.get("QWEN35_MXFP6_TP2") != "1"
            or self.ocp_mx_scheme != _EXPECTED_SCHEME
            or not getattr(self, "_mxfp6_tp2_runner_owned", False)
        ):
            return original(self, layer)

        import mxfp6

        mxfp6.load_library()
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.contiguous(), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.contiguous(), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            _pack_expert_scales(layer.w13_weight_scale.data),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            _pack_expert_scales(layer.w2_weight_scale.data),
            requires_grad=False,
        )
        self._mxfp6_tp2_native = True

    QuarkOCP_MX_MoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    QuarkOCP_MX_MoEMethod._mxfp6_tp2_installed = True


class Mxfp6Tp2MoERunner:
    """Qwen3.5-35B-A3B TP2 runner backed by mxfp6-sm120 grouped MoE."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        parallel = self.moe_config.moe_parallel_config
        routed = self.routed_experts
        failures: list[str] = []
        if parallel.tp_size != 2:
            failures.append(f"tp_size={parallel.tp_size}, expected 2")
        if parallel.ep_size != 1 or parallel.use_ep:
            failures.append("expert parallelism must be disabled")
        if parallel.dp_size != 1 or parallel.pcp_size != 1:
            failures.append("DP and PCP must both be 1")
        if parallel.is_sequence_parallel:
            failures.append("sequence parallelism must be disabled")
        if getattr(parallel, "enable_eplb", False):
            failures.append("EPLB must be disabled")
        if self.enable_dbo:
            failures.append("dual-batch overlap must be disabled")
        if routed.hidden_size != 2048:
            failures.append(f"hidden_size={routed.hidden_size}, expected 2048")
        if routed.global_num_experts != 256:
            failures.append(
                f"num_experts={routed.global_num_experts}, expected 256"
            )
        if routed.local_num_experts != 256 or routed.expert_map is not None:
            failures.append("expert remapping and redundant experts are unsupported")
        if routed.top_k != 8:
            failures.append(f"top_k={routed.top_k}, expected 8")
        if routed.intermediate_size_per_partition != 256:
            failures.append(
                "local intermediate_size="
                f"{routed.intermediate_size_per_partition}, expected 256"
            )
        if self.gate is None:
            failures.append("the replicated routed gate is unavailable")
        if self._shared_experts is None:
            failures.append("a separate shared expert is required")
        from vllm.model_executor.layers.fused_moe.config import MoEActivation

        if routed.activation is not MoEActivation.SILU:
            failures.append(f"activation={routed.activation!r}, expected SILU")
        if routed.scoring_func != "softmax":
            failures.append(
                f"scoring_func={routed.scoring_func!r}, expected 'softmax'"
            )
        if routed.use_grouped_topk or routed.custom_routing_function is not None:
            failures.append("custom and grouped-topk routing are unsupported")
        if routed.w13_bias is not None or routed.w2_bias is not None:
            failures.append("expert bias is unsupported")
        quant_method = routed.quant_method
        if getattr(quant_method, "ocp_mx_scheme", None) != _EXPECTED_SCHEME:
            failures.append(
                "quant scheme="
                f"{getattr(quant_method, 'ocp_mx_scheme', None)!r}, "
                f"expected {_EXPECTED_SCHEME!r}"
            )
        if failures:
            raise RuntimeError(
                "MXFP6 TP2 adapter configuration mismatch: "
                + "; ".join(failures)
            )

        # The Quark class is process-global. Mark only the quantization methods
        # owned by this runner so unrelated MoE layers retain their normal
        # finalization path.
        quant_method._mxfp6_tp2_runner_owned = True

        # This runner returns one tensor that already contains both routed and
        # shared expert contributions. Select the single-output custom-op
        # schema even though the fallback path still owns the vLLM shared
        # expert wrapper.
        self._forward_entry = torch.ops.vllm.moe_forward

        self._mxfp6_max_tokens = int(
            os.environ.get("QWEN35_MXFP6_TP2_MAX_TOKENS", "96")
        )
        if not 1 <= self._mxfp6_max_tokens <= 96:
            raise ValueError(
                "QWEN35_MXFP6_TP2_MAX_TOKENS must be between 1 and 96"
            )
        self._mxfp6_b4_vector_loads = (
            os.environ.get("QWEN35_MXFP6_B4_VECTOR_LOADS") == "1"
        )
        self._mxfp6_storage: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._mxfp6_workspace_views: dict[int, tuple[Any, torch.Tensor]] = {}
        self._mxfp6_b1_workspace: Any | None = None
        self._mxfp6_small_workspaces: dict[int, _Mxfp6SmallWorkspace] = {}
        self._mxfp6_shared_tensors: dict[str, torch.Tensor] = {}
        self._mxfp6_ready = False

    def maybe_init_modular_kernel(self) -> None:
        # The custom runner owns routing and expert execution.  Prevent vLLM
        # from asking the intentionally bypassed Quark emulation method to
        # construct a modular kernel.
        self._initialize_mxfp6()

    def _initialize_mxfp6(self) -> None:
        if self._mxfp6_ready:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MXFP6 TP2 adapter must be initialized before CUDA Graph capture"
            )

        import mxfp6

        quant_method = self.routed_experts.quant_method
        if not getattr(quant_method, "_mxfp6_tp2_native", False):
            raise RuntimeError(
                "MXFP6 TP2 packed weights were not finalized by the adapter"
            )
        tensors = {
            "w13_weight": (
                self.routed_experts.w13_weight,
                (256, 512, 1536),
            ),
            "w2_weight": (
                self.routed_experts.w2_weight,
                (256, 2048, 192),
            ),
            "w13_weight_scale": (
                self.routed_experts.w13_weight_scale,
                (256, 32768),
            ),
            "w2_weight_scale": (
                self.routed_experts.w2_weight_scale,
                (256, 16384),
            ),
        }
        assert self._shared_experts is not None
        shared_layer = self._shared_experts._layer
        shared_tensors = {
            "routed_gate": self.gate.weight,
            "shared_gate": shared_layer.expert_gate.weight,
            "shared_w1": shared_layer.gate_up_proj.weight.unsqueeze(0),
            "shared_w1_scale": shared_layer.gate_up_proj.weight_scale,
            "shared_w2": shared_layer.down_proj.weight.unsqueeze(0),
            "shared_w2_scale": shared_layer.down_proj.weight_scale,
        }
        shared_expected = {
            "routed_gate": (torch.bfloat16, (256, 2048)),
            "shared_gate": (torch.bfloat16, (1, 2048)),
            "shared_w1": (torch.uint8, (1, 512, 1536)),
            "shared_w2": (torch.uint8, (1, 2048, 192)),
        }
        failures: list[str] = []
        devices: set[torch.device] = set()
        for name, (tensor, expected_shape) in tensors.items():
            if tuple(tensor.shape) != expected_shape:
                failures.append(
                    f"{name}.shape={tuple(tensor.shape)}, expected {expected_shape}"
                )
            if tensor.dtype is not torch.uint8:
                failures.append(f"{name}.dtype={tensor.dtype}, expected uint8")
            if not tensor.is_cuda:
                failures.append(f"{name} must be a CUDA tensor")
            else:
                devices.add(tensor.device)
            if not tensor.is_contiguous():
                failures.append(f"{name} must be contiguous")
        for name, (expected_dtype, expected_shape) in shared_expected.items():
            tensor = shared_tensors[name]
            if tensor.dtype is not expected_dtype:
                failures.append(
                    f"{name}.dtype={tensor.dtype}, expected {expected_dtype}"
                )
            if tuple(tensor.shape) != expected_shape:
                failures.append(
                    f"{name}.shape={tuple(tensor.shape)}, expected {expected_shape}"
                )
            if not tensor.is_cuda:
                failures.append(f"{name} must be a CUDA tensor")
            else:
                devices.add(tensor.device)
            if not tensor.is_contiguous():
                failures.append(f"{name} must be contiguous")
        for name, minimum in (
            ("shared_w1_scale", 512 * 64),
            ("shared_w2_scale", 2048 * 8),
        ):
            tensor = shared_tensors[name]
            if tensor.dtype is not torch.uint8 or tensor.numel() < minimum:
                failures.append(
                    f"{name} must be uint8 with at least {minimum} values"
                )
            if not tensor.is_cuda:
                failures.append(f"{name} must be a CUDA tensor")
            else:
                devices.add(tensor.device)
            if not tensor.is_contiguous():
                failures.append(f"{name} must be contiguous")
        if len(devices) != 1:
            failures.append(f"weights span {len(devices)} CUDA devices")
        elif torch.cuda.get_device_capability(next(iter(devices))) != (12, 0):
            failures.append(
                "MXFP6 TP2 requires an SM120 GPU, got compute capability "
                f"{torch.cuda.get_device_capability(next(iter(devices)))}"
            )
        if failures:
            raise RuntimeError(
                "MXFP6 TP2 weight/ABI mismatch: " + "; ".join(failures)
            )

        mxfp6.load_library()
        first_shape, second_shape = mxfp6.qwen35_grouped_workspace_shapes(
            self._mxfp6_max_tokens
        )
        device = next(iter(devices))
        self._mxfp6_storage = (
            torch.empty(
                (self._mxfp6_max_tokens, 2048),
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.empty(
                (first_shape[0] * first_shape[1],),
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.empty(
                (second_shape[0] * second_shape[1],),
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        b1_output = self._mxfp6_storage[0][:1]
        self._mxfp6_b1_workspace = mxfp6.Qwen35MoeB1Workspace(
            quantized=torch.empty((1, 2048), device=device, dtype=torch.uint8),
            input_scales=torch.empty((1, 64), device=device, dtype=torch.uint8),
            routed_logits=torch.empty(
                (1, 256), device=device, dtype=torch.bfloat16
            ),
            topk_weights=torch.empty((1, 8), device=device, dtype=torch.float32),
            topk_ids=torch.empty((1, 8), device=device, dtype=torch.int32),
            shared_gate=torch.empty((1,), device=device, dtype=torch.bfloat16),
            w1_partial=torch.empty((576, 64), device=device, dtype=torch.float32),
            activation=torch.empty((9, 256), device=device, dtype=torch.uint8),
            activation_scales=torch.empty((9, 8), device=device, dtype=torch.uint8),
            w2_partial=torch.empty(
                (256, 9, 16), device=device, dtype=torch.float32
            ),
            output=b1_output,
        )
        for batch_size in (2, 4):
            self._mxfp6_small_workspaces[batch_size] = _Mxfp6SmallWorkspace(
                quantized=torch.empty(
                    (batch_size, 2048), device=device, dtype=torch.uint8
                ),
                input_scales=torch.empty(
                    (batch_size, 64), device=device, dtype=torch.uint8
                ),
                routed_logits=torch.empty(
                    (batch_size, 256), device=device, dtype=torch.bfloat16
                ),
                topk_weights=torch.empty(
                    (batch_size, 8), device=device, dtype=torch.float32
                ),
                topk_ids=torch.empty(
                    (batch_size, 8), device=device, dtype=torch.int32
                ),
                shared_gate=torch.empty(
                    (batch_size,), device=device, dtype=torch.bfloat16
                ),
                w1_partial=torch.empty(
                    (576, 64), device=device, dtype=torch.float32
                ),
                activation=torch.empty(
                    (batch_size * 9, 256), device=device, dtype=torch.uint8
                ),
                activation_scales=torch.empty(
                    (batch_size * 9, 8), device=device, dtype=torch.uint8
                ),
                output=self._mxfp6_storage[0][:batch_size],
            )
        self._mxfp6_shared_tensors = shared_tensors
        for batch_size in range(1, self._mxfp6_max_tokens + 1):
            output = self._mxfp6_storage[0][:batch_size]
            workspace = mxfp6.Qwen35GroupedWorkspace.from_storage(
                output,
                self._mxfp6_storage[1],
                self._mxfp6_storage[2],
            )
            self._mxfp6_workspace_views[batch_size] = (workspace, output)
        self._mxfp6_ready = True

    def _get_workspace(self, batch_size: int):
        if batch_size > self._mxfp6_max_tokens:
            raise RuntimeError(
                f"internal MXFP6 chunk exceeds {self._mxfp6_max_tokens} tokens: "
                f"got {batch_size}"
            )
        if not self._mxfp6_ready:
            self._initialize_mxfp6()
        cached = self._mxfp6_workspace_views.get(batch_size)
        assert cached is not None
        return cached

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ):
        del router_logits, input_ids
        self._initialize_mxfp6()
        assert shared_experts_input is not None

        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )
        import mxfp6

        num_tokens = hidden_states.shape[0]
        if num_tokens == 1:
            workspace = self._mxfp6_b1_workspace
            assert workspace is not None
            shared = self._mxfp6_shared_tensors
            mxfp6.qwen35_router_quant_out(
                workspace.quantized,
                workspace.input_scales,
                workspace.routed_logits,
                workspace.topk_weights,
                workspace.topk_ids,
                workspace.shared_gate,
                hidden_states,
                shared["routed_gate"],
                renormalize=self.routed_experts.renormalize,
                shared_gate_weight=shared["shared_gate"],
            )
            mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
                workspace.activation,
                workspace.activation_scales,
                workspace.w1_partial,
                workspace.quantized,
                workspace.input_scales,
                self.routed_experts.w13_weight,
                self.routed_experts.w13_weight_scale,
                workspace.topk_ids,
                shared_weight=shared["shared_w1"],
                shared_weight_scales=shared["shared_w1_scale"],
            )
            mxfp6.qwen35_w2_splitk_reduce_out(
                workspace.output,
                workspace.w2_partial,
                workspace.activation,
                workspace.activation_scales,
                self.routed_experts.w2_weight,
                self.routed_experts.w2_weight_scale,
                workspace.topk_ids,
                workspace.topk_weights,
                workspace.shared_gate,
                shared_weight=shared["shared_w2"],
                shared_weight_scales=shared["shared_w2_scale"],
            )
            return workspace.output

        if num_tokens in (2, 4):
            workspace = self._mxfp6_small_workspaces[num_tokens]
            shared = self._mxfp6_shared_tensors
            mxfp6.qwen35_router_quant_out(
                workspace.quantized,
                workspace.input_scales,
                workspace.routed_logits,
                workspace.topk_weights,
                workspace.topk_ids,
                workspace.shared_gate,
                hidden_states,
                shared["routed_gate"],
                renormalize=self.routed_experts.renormalize,
                shared_gate_weight=shared["shared_gate"],
            )
            mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
                workspace.activation,
                workspace.activation_scales,
                workspace.w1_partial,
                workspace.quantized,
                workspace.input_scales,
                self.routed_experts.w13_weight,
                self.routed_experts.w13_weight_scale,
                workspace.topk_ids,
                shared_weight=shared["shared_w1"],
                shared_weight_scales=shared["shared_w1_scale"],
            )
            mxfp6.array_gemm_w6a8_reduce_out(
                workspace.output,
                workspace.activation,
                workspace.activation_scales,
                self.routed_experts.w2_weight,
                self.routed_experts.w2_weight_scale,
                workspace.topk_ids,
                workspace.topk_weights,
                workspace.shared_gate,
                shared_weight=shared["shared_w2"],
                shared_weight_scales=shared["shared_w2_scale"],
                use_packed_vector_loads=(
                    num_tokens == 4 and self._mxfp6_b4_vector_loads
                ),
            )
            return workspace.output

        self._maybe_sync_shared_experts_stream(shared_experts_input)
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )
        routed_logits, _ = self.gate(hidden_states)
        if num_tokens <= self._mxfp6_max_tokens:
            chunks = ((0, num_tokens),)
            _, output = self._get_workspace(num_tokens)
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "CUDA Graph capture above QWEN35_MXFP6_TP2_MAX_TOKENS is "
                    "unsupported; lower the capture size or raise the validated "
                    "chunk limit"
                )
            output = torch.empty_like(hidden_states)
            chunks = tuple(
                (start, min(start + self._mxfp6_max_tokens, num_tokens))
                for start in range(0, num_tokens, self._mxfp6_max_tokens)
            )
        for start, end in chunks:
            workspace, workspace_output = self._get_workspace(end - start)
            chunk_output = (
                output if num_tokens <= self._mxfp6_max_tokens else output[start:end]
            )
            # The workspace owns its output view. For long prefills reuse the
            # bounded workspace, then copy each completed chunk to the result.
            mxfp6.qwen35_grouped_moe_logits_out(
                workspace,
                workspace_output,
                hidden_states[start:end],
                routed_logits[start:end],
                self.routed_experts.w13_weight,
                self.routed_experts.w13_weight_scale,
                self.routed_experts.w2_weight,
                self.routed_experts.w2_weight_scale,
                renormalize=self.routed_experts.renormalize,
            )
            if chunk_output.data_ptr() != workspace_output.data_ptr():
                chunk_output.copy_(workspace_output)
        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )
        shared_output = self._shared_experts.output
        output.add_(shared_output)
        return output


@lru_cache(maxsize=1)
def mxfp6_tp2_runner_cls():
    _install_quark_weight_finalizer()
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    class _Mxfp6Tp2MoERunner(Mxfp6Tp2MoERunner, MoERunner):
        pass

    return _Mxfp6Tp2MoERunner
