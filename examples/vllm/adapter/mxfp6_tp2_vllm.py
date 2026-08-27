from __future__ import annotations


import torch

from vllm.model_executor.kernels.linear.mxfp6.base import (
    MxFp6LinearKernel,
    MxFp6LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


MXFP6_TP2_ADAPTER_VERSION = 3
_EXPECTED_SCHEME = "w_mxfp6_e3m2_a_fp8"


def _mxfp6_sm120_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    import mxfp6

    mxfp6.load_library()
    return torch.ops.mxfp6.gemm_from_float(
        x,
        weight,
        weight_scale,
        output_features,
        1.0,
        output_dtype,
    )


def _mxfp6_sm120_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    del weight, weight_scale
    return torch.empty(
        (x.shape[0], output_features),
        device=x.device,
        dtype=output_dtype,
    )


direct_register_custom_op(
    op_name="mxfp6_sm120_gemm",
    op_func=_mxfp6_sm120_gemm_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_gemm_fake,
)


class Mxfp6Sm120LinearKernel(MxFp6LinearKernel):
    """Native SM120 W6A8 backend for vLLM's MXFP6 linear interface."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = (
                capability[0] * 10 + capability[1] if capability is not None else None
            )
        if compute_capability != 120:
            return False, "requires an SM120 GPU"
        try:
            import mxfp6

            required = (
                "begin_workspace_planning",
                "finalize_workspace_planning",
                "gemm_w6a8",
                "is_available",
                "load_library",
                "pack_scales",
                "warmup_w6a8",
                "workspace_stats",
            )
            if not all(hasattr(mxfp6, name) for name in required):
                return False, "mxfp6-sm120 does not provide the required W6A8 API"
            mxfp6.load_library()
            if not mxfp6.is_available():
                return False, "mxfp6-sm120 is unavailable on this device"
        except Exception as exc:
            return False, f"mxfp6-sm120 failed to load: {exc}"
        return True, None

    @classmethod
    def can_implement(
        cls, config: MxFp6LinearLayerConfig
    ) -> tuple[bool, str | None]:
        if config.weight_quant_key != kMxfp6E3M2Static:
            return False, "requires static MXFP6 E3M2 weights"
        if config.activation_quant_key != kMxfp8Dynamic:
            return False, "requires dynamic MXFP8 E4M3 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import mxfp6

        layer.weight = torch.nn.Parameter(
            layer.weight.data.contiguous(), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            mxfp6.pack_scales(layer.weight_scale.data.contiguous()),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_features = layer.weight.shape[0]
        output_shape = (*x.shape[:-1], output_features)
        y = torch.ops.vllm.mxfp6_sm120_gemm(
            x.reshape(-1, x.shape[-1]).contiguous(),
            layer.weight,
            layer.weight_scale,
            output_features,
            x.dtype,
        ).reshape(output_shape)
        if bias is not None:
            y = y + bias
        return y


def _collect_w6a8_problems(
    model: torch.nn.Module,
) -> list[tuple[int, int, torch.Tensor, torch.Tensor]]:
    problems: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in model.modules():
        scheme = getattr(layer, "scheme", None)
        kernel = getattr(scheme, "ocp_mx_linear", None)
        if not isinstance(kernel, Mxfp6Sm120LinearKernel):
            continue
        output_features, packed_features = layer.weight.shape
        input_features = packed_features * 4 // 3
        problems.setdefault(
            (output_features, input_features),
            (layer.weight, layer.weight_scale),
        )
    return [
        (output_features, input_features, weight, weight_scale)
        for (output_features, input_features), (weight, weight_scale) in problems.items()
    ]


def _warm_w6a8_problems(
    problems: list[tuple[int, int, torch.Tensor, torch.Tensor]],
    token_sizes: list[int],
    dtype: torch.dtype,
    *,
    stop_after_new_lane: int | None = None,
) -> None:
    import mxfp6

    for output_features, input_features, weight, weight_scale in problems:
        packed_weight = mxfp6.PackedMXFP6Tensor(
            values=weight,
            scales=weight_scale,
            rows=output_features,
            k=input_features,
        )
        for num_tokens in token_sizes:
            x = torch.empty(
                (num_tokens, input_features),
                device=weight.device,
                dtype=dtype,
            ).uniform_(-1.0, 1.0)
            mxfp6.warmup_w6a8(x, packed_weight, out_dtype=dtype, iterations=1)
            if (
                stop_after_new_lane is not None
                and mxfp6.workspace_stats(weight.device)["lanes"]
                > stop_after_new_lane
            ):
                return


@torch.inference_mode()
def warmup_mxfp6_sm120(
    model: torch.nn.Module,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    problems = _collect_w6a8_problems(model)
    sizes = sorted({size for size in token_sizes if size > 0}, reverse=True)
    if not problems or not sizes:
        return
    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.bfloat16

    import mxfp6

    device = problems[0][2].device
    mxfp6.begin_workspace_planning(device)
    _warm_w6a8_problems(problems, sizes, dtype)
    mxfp6.finalize_workspace_planning(device)
    torch.cuda.synchronize(device)


@torch.inference_mode()
def warmup_mxfp6_sm120_stream(
    model: torch.nn.Module,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    problems = _collect_w6a8_problems(model)
    sizes = sorted({size for size in token_sizes if size > 0})
    if not problems or not sizes:
        return
    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.bfloat16

    import mxfp6

    device = problems[0][2].device
    lane_count = mxfp6.workspace_stats(device)["lanes"]
    if lane_count == 0:
        return
    _warm_w6a8_problems(
        problems,
        sizes,
        dtype,
        stop_after_new_lane=lane_count,
    )
    torch.cuda.synchronize(device)
