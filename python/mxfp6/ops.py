from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._loader import load_library
from ._workspace import WorkspaceAPI
from .autotune import (
    _needs_w6a8_autotune,
    _ensure_w6a8_autotuned,
    TUNED_NK as TUNED_NK,
    TUNED_M as TUNED_M,
    is_tuned_shape as is_tuned_shape,
)
from ._validation import (
    _require_cuda_uint8_contiguous,
    _require_sm120,
    _validate_problem,
    _validate_output_dtype,
    _validate_float_activation,
    SCALE_VECTOR_SIZE as SCALE_VECTOR_SIZE,
)
# Preserve the historical ops import paths while implementations live in
# their owning modules. These are direct aliases, with no forwarding calls.
from .quantization import (
    PackedMXFP6Tensor as PackedMXFP6Tensor,
    MXFP8Tensor as MXFP8Tensor,
    pack_fp6 as pack_fp6,
    unpack_fp6 as unpack_fp6,
    pad_fp6 as pad_fp6,
    to_mma_k64_weight as to_mma_k64_weight,
    expand_fp6_to_fp8 as expand_fp6_to_fp8,
    pack_scales as pack_scales,
    unpack_scales as unpack_scales,
    pack_operand as pack_operand,
    unpack_operand as unpack_operand,
    quantize_mxfp8 as quantize_mxfp8,
    quantize_mxfp8_logical as quantize_mxfp8_logical,
    quantize_mxfp8_dual as quantize_mxfp8_dual,
    quantize_mxfp8_dual_out as quantize_mxfp8_dual_out,
    quantize_mxfp6 as quantize_mxfp6,
    silu_and_mul_mxfp8 as silu_and_mul_mxfp8,
    quantize_activation as quantize_activation,
)
from .moe import (
    pack_grouped_scales as pack_grouped_scales,
    silu_and_mul_mxfp8_grouped as silu_and_mul_mxfp8_grouped,
    silu_and_mul_mxfp8_grouped_out as silu_and_mul_mxfp8_grouped_out,
    silu_and_mul_mxfp8_logical as silu_and_mul_mxfp8_logical,
    silu_and_mul_mxfp8_packed_out as silu_and_mul_mxfp8_packed_out,
    route_mxfp8_out as route_mxfp8_out,
    qwen35_route_mxfp8_out as qwen35_route_mxfp8_out,
    moe_reduce_out as moe_reduce_out,
    moe_reduce_tp2_out as moe_reduce_tp2_out,
    qwen35_grouped_w1_silu_mxfp8_out as qwen35_grouped_w1_silu_mxfp8_out,
    grouped_gemm_w6a8_out as grouped_gemm_w6a8_out,
    array_gemm_w6a8_out as array_gemm_w6a8_out,
    array_gemm_w6a8_silu_mxfp8_out as array_gemm_w6a8_silu_mxfp8_out,
    qwen35_router_quant_out as qwen35_router_quant_out,
    qwen35_topk_quant_out as qwen35_topk_quant_out,
    qwen35_topk_quant_route_out as qwen35_topk_quant_route_out,
    qwen35_w1_splitk_silu_mxfp8_out as qwen35_w1_splitk_silu_mxfp8_out,
    qwen35_w2_splitk_reduce_out as qwen35_w2_splitk_reduce_out,
    array_gemm_w6a8_reduce_out as array_gemm_w6a8_reduce_out,
    moe_reduce_array_out as moe_reduce_array_out,
    grouped_gemm_w6a8 as grouped_gemm_w6a8,
)

if TYPE_CHECKING:
    from .autotune import W6A8Config


_workspace = WorkspaceAPI("mxfp6", load_library)
begin_workspace_planning = _workspace.begin_workspace_planning
finalize_workspace_planning = _workspace.finalize_workspace_planning
workspace_stats = _workspace.workspace_stats
workspace_barriers_zero = _workspace.workspace_barriers_zero


def gemm_packed(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    alpha: float = 1.0,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Low-level ``A @ B.T`` for already packed values and scales."""
    out_dtype = _validate_output_dtype(out_dtype)
    _validate_problem(m, n, k)
    for tensor, name in ((a, "a"), (b, "b"), (sfa, "sfa"), (sfb, "sfb")):
        _require_cuda_uint8_contiguous(tensor, name)
    devices = {a.device, b.device, sfa.device, sfb.device}
    if len(devices) != 1:
        raise ValueError("a, b, sfa, and sfb must be on the same CUDA device")
    _require_sm120(a.device)
    load_library()
    return torch.ops.mxfp6.gemm(a, b, sfa, sfb, m, n, k, alpha, out_dtype)


def gemm_w6a8(
    a: MXFP8Tensor,
    b: PackedMXFP6Tensor,
    alpha: float = 1.0,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute native ``MXFP8(A) @ MXFP6(B).T`` on SM120."""
    out_dtype = _validate_output_dtype(out_dtype)
    if not isinstance(a, MXFP8Tensor):
        raise TypeError("a must be an MXFP8Tensor instance")
    if not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("b must be a PackedMXFP6Tensor instance")
    if a.k != b.k:
        raise ValueError(f"a.k and b.k must match; got {a.k} and {b.k}")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    _validate_problem(a.rows, b.rows, a.k)
    for tensor, name in (
        (a.values, "a.values"),
        (a.scales, "a.scales"),
        (b.values, "b.values"),
        (b.scales, "b.scales"),
    ):
        _require_cuda_uint8_contiguous(tensor, name)
    _require_sm120(a.device)
    load_library()
    if _needs_w6a8_autotune(a.device, a.rows, b.rows, a.k, out_dtype):
        _ensure_w6a8_autotuned(
            a.values,
            b.values,
            a.scales,
            b.scales,
            a.rows,
            b.rows,
            a.k,
            out_dtype=out_dtype,
        )
    return torch.ops.mxfp6.gemm_w6a8(
        a.values,
        b.values,
        a.scales,
        b.scales,
        a.rows,
        b.rows,
        a.k,
        alpha,
        out_dtype,
    )


def gemm_from_float(
    a: torch.Tensor,
    b: PackedMXFP6Tensor,
    alpha: float = 1.0,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Quantize FP16/BF16 A to MXFP8 and run the native W6A8 GEMM.

    Weight B remains packed at six bits. Quantization and GEMM are launched on
    the current CUDA stream with programmatic dependent launch enabled.
    """
    out_dtype = _validate_output_dtype(out_dtype)
    rows, k = _validate_float_activation(a)
    if not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("b must be a PackedMXFP6Tensor instance")
    if k != b.k:
        raise ValueError(f"a.shape[1] and b.k must match; got {k} and {b.k}")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    _validate_problem(rows, b.rows, k)
    _require_sm120(a.device)
    load_library()
    if _needs_w6a8_autotune(a.device, rows, b.rows, k, out_dtype):
        # Quantize only once for candidate selection. Quantization is outside
        # every candidate timing; the normal fused call below remains the
        # production 16->8 + GEMM path and uses the installed C++ override.
        tune_values, tune_scales = torch.ops.mxfp6.quantize_mxfp8(a)
        _ensure_w6a8_autotuned(
            tune_values,
            b.values,
            tune_scales,
            b.scales,
            rows,
            b.rows,
            k,
            out_dtype=out_dtype,
        )
    return torch.ops.mxfp6.gemm_from_float(
        a, b.values, b.scales, b.rows, alpha, out_dtype
    )


def autotune_w6a8(
    a: torch.Tensor | MXFP8Tensor,
    b: PackedMXFP6Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
    force: bool = False,
) -> W6A8Config | None:
    """Preselect and persist a native W6A8 config before graph capture.

    This explicit entry is useful during model warmup. Ordinary unknown-shape
    calls invoke the same tuner automatically unless deployment uses
    ``MXFP6_AUTOTUNE=cache_only``. In normal tuning mode, checked-in exact
    shapes keep deterministic dispatch unless ``force=True``; cache-only mode
    also checks those shapes for a persistent override.
    """
    out_dtype = _validate_output_dtype(out_dtype)
    if not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("b must be a PackedMXFP6Tensor instance")
    if isinstance(a, torch.Tensor):
        rows, k = _validate_float_activation(a)
        if k != b.k or a.device != b.device:
            raise ValueError("a and b must have matching K and CUDA device")
        _validate_problem(rows, b.rows, k)
        a = quantize_mxfp8(a)
    elif not isinstance(a, MXFP8Tensor):
        raise TypeError("a must be an FP16/BF16 Tensor or MXFP8Tensor")
    if a.k != b.k or a.device != b.device:
        raise ValueError("a and b must have matching K and CUDA device")
    _validate_problem(a.rows, b.rows, a.k)
    _require_sm120(a.device)
    load_library()
    if not force and is_tuned_shape(a.rows, b.rows, a.k):
        from .autotune import (
            is_autotune_cache_only,
            should_tune_exact_shapes,
        )

        if not should_tune_exact_shapes() and not is_autotune_cache_only():
            return None
    return _ensure_w6a8_autotuned(
        a.values,
        b.values,
        a.scales,
        b.scales,
        a.rows,
        b.rows,
        a.k,
        out_dtype=out_dtype,
        force=force,
    )


def warmup_w6a8(
    a: torch.Tensor | MXFP8Tensor,
    b: PackedMXFP6Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
    iterations: int = 3,
    autotune: bool = True,
    force: bool = False,
) -> W6A8Config | None:
    """Warm the production W6A8 path before graph capture or serving.

    When ``a`` is FP16/BF16, each warmup iteration exercises the fused
    activation-quantization and GEMM launch. ``autotune=True`` first selects
    and caches a config for the requested output dtype. The function
    synchronizes the activation device before returning.
    """
    out_dtype = _validate_output_dtype(out_dtype)
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("iterations must be an integer")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive; got {iterations}")
    if force and not autotune:
        raise ValueError("force=True requires autotune=True")
    if not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("b must be a PackedMXFP6Tensor instance")

    config = None
    if isinstance(a, torch.Tensor):
        rows, k = _validate_float_activation(a)
        if k != b.k or a.device != b.device:
            raise ValueError("a and b must have matching K and CUDA device")
        _validate_problem(rows, b.rows, k)
        _require_sm120(a.device)
        load_library()
        if autotune:
            config = autotune_w6a8(a, b, out_dtype=out_dtype, force=force)
        for _ in range(iterations):
            torch.ops.mxfp6.gemm_from_float(
                a, b.values, b.scales, b.rows, 1.0, out_dtype
            )
        device = a.device
    elif isinstance(a, MXFP8Tensor):
        if a.k != b.k or a.device != b.device:
            raise ValueError("a and b must have matching K and CUDA device")
        _validate_problem(a.rows, b.rows, a.k)
        _require_sm120(a.device)
        load_library()
        if autotune:
            config = autotune_w6a8(a, b, out_dtype=out_dtype, force=force)
        for _ in range(iterations):
            torch.ops.mxfp6.gemm_w6a8(
                a.values,
                b.values,
                a.scales,
                b.scales,
                a.rows,
                b.rows,
                a.k,
                1.0,
                out_dtype,
            )
        device = a.device
    else:
        raise TypeError("a must be an FP16/BF16 Tensor or MXFP8Tensor")

    torch.cuda.synchronize(device)
    return config


def gemm(
    a: torch.Tensor | MXFP8Tensor | PackedMXFP6Tensor,
    b: PackedMXFP6Tensor,
    alpha: float = 1.0,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute ``A @ B.T`` using the native current-repository kernels.

    FP16/BF16 activation tensors are dynamically mapped 16-to-8 and use W6A8
    for every batch size. :class:`MXFP8Tensor` skips that conversion. Packed
    MXFP6 activation inputs retain the legacy W6A6-compatible API.

    """
    out_dtype = _validate_output_dtype(out_dtype)
    if isinstance(a, torch.Tensor):
        if not isinstance(b, PackedMXFP6Tensor):
            raise TypeError(
                "FP16/BF16 activation input requires a PackedMXFP6Tensor weight"
            )
        return gemm_from_float(a, b, alpha, out_dtype=out_dtype)

    if isinstance(a, MXFP8Tensor):
        if not isinstance(b, PackedMXFP6Tensor):
            raise TypeError("MXFP8 activation requires a PackedMXFP6Tensor weight")
        return gemm_w6a8(a, b, alpha, out_dtype=out_dtype)

    if not isinstance(a, PackedMXFP6Tensor):
        raise TypeError(
            "a must be an FP16/BF16 Tensor, MXFP8Tensor, or PackedMXFP6Tensor"
        )

    if not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("b must be a PackedMXFP6Tensor instance")
    if a.k != b.k:
        raise ValueError(f"a.k and b.k must match; got {a.k} and {b.k}")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    return gemm_packed(
        a.values,
        b.values,
        a.scales,
        b.scales,
        a.rows,
        b.rows,
        a.k,
        alpha,
        out_dtype=out_dtype,
    )


def gemm_from_codes(
    a_codes: torch.Tensor,
    b_codes: torch.Tensor,
    sfa_logical: torch.Tensor,
    sfb_logical: torch.Tensor,
    alpha: float = 1.0,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Convenience path that packs both operands before ``A @ B.T``.

    For inference, prepack persistent weights with :func:`pack_operand` and use
    :func:`gemm`; this convenience function intentionally includes conversion
    costs on every call.
    """
    return gemm(
        pack_operand(a_codes, sfa_logical),
        pack_operand(b_codes, sfb_logical),
        alpha,
        out_dtype=out_dtype,
    )


def is_available() -> bool:
    """Return whether the library can load and the current GPU is SM120."""
    if not torch.cuda.is_available():
        return False
    try:
        load_library()
        return torch.cuda.get_device_capability() == (12, 0)
    except (ImportError, OSError, RuntimeError):
        return False


def gemm_from_swiglu(
    a: torch.Tensor,
    b: PackedMXFP6Tensor,
    alpha: float = 1.0,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Fused rounded SwiGLU/MXFP8 producer followed by PDL-enabled W6A8.

    Uses the existing installed static dispatch/overrides and workspace plan.
    Warm the matching [M,K] down-projection before graph capture. This entry
    point does not run a second activation quantizer or perform a collective.
    """
    out_dtype = _validate_output_dtype(out_dtype)
    rows, twice_k = _validate_float_activation(a)
    if not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("b must be a PackedMXFP6Tensor instance")
    if twice_k != 2*b.k or a.device != b.device:
        raise ValueError("SwiGLU input must match weight device and have 2K columns")
    _validate_problem(rows, b.rows, b.k)
    _require_sm120(a.device)
    load_library()
    return torch.ops.mxfp6.gemm_from_swiglu(a,b.values,b.scales,b.rows,alpha,out_dtype)


def gemm_from_gdn(core, gate, norm_weight, weight, weight_scale, eps=1.e-6):
    """Native TP2 GDN gated norm/MXFP8 producer and PDL output projection."""
    from .gdn import gated_norm_mxfp8

    load_library()
    q, scales = gated_norm_mxfp8(core, gate, norm_weight, eps)
    return torch.ops.mxfp6.gemm_w6a8_pdl(q, weight, scales, weight_scale,
        core.shape[0], weight.shape[0], 3072, 1.0, core.dtype)
