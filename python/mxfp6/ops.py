from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

import torch

from ._loader import load_library

if TYPE_CHECKING:
    from .autotune import W6A8Config


SCALE_VECTOR_SIZE = 32
TUNED_NK = frozenset(
    {
        (8192, 5120),
        (5120, 3072),
        (7168, 5120),
        (17408, 5120),
        (5120, 8704),
    }
)
TUNED_M = frozenset((1, 16, 32, 64, 96, 512, 1024, 2048, 4096, 8192))
_AUTOTUNE_STATE_LOCK = threading.Lock()
_AUTOTUNE_READY: set[tuple[int, int, int, int, torch.dtype]] = set()


def _require_cuda_uint8_contiguous(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor; got {tensor.device}")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must have dtype torch.uint8; got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_sm120(device: torch.device) -> None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    if capability != (12, 0):
        raise RuntimeError(
            f"MXFP6 native GEMM requires SM120; device {index} is "
            f"SM{capability[0]}{capability[1]}"
        )


def _workspace_anchor(
    device: torch.device | str | int | None,
) -> torch.Tensor:
    if device is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int) and not isinstance(device, bool):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"workspace device must be CUDA; got {resolved}")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    _require_sm120(resolved)
    load_library()
    return torch.empty(0, device=resolved, dtype=torch.uint8)


def begin_workspace_planning(
    device: torch.device | str | int | None = None,
) -> None:
    """Collect Stream-K workspace layouts selected during subsequent warmup.

    Planning keeps the compatibility path active, including CUTLASS's normal
    one-time-per-call barrier initialization. Call
    :func:`finalize_workspace_planning` after all production shapes and
    autotuned configs have run.
    """
    anchor = _workspace_anchor(device)
    torch.ops.mxfp6.begin_workspace_planning(anchor)


def finalize_workspace_planning(
    device: torch.device | str | int | None = None,
) -> dict[str, int]:
    """Freeze the arena capacity and allocate its current-stream lane."""
    anchor = _workspace_anchor(device)
    stats = torch.ops.mxfp6.finalize_workspace_planning(anchor)
    return {str(key): int(value) for key, value in stats.items()}


def workspace_stats(
    device: torch.device | str | int | None = None,
) -> dict[str, int]:
    """Return planner capacity and launch counters for ``device``."""
    anchor = _workspace_anchor(device)
    stats = torch.ops.mxfp6.workspace_stats(anchor)
    return {str(key): int(value) for key, value in stats.items()}


def workspace_barriers_zero(
    device: torch.device | str | int | None = None,
) -> bool:
    """Synchronize and verify every frozen lane's barrier tail is zero."""
    anchor = _workspace_anchor(device)
    return bool(torch.ops.mxfp6.workspace_barriers_zero(anchor))


def _validate_problem(m: int, n: int, k: int) -> None:
    if m <= 0:
        raise ValueError(f"m must be positive; got {m}")
    if n <= 0 or n % 8:
        raise ValueError(f"n must be a positive multiple of 8; got {n}")
    if k <= 0 or k % 128:
        raise ValueError(f"k must be a positive multiple of 128; got {k}")


def _validate_output_dtype(out_dtype: torch.dtype) -> torch.dtype:
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"out_dtype must be torch.float16 or torch.bfloat16; got {out_dtype}"
        )
    return out_dtype


def _autotune_process_key(
    device: torch.device, m: int, n: int, k: int, out_dtype: torch.dtype
) -> tuple[int, int, int, int, torch.dtype]:
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return (device_index, m, n, k, out_dtype)


def _needs_w6a8_autotune(
    device: torch.device,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> bool:
    from .autotune import (
        can_autotune_now,
        is_autotune_cache_only,
        is_autotune_enabled,
        should_tune_exact_shapes,
    )

    if not is_autotune_enabled() or not can_autotune_now():
        return False
    # In deployment cache-only mode, exact built-in shapes may still have a
    # machine-local cached override. A miss will return to that built-in path
    # without profiling, so it is safe (and useful) to check the cache.
    if (
        is_tuned_shape(m, n, k)
        and not should_tune_exact_shapes()
        and not is_autotune_cache_only()
    ):
        return False
    key = _autotune_process_key(device, m, n, k, out_dtype)
    with _AUTOTUNE_STATE_LOCK:
        return key not in _AUTOTUNE_READY


def _ensure_w6a8_autotuned(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    *,
    out_dtype: torch.dtype,
    force: bool = False,
) -> W6A8Config | None:
    from .autotune import (
        can_autotune_now,
        ensure_w6a8_tuned,
        is_autotune_cache_only,
    )

    cache_only = is_autotune_cache_only()
    config = ensure_w6a8_tuned(
        a, b, sfa, sfb, m, n, k, out_dtype=out_dtype, force=force
    )
    # Cache-only misses deliberately return None. Remember that result for the
    # process so an eager-prefill shape does not repeatedly touch the cache on
    # every request; deployment caches are expected to be immutable in-process.
    if config is not None or (cache_only and can_autotune_now()):
        key = _autotune_process_key(a.device, m, n, k, out_dtype)
        with _AUTOTUNE_STATE_LOCK:
            _AUTOTUNE_READY.add(key)
    return config


@dataclass(frozen=True)
class PackedMXFP6Tensor:
    """An E3M2 matrix and its UE8M0/32 scales in SM120 physical layout."""

    values: torch.Tensor
    scales: torch.Tensor
    rows: int
    k: int
    logical_scales: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.k)


@dataclass(frozen=True)
class MXFP8Tensor:
    """A byte-aligned E4M3 matrix and UE8M0/32 scales for native W6A8."""

    values: torch.Tensor
    scales: torch.Tensor
    rows: int
    k: int

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.k)

    def dequantized_values(self) -> torch.Tensor:
        """Return the E4M3 payload viewed as a float8 matrix (without scales)."""
        return self.values.view(self.rows, self.k).view(torch.float8_e4m3fn)


def _validate_float_activation(input: torch.Tensor) -> tuple[int, int]:
    if input.device.type != "cuda":
        raise ValueError(f"input must be a CUDA tensor; got {input.device}")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"input must have dtype torch.float16 or torch.bfloat16; got {input.dtype}"
        )
    if input.ndim != 2:
        raise ValueError(f"input must have shape [M,K]; got {tuple(input.shape)}")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    rows, k = (int(value) for value in input.shape)
    if rows <= 0 or k <= 0 or k % SCALE_VECTOR_SIZE:
        raise ValueError("M must be positive and K a positive multiple of 32")
    return rows, k


def pack_fp6(codes: torch.Tensor) -> torch.Tensor:
    """Pack CUDA uint8 E3M2 codes, four 6-bit values into three bytes.

    ``codes`` may have any shape, must be contiguous, and must contain a number
    of elements divisible by four. Only each byte's low six bits are used. The
    returned packed representation is one-dimensional.
    """
    _require_cuda_uint8_contiguous(codes, "codes")
    if codes.numel() % 4:
        raise ValueError("codes.numel() must be divisible by four")
    load_library()
    return torch.ops.mxfp6.pack_fp6(codes)


def unpack_fp6(packed: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Unpack a physical E3M2 bitstream to CUDA uint8 codes ``[rows, k]``."""
    _require_cuda_uint8_contiguous(packed, "packed")
    if rows <= 0 or k <= 0 or k % 4:
        raise ValueError("rows must be positive and k must be divisible by four")
    load_library()
    return torch.ops.mxfp6.unpack_fp6(packed, rows, k)


def pad_fp6(packed: torch.Tensor) -> torch.Tensor:
    """Pad packed FP6 12-byte segments to aligned 16-byte MMA segments."""
    _require_cuda_uint8_contiguous(packed, "packed")
    if packed.ndim < 1 or packed.shape[-1] <= 0 or packed.shape[-1] % 12:
        raise ValueError("packed last dimension must be a positive multiple of 12")
    load_library()
    return torch.ops.mxfp6.pad_fp6(packed)


def to_mma_k64_weight(weight: torch.Tensor) -> torch.Tensor:
    """Transpose packed ``[E,N,64]`` weights for the compact SM120 TMA path."""
    _require_cuda_uint8_contiguous(weight, "weight")
    if weight.ndim != 3 or weight.shape[2] != 48:
        raise ValueError("weight must contain packed [E,N,64] FP6 values")
    experts, rows, _ = weight.shape
    codes = unpack_fp6(weight, experts * rows, 64)
    transposed = codes.view(experts, rows, 64).transpose(1, 2).contiguous()
    return pack_fp6(transposed.flatten(0, 1)).view_as(weight)


def expand_fp6_to_fp8(packed: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Losslessly expand packed E3M2 to a ``torch.float8_e4m3fn`` matrix."""
    _require_cuda_uint8_contiguous(packed, "packed")
    if rows <= 0 or k <= 0 or k % 16:
        raise ValueError("rows must be positive and k must be divisible by 16")
    load_library()
    output = torch.ops.mxfp6.expand_fp6_to_fp8(packed, rows, k)
    return output.view(torch.float8_e4m3fn)


def pack_scales(logical: torch.Tensor) -> torch.Tensor:
    """Reorder logical UE8M0 scales to the SM120 CUTLASS physical layout.

    ``logical`` is a contiguous CUDA uint8 tensor of shape ``[rows, k / 32]``.
    Byte values are encoded UE8M0 exponents (``0x7f`` represents 1.0).
    """
    _require_cuda_uint8_contiguous(logical, "logical scales")
    if logical.ndim != 2 or logical.shape[0] <= 0 or logical.shape[1] <= 0:
        raise ValueError("logical scales must have shape [rows, k / 32]")
    rows, k_blocks = logical.shape
    load_library()
    return torch.ops.mxfp6.pack_scales(
        logical, int(rows), int(k_blocks * SCALE_VECTOR_SIZE)
    )


def unpack_scales(packed: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Convert packed SM120 scales back to logical ``[rows, k / 32]``."""
    _require_cuda_uint8_contiguous(packed, "packed scales")
    if rows <= 0 or k <= 0 or k % SCALE_VECTOR_SIZE:
        raise ValueError("rows must be positive and k must be divisible by 32")
    load_library()
    return torch.ops.mxfp6.unpack_scales(packed, rows, k)


def pack_grouped_scales(
    logical: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack logical scales independently for each contiguous expert group.

    ``logical`` has shape ``[M, K / 32]`` after MoE routing and
    ``expert_offsets`` contains ``E + 1`` int64 row offsets. The returned
    scale offsets keep every expert on a 128-row SM120 scale-atom boundary.
    """
    _require_cuda_uint8_contiguous(logical, "logical scales")
    if logical.ndim != 2 or logical.shape[0] <= 0 or logical.shape[1] <= 0:
        raise ValueError("logical scales must have shape [M, K / 32]")
    if (
        expert_offsets.device != logical.device
        or expert_offsets.dtype != torch.int64
        or expert_offsets.ndim != 1
        or expert_offsets.numel() < 2
        or not expert_offsets.is_contiguous()
    ):
        raise ValueError(
            "expert_offsets must be a contiguous CUDA int64 tensor with E+1 entries"
        )
    load_library()
    return torch.ops.mxfp6.pack_grouped_scales(logical, expert_offsets)


def silu_and_mul_mxfp8_grouped(
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SiLU-and-mul with routed MXFP8 quantization and scale packing."""
    if (
        input.device.type != "cuda"
        or input.dtype not in (torch.float16, torch.bfloat16)
        or input.ndim != 2
        or not input.is_contiguous()
        or input.shape[1] % 64
    ):
        raise ValueError(
            "input must be contiguous CUDA float16/bfloat16 [M,2K] "
            "with K divisible by 32"
        )
    if (
        expert_offsets.device != input.device
        or expert_offsets.dtype != torch.int64
        or expert_offsets.ndim != 1
        or expert_offsets.numel() < 2
        or not expert_offsets.is_contiguous()
    ):
        raise ValueError(
            "expert_offsets must be a contiguous CUDA int64 tensor with E+1 entries"
        )
    if scale_offsets is not None and (
        scale_offsets.device != input.device
        or scale_offsets.dtype != torch.int64
        or scale_offsets.ndim != 1
        or scale_offsets.shape != expert_offsets.shape
        or not scale_offsets.is_contiguous()
    ):
        raise ValueError(
            "scale_offsets must be a contiguous CUDA int64 tensor with E+1 entries"
        )
    _require_sm120(input.device)
    load_library()
    values, scales, scale_offsets = torch.ops.mxfp6.silu_and_mul_mxfp8_grouped(
        input, expert_offsets, scale_offsets
    )
    return values.view(torch.float8_e4m3fn), scales, scale_offsets


def silu_and_mul_mxfp8_grouped_out(
    output: torch.Tensor,
    scales: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    use_pdl: bool = False,
    grouped_output: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    weight_scales: torch.Tensor | None = None,
) -> None:
    """Write grouped activation, optionally launching its W2 GEMM."""
    if (
        input.device.type != "cuda"
        or input.dtype not in (torch.float16, torch.bfloat16)
        or input.ndim != 2
        or not input.is_contiguous()
        or input.shape[1] % 64
    ):
        raise ValueError(
            "input must be contiguous CUDA float16/bfloat16 [M,2K] "
            "with K divisible by 32"
        )
    rows = input.shape[0]
    output_features = input.shape[1] // 2
    if (
        output.device != input.device
        or output.dtype != torch.uint8
        or output.shape != (rows, output_features)
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous CUDA uint8 "
            f"{(rows, output_features)}"
        )
    for tensor, name in (
        (expert_offsets, "expert_offsets"),
        (scale_offsets, "scale_offsets"),
    ):
        if (
            tensor.device != input.device
            or tensor.dtype != torch.int64
            or tensor.ndim != 1
            or tensor.numel() < 2
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA int64 [E+1]"
            )
    if scale_offsets.shape != expert_offsets.shape:
        raise ValueError("scale_offsets must match expert_offsets")
    experts = expert_offsets.numel() - 1
    packed_rows = rows + min(rows, experts) * 127
    packed_groups = (
        (output_features // SCALE_VECTOR_SIZE + 3) // 4 * 4
    )
    if (
        scales.device != input.device
        or scales.dtype != torch.uint8
        or scales.numel() < packed_rows * packed_groups
        or not scales.is_contiguous()
    ):
        raise ValueError(
            "scales must be contiguous CUDA uint8 with at least "
            f"{packed_rows * packed_groups} values"
        )
    fused_grouped = grouped_output is not None
    if fused_grouped != (weight is not None) or fused_grouped != (
        weight_scales is not None
    ):
        raise ValueError(
            "grouped_output, weight, and weight_scales must be provided together"
        )
    _require_sm120(input.device)
    load_library()
    torch.ops.mxfp6.silu_and_mul_mxfp8_grouped_out(
        output,
        scales,
        input,
        expert_offsets,
        scale_offsets,
        use_pdl,
        grouped_output,
        weight,
        weight_scales,
    )


def silu_and_mul_mxfp8_logical(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiLU-and-mul with row-major logical MXFP8 quantization."""
    if (
        input.device.type != "cuda"
        or input.dtype not in (torch.float16, torch.bfloat16)
        or input.ndim != 2
        or not input.is_contiguous()
        or input.shape[1] % 64
    ):
        raise ValueError(
            "input must be contiguous CUDA float16/bfloat16 [M,2K] "
            "with K divisible by 32"
        )
    _require_sm120(input.device)
    load_library()
    values, scales = torch.ops.mxfp6.silu_and_mul_mxfp8_logical(input)
    return values.view(torch.float8_e4m3fn), scales


def silu_and_mul_mxfp8_packed_out(
    output: torch.Tensor,
    packed_scales: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """Fuse SiLU-and-mul with MXFP8 quantization into dense packed scales."""
    if (
        input.device.type != "cuda"
        or input.dtype not in (torch.float16, torch.bfloat16)
        or input.ndim != 2
        or not input.is_contiguous()
        or input.shape[1] % 64
    ):
        raise ValueError(
            "input must be contiguous CUDA float16/bfloat16 [M,2K] "
            "with K divisible by 32"
        )
    rows = input.shape[0]
    output_features = input.shape[1] // 2
    if (
        output.device != input.device
        or output.dtype != torch.uint8
        or output.shape != (rows, output_features)
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous CUDA uint8 "
            f"{(rows, output_features)}"
        )
    packed_groups = (
        (output_features // SCALE_VECTOR_SIZE + 3) // 4 * 4
    )
    packed_size = ((rows + 127) // 128 * 128) * packed_groups
    if (
        packed_scales.device != input.device
        or packed_scales.dtype != torch.uint8
        or packed_scales.numel() < packed_size
        or not packed_scales.is_contiguous()
    ):
        raise ValueError(
            "packed_scales must be contiguous CUDA uint8 with at least "
            f"{packed_size} values"
        )
    _require_sm120(input.device)
    load_library()
    torch.ops.mxfp6.silu_and_mul_mxfp8_packed_out(
        output,
        packed_scales,
        input,
    )


def pack_operand(
    codes: torch.Tensor, logical_scales: torch.Tensor
) -> PackedMXFP6Tensor:
    """Pack a logical ``[rows, k]`` E3M2 operand and its UE8M0 scales."""
    _require_cuda_uint8_contiguous(codes, "codes")
    _require_cuda_uint8_contiguous(logical_scales, "logical_scales")
    if codes.ndim != 2:
        raise ValueError("codes must have shape [rows, k]")
    rows, k = (int(value) for value in codes.shape)
    expected_scale_shape = (rows, k // SCALE_VECTOR_SIZE)
    if k % SCALE_VECTOR_SIZE or tuple(logical_scales.shape) != expected_scale_shape:
        raise ValueError(
            f"logical_scales must have shape {expected_scale_shape}; "
            f"got {tuple(logical_scales.shape)}"
        )
    if codes.device != logical_scales.device:
        raise ValueError("codes and logical_scales must be on the same device")
    return PackedMXFP6Tensor(
        pack_fp6(codes),
        pack_scales(logical_scales),
        rows,
        k,
        logical_scales,
    )


def unpack_operand(
    operand: PackedMXFP6Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logical E3M2 codes and logical UE8M0 scales for an operand."""
    logical_scales = operand.logical_scales
    if logical_scales is None:
        logical_scales = unpack_scales(operand.scales, operand.rows, operand.k)
    return (
        unpack_fp6(operand.values, operand.rows, operand.k),
        logical_scales,
    )


def quantize_mxfp8(input: torch.Tensor) -> MXFP8Tensor:
    """Dynamically map a contiguous FP16/BF16 ``[M,K]`` activation to MXFP8.

    Quantization uses one power-of-two UE8M0 scale per 32 consecutive K
    values. The returned value bytes and scales are already in the layouts
    consumed by the native SM120 W6A8 kernels.
    """
    rows, k = _validate_float_activation(input)
    _require_sm120(input.device)
    load_library()
    values, scales = torch.ops.mxfp6.quantize_mxfp8(input)
    return MXFP8Tensor(values, scales, rows, k)


def quantize_mxfp8_logical(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP8 with row-major ``[M,K/32]`` UE8M0 scales."""
    _validate_float_activation(input)
    _require_sm120(input.device)
    load_library()
    values, scales = torch.ops.mxfp6.quantize_mxfp8_logical(input)
    return values.view(torch.float8_e4m3fn), scales


def quantize_mxfp8_dual(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize once and emit both routed and dense CUTLASS scale layouts."""
    _validate_float_activation(input)
    _require_sm120(input.device)
    load_library()
    values, logical_scales, packed_scales = (
        torch.ops.mxfp6.quantize_mxfp8_dual(input)
    )
    return (
        values.view(torch.float8_e4m3fn),
        logical_scales,
        packed_scales,
    )


def quantize_mxfp8_dual_out(
    output: torch.Tensor,
    logical_scales: torch.Tensor,
    packed_scales: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """Write one MXFP8 value tensor and both scale layouts in one kernel."""
    rows, k = _validate_float_activation(input)
    _require_sm120(input.device)
    expected = (
        (output, torch.uint8, (rows, k), "output"),
        (
            logical_scales,
            torch.uint8,
            (rows, k // SCALE_VECTOR_SIZE),
            "logical_scales",
        ),
    )
    for tensor, dtype, shape, name in expected:
        if (
            tensor.device != input.device
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA {dtype} {shape}"
            )
    packed_groups = (k // SCALE_VECTOR_SIZE + 3) // 4 * 4
    packed_size = ((rows + 127) // 128 * 128) * packed_groups
    if (
        packed_scales.device != input.device
        or packed_scales.dtype != torch.uint8
        or packed_scales.numel() < packed_size
        or not packed_scales.is_contiguous()
    ):
        raise ValueError(
            "packed_scales must be contiguous CUDA uint8 with at least "
            f"{packed_size} values"
        )
    load_library()
    torch.ops.mxfp6.quantize_mxfp8_dual_out(
        output,
        logical_scales,
        packed_scales,
        input,
    )


def route_mxfp8_out(
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_experts: int,
    permuted_activation: torch.Tensor,
    include_shared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route MXFP8 rows and optionally append an always-hit shared group."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.ndim != 2
        or not activation.is_contiguous()
        or activation.shape[1] % SCALE_VECTOR_SIZE
    ):
        raise ValueError(
            "activation must be contiguous CUDA float8_e4m3fn/uint8 "
            "[M,K] with K divisible by 32"
        )
    expected_scale_shape = (
        activation.shape[0],
        activation.shape[1] // SCALE_VECTOR_SIZE,
    )
    if (
        logical_scales.device != activation.device
        or logical_scales.dtype != torch.uint8
        or logical_scales.shape != expected_scale_shape
        or not logical_scales.is_contiguous()
    ):
        raise ValueError(
            f"logical_scales must be contiguous CUDA uint8 {expected_scale_shape}"
        )
    if (
        topk_ids.device != activation.device
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_ids.ndim != 2
        or topk_ids.shape[0] != activation.shape[0]
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("topk_ids must be contiguous CUDA int32/int64 [M,topk]")
    routed_shape = (
        activation.shape[0] * (topk_ids.shape[1] + int(include_shared)),
        activation.shape[1],
    )
    if (
        permuted_activation.device != activation.device
        or permuted_activation.dtype != activation.dtype
        or permuted_activation.shape != routed_shape
        or not permuted_activation.is_contiguous()
    ):
        raise ValueError(
            f"permuted_activation must be contiguous {routed_shape} "
            "with the activation dtype and device"
        )
    minimum_experts = 2 if include_shared else 1
    if not minimum_experts <= local_experts <= 257:
        raise ValueError(
            "local_experts must be in [1, 257] and include one trailing "
            "shared expert when include_shared is true"
        )
    if expert_map is not None and (
        expert_map.device != activation.device
        or expert_map.dtype != torch.int32
        or expert_map.ndim != 1
        or not expert_map.is_contiguous()
    ):
        raise ValueError("expert_map must be a contiguous CUDA int32 vector")
    _require_sm120(activation.device)
    load_library()
    return torch.ops.mxfp6.route_mxfp8_out(
        activation,
        logical_scales,
        topk_ids,
        expert_map,
        local_experts,
        permuted_activation,
        include_shared,
    )


def qwen35_route_mxfp8_out(
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    permuted_activation: torch.Tensor,
    packed_scales: torch.Tensor,
    expert_cursors: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    inverse_permutation: torch.Tensor,
    include_shared: bool = False,
) -> None:
    """Route Qwen3.5 MXFP8 rows into preallocated grouped buffers."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.ndim != 2
        or activation.shape[0] < 1
        or activation.shape[1] != 2048
        or not activation.is_contiguous()
    ):
        raise ValueError(
            "activation must be contiguous CUDA MXFP8 [B,2048] with B > 0"
        )
    _require_sm120(activation.device)
    batch = activation.shape[0]
    routes = batch * 8
    shared_rows = batch if include_shared else 0
    total_rows = routes + shared_rows
    experts = 256 + int(include_shared)
    expected = (
        (logical_scales, torch.uint8, (batch, 64), "logical_scales"),
        (topk_ids, torch.int32, (batch, 8), "topk_ids"),
        (
            permuted_activation,
            activation.dtype,
            (total_rows, 2048),
            "permuted_activation",
        ),
        (expert_cursors, torch.int32, (experts,), "expert_cursors"),
        (
            expert_offsets,
            torch.int64,
            (experts + 1,),
            "expert_offsets",
        ),
        (
            scale_offsets,
            torch.int64,
            (experts + 1,),
            "scale_offsets",
        ),
        (
            inverse_permutation,
            torch.int32,
            (routes,),
            "inverse_permutation",
        ),
    )
    for tensor, dtype, shape, name in expected:
        if (
            tensor.device != activation.device
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA {dtype} {tuple(shape)}"
            )
    active_experts = min(routes, 256) + int(include_shared)
    packed_size = (total_rows + active_experts * 127) * 64
    if (
        packed_scales.device != activation.device
        or packed_scales.dtype != torch.uint8
        or packed_scales.numel() < packed_size
        or not packed_scales.is_contiguous()
    ):
        raise ValueError(
            "packed_scales must be contiguous CUDA uint8 with at least "
            f"{packed_size} values"
        )
    load_library()
    torch.ops.mxfp6.qwen35_route_mxfp8_out(
        activation,
        logical_scales,
        topk_ids,
        permuted_activation,
        packed_scales,
        expert_cursors,
        expert_offsets,
        scale_offsets,
        inverse_permutation,
        include_shared,
    )


def moe_reduce_out(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
    shared_output: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    use_pdl: bool = False,
) -> None:
    """Reduce routed rows and optionally gate/add a shared-expert output."""
    if (
        routed_output.device.type != "cuda"
        or routed_output.dtype not in (torch.float16, torch.bfloat16)
        or routed_output.ndim != 2
        or not routed_output.is_contiguous()
    ):
        raise ValueError(
            "routed_output must be contiguous CUDA float16/bfloat16 [routes,H]"
        )
    if (
        output.device != routed_output.device
        or output.dtype != routed_output.dtype
        or output.ndim != 2
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous CUDA [tokens,H] with routed_output's dtype"
        )
    if (
        topk_weights.device != routed_output.device
        or topk_weights.dtype != torch.float32
        or topk_weights.ndim != 2
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("topk_weights must be contiguous CUDA float32 [tokens,topk]")
    if (
        inverse_permutation.device != routed_output.device
        or inverse_permutation.dtype != torch.int32
        or inverse_permutation.ndim != 1
        or not inverse_permutation.is_contiguous()
    ):
        raise ValueError("inverse_permutation must be a contiguous CUDA int32 vector")
    if shared_output is not None and (
        shared_output.device != routed_output.device
        or shared_output.dtype != routed_output.dtype
        or shared_output.shape != output.shape
        or not shared_output.is_contiguous()
    ):
        raise ValueError(
            "shared_output must be contiguous and match output's device, "
            "dtype, and shape"
        )
    if shared_gate is not None and (
        shared_output is None
        or shared_gate.device != routed_output.device
        or shared_gate.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not shared_gate.is_contiguous()
        or (
            shared_gate.numel() != output.shape[0]
            and (shared_gate.ndim != 2 or shared_gate.shape[0] != output.shape[0])
        )
    ):
        raise ValueError(
            "shared_gate requires shared_output and one contiguous floating "
            "value or row per token"
        )
    load_library()
    torch.ops.mxfp6.moe_reduce_out(
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
        shared_output,
        shared_gate,
        use_pdl,
    )


def moe_reduce_tp2_out(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
    shared_output: torch.Tensor,
    shared_gate: torch.Tensor,
    signal_ptrs: list[int],
    buffer_ptrs: list[int],
    rank: int,
) -> None:
    """Fuse BF16 MoE reduction with vLLM's two-rank custom all-reduce."""
    if (
        routed_output.device.type != "cuda"
        or routed_output.dtype != torch.bfloat16
        or routed_output.ndim != 2
        or not routed_output.is_contiguous()
    ):
        raise ValueError(
            "routed_output must be contiguous CUDA bfloat16 [routes,H]"
        )
    if (
        output.device != routed_output.device
        or output.dtype != torch.bfloat16
        or output.ndim != 2
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous CUDA bfloat16 [tokens,H]")
    if (
        topk_weights.device != routed_output.device
        or topk_weights.dtype != torch.float32
        or topk_weights.ndim != 2
        or topk_weights.shape[1] != 8
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("topk_weights must be contiguous CUDA float32 [tokens,8]")
    if (
        inverse_permutation.device != routed_output.device
        or inverse_permutation.dtype != torch.int32
        or inverse_permutation.ndim != 1
        or not inverse_permutation.is_contiguous()
    ):
        raise ValueError("inverse_permutation must be contiguous CUDA int32")
    if (
        shared_output.device != routed_output.device
        or shared_output.dtype != torch.bfloat16
        or shared_output.shape != output.shape
        or not shared_output.is_contiguous()
    ):
        raise ValueError("shared_output must match output")
    if (
        shared_gate.device != routed_output.device
        or shared_gate.dtype
        not in (torch.float16, torch.bfloat16, torch.float32)
        or not shared_gate.is_contiguous()
        or (
            shared_gate.numel() != output.shape[0]
            and (shared_gate.ndim != 2 or shared_gate.shape[0] != output.shape[0])
        )
    ):
        raise ValueError("shared_gate must contain one value or row per token")
    if len(signal_ptrs) != 2 or len(buffer_ptrs) != 2 or rank not in (0, 1):
        raise ValueError("TP2 fused reduce requires two pointers and rank 0 or 1")
    _require_sm120(routed_output.device)
    load_library()
    torch.ops.mxfp6.moe_reduce_tp2_out(
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
        shared_output,
        shared_gate,
        signal_ptrs,
        buffer_ptrs,
        rank,
    )


def quantize_mxfp6(input: torch.Tensor) -> PackedMXFP6Tensor:
    """Dynamically map a contiguous FP16/BF16 ``[M,K]`` matrix to MXFP6."""
    rows, k = _validate_float_activation(input)
    _require_sm120(input.device)
    load_library()
    values, scales = torch.ops.mxfp6.quantize_mxfp6(input)
    return PackedMXFP6Tensor(values, scales, rows, k)


# The production activation path is W6A8; keep this intent-revealing alias for
# model integrations that should not have to name the storage format.
quantize_activation = quantize_mxfp8


def qwen35_grouped_w1_silu_mxfp8_out(
    output: torch.Tensor,
    output_scales: torch.Tensor,
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    use_pdl: bool = False,
    source_tokens: torch.Tensor | None = None,
) -> None:
    """Run the SM120 Qwen3.5 grouped W1 with fused SiLU/MXFP8 output."""
    _require_sm120(activation.device)
    if source_tokens is not None and (
        source_tokens.device != activation.device
        or source_tokens.dtype != torch.int32
        or source_tokens.ndim != 1
        or source_tokens.numel() != output.shape[0]
        or not source_tokens.is_contiguous()
    ):
        raise ValueError(
            "source_tokens must be contiguous CUDA int32 [routes]"
        )
    load_library()
    torch.ops.mxfp6.qwen35_grouped_w1_silu_mxfp8_out(
        output,
        output_scales,
        activation,
        activation_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
        use_pdl,
        source_tokens,
    )


def grouped_gemm_w6a8_out(
    output: torch.Tensor,
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    tile_n: int = 0,
    use_pdl: bool = False,
) -> None:
    """Run native SM120 grouped ``A8 @ W6.T`` into a preallocated output."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.ndim != 2
        or not activation.is_contiguous()
    ):
        raise ValueError("activation must be contiguous CUDA float8_e4m3fn/uint8 [M,K]")
    _require_sm120(activation.device)
    for tensor, name in (
        (activation_scales, "activation_scales"),
        (weight, "weight"),
        (weight_scales, "weight_scales"),
    ):
        _require_cuda_uint8_contiguous(tensor, name)
        if tensor.device != activation.device:
            raise ValueError(f"{name} must be on {activation.device}")
    if weight.ndim != 3:
        raise ValueError("weight must have shape [E,N,packed_K]")
    for tensor, name in (
        (expert_offsets, "expert_offsets"),
        (scale_offsets, "scale_offsets"),
    ):
        if (
            tensor.device != activation.device
            or tensor.dtype != torch.int64
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int64 tensor")
    if (
        output.device != activation.device
        or output.dtype not in (torch.float16, torch.bfloat16)
        or output.ndim != 2
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous CUDA float16/bfloat16 [M,N]")
    load_library()
    torch.ops.mxfp6.grouped_gemm_w6a8_out(
        output,
        activation,
        activation_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
        tile_n,
        use_pdl,
    )


def array_gemm_w6a8_out(
    output: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    include_shared: bool = False,
) -> None:
    """Run equal-shape route GEMMs, optionally appending a shared route."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.ndim != 2
        or not activation.is_contiguous()
    ):
        raise ValueError("activation must be contiguous CUDA float8_e4m3fn/uint8 [M,K]")
    _require_sm120(activation.device)
    for tensor, name in (
        (logical_scales, "logical_scales"),
        (weight, "weight"),
        (weight_scales, "weight_scales"),
    ):
        _require_cuda_uint8_contiguous(tensor, name)
        if tensor.device != activation.device:
            raise ValueError(f"{name} must be on {activation.device}")
    if (
        topk_ids.device != activation.device
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_ids.ndim != 2
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("topk_ids must be contiguous CUDA int32/int64 [tokens,topk]")
    if (
        output.device != activation.device
        or output.dtype not in (torch.float16, torch.bfloat16)
        or output.ndim != 2
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous CUDA float16/bfloat16 [routes,N]")
    load_library()
    torch.ops.mxfp6.array_gemm_w6a8_out(
        output,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        include_shared,
    )


def array_gemm_w6a8_silu_mxfp8_out(
    output: torch.Tensor,
    output_scales: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    include_shared: bool = False,
) -> None:
    """Fuse route W6A8 gate/up GEMM, SiLU multiply, and MXFP8 quantization."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.ndim != 2
        or not activation.is_contiguous()
    ):
        raise ValueError("activation must be contiguous CUDA float8_e4m3fn/uint8 [M,K]")
    _require_sm120(activation.device)
    for tensor, name in (
        (output, "output"),
        (output_scales, "output_scales"),
        (logical_scales, "logical_scales"),
        (weight, "weight"),
        (weight_scales, "weight_scales"),
    ):
        _require_cuda_uint8_contiguous(tensor, name)
        if tensor.device != activation.device:
            raise ValueError(f"{name} must be on {activation.device}")
    if (
        topk_ids.device != activation.device
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_ids.ndim != 2
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("topk_ids must be contiguous CUDA int32/int64 [tokens,topk]")
    load_library()
    torch.ops.mxfp6.array_gemm_w6a8_silu_mxfp8_out(
        output,
        output_scales,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        include_shared,
    )


def qwen35_router_quant_out(
    quantized: torch.Tensor,
    logical_scales: torch.Tensor,
    routed_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate: torch.Tensor,
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    renormalize: bool = False,
    shared_gate_weight: torch.Tensor | None = None,
) -> None:
    """Fuse the specialized Qwen3.5 router, top-k, and MXFP8 quantization."""
    if (
        hidden.device.type != "cuda"
        or hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or hidden.shape[0] not in (1, 2, 4, 8, 16)
        or hidden.shape[1] != 2048
        or not hidden.is_contiguous()
    ):
        raise ValueError(
            "hidden must be contiguous CUDA bfloat16 [B,2048], "
            "with B in {1,2,4,8,16}"
        )
    _require_sm120(hidden.device)
    gate_rows = 256 if shared_gate_weight is not None else 257
    if (
        gate_weight.device != hidden.device
        or gate_weight.dtype != torch.bfloat16
        or gate_weight.shape != (gate_rows, 2048)
        or not gate_weight.is_contiguous()
    ):
        raise ValueError(
            f"gate_weight must be contiguous CUDA bfloat16 [{gate_rows},2048]"
        )
    if shared_gate_weight is not None and (
        shared_gate_weight.device != hidden.device
        or shared_gate_weight.dtype != torch.bfloat16
        or shared_gate_weight.shape != (1, 2048)
        or not shared_gate_weight.is_contiguous()
    ):
        raise ValueError(
            "shared_gate_weight must be contiguous CUDA bfloat16 [1,2048]"
        )
    batch = hidden.shape[0]
    for tensor, name, shape in (
        (quantized, "quantized", hidden.shape),
        (logical_scales, "logical_scales", (batch, 64)),
    ):
        if (
            tensor.device != hidden.device
            or tensor.dtype != torch.uint8
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA uint8 {tuple(shape)}")
    if (
        routed_logits.device != hidden.device
        or routed_logits.dtype != torch.bfloat16
        or routed_logits.shape != (batch, 256)
        or not routed_logits.is_contiguous()
    ):
        raise ValueError("routed_logits must be contiguous CUDA bfloat16 [B,256]")
    if (
        topk_weights.device != hidden.device
        or topk_weights.dtype != torch.float32
        or topk_weights.shape != (batch, 8)
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("topk_weights must be contiguous CUDA float32 [B,8]")
    if (
        topk_ids.device != hidden.device
        or topk_ids.dtype != torch.int32
        or topk_ids.shape != (batch, 8)
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("topk_ids must be contiguous CUDA int32 [B,8]")
    if (
        shared_gate.device != hidden.device
        or shared_gate.dtype != torch.bfloat16
        or shared_gate.numel() != batch
        or not shared_gate.is_contiguous()
    ):
        raise ValueError("shared_gate must be contiguous CUDA bfloat16 with B values")
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be a bool")
    load_library()
    torch.ops.mxfp6.qwen35_router_quant_out(
        quantized,
        logical_scales,
        routed_logits,
        topk_weights,
        topk_ids,
        shared_gate,
        hidden,
        gate_weight,
        renormalize,
        shared_gate_weight,
    )


def qwen35_topk_quant_out(
    quantized: torch.Tensor,
    logical_scales: torch.Tensor,
    packed_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    hidden: torch.Tensor,
    routed_logits: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """Fuse Qwen3.5 top-k routing with dual-layout MXFP8 quantization."""
    if (
        hidden.device.type != "cuda"
        or hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or hidden.shape[0] < 1
        or hidden.shape[1] != 2048
        or not hidden.is_contiguous()
    ):
        raise ValueError(
            "hidden must be contiguous CUDA bfloat16 [B,2048] with B > 0"
        )
    _require_sm120(hidden.device)
    batch = hidden.shape[0]
    expected = (
        (quantized, torch.uint8, hidden.shape, "quantized"),
        (logical_scales, torch.uint8, (batch, 64), "logical_scales"),
        (topk_weights, torch.float32, (batch, 8), "topk_weights"),
        (topk_ids, torch.int32, (batch, 8), "topk_ids"),
        (
            routed_logits,
            torch.bfloat16,
            (batch, 256),
            "routed_logits",
        ),
    )
    for tensor, dtype, shape, name in expected:
        if (
            tensor.device != hidden.device
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA {dtype} {tuple(shape)}"
            )
    packed_size = ((batch + 127) // 128 * 128) * 64
    if (
        packed_scales.device != hidden.device
        or packed_scales.dtype != torch.uint8
        or packed_scales.numel() < packed_size
        or not packed_scales.is_contiguous()
    ):
        raise ValueError(
            "packed_scales must be contiguous CUDA uint8 with at least "
            f"{packed_size} values"
        )
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be a bool")
    load_library()
    torch.ops.mxfp6.qwen35_topk_quant_out(
        quantized,
        logical_scales,
        packed_scales,
        topk_weights,
        topk_ids,
        hidden,
        routed_logits,
        renormalize,
    )


def qwen35_topk_quant_route_out(
    quantized: torch.Tensor,
    logical_scales: torch.Tensor,
    dense_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    permuted_activation: torch.Tensor,
    grouped_scales: torch.Tensor,
    expert_cursors: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    inverse_permutation: torch.Tensor,
    hidden: torch.Tensor,
    routed_logits: torch.Tensor,
    grouped_output: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    weight_scales: torch.Tensor | None = None,
    source_tokens: torch.Tensor | None = None,
    renormalize: bool = False,
) -> None:
    """Fuse top-k routing, optionally launching its grouped W1 GEMM."""
    if (
        hidden.device.type != "cuda"
        or hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or not 1 <= hidden.shape[0] <= 256
        or hidden.shape[1] != 2048
        or not hidden.is_contiguous()
    ):
        raise ValueError(
            "hidden must be contiguous CUDA bfloat16 [B,2048], "
            "with 1 <= B <= 256"
        )
    _require_sm120(hidden.device)
    batch = hidden.shape[0]
    routes = batch * 8
    expected = (
        (quantized, torch.uint8, (batch, 2048), "quantized"),
        (logical_scales, torch.uint8, (batch, 64), "logical_scales"),
        (topk_weights, torch.float32, (batch, 8), "topk_weights"),
        (topk_ids, torch.int32, (batch, 8), "topk_ids"),
        (
            routed_logits,
            torch.bfloat16,
            (batch, 256),
            "routed_logits",
        ),
        (expert_cursors, torch.int32, (256,), "expert_cursors"),
        (expert_offsets, torch.int64, (257,), "expert_offsets"),
        (scale_offsets, torch.int64, (257,), "scale_offsets"),
        (
            inverse_permutation,
            torch.int32,
            (routes,),
            "inverse_permutation",
        ),
    )
    for tensor, dtype, shape, name in expected:
        if (
            tensor.device != hidden.device
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA {dtype} {tuple(shape)}"
            )
    if (
        permuted_activation.device != hidden.device
        or permuted_activation.dtype
        not in (torch.uint8, torch.float8_e4m3fn)
        or permuted_activation.shape != (routes, 2048)
        or not permuted_activation.is_contiguous()
    ):
        raise ValueError(
            "permuted_activation must be contiguous CUDA MXFP8 "
            f"{(routes, 2048)}"
        )
    dense_size = ((batch + 127) // 128 * 128) * 64
    grouped_size = (routes + min(routes, 256) * 127) * 64
    for tensor, size, name in (
        (dense_scales, dense_size, "dense_scales"),
        (grouped_scales, grouped_size, "grouped_scales"),
    ):
        if (
            tensor.device != hidden.device
            or tensor.dtype != torch.uint8
            or tensor.numel() < size
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA uint8 with at least "
                f"{size} values"
            )
    fused_grouped = grouped_output is not None
    if fused_grouped != (weight is not None) or fused_grouped != (
        weight_scales is not None
    ):
        raise ValueError(
            "grouped_output, weight, and weight_scales must be provided together"
        )
    if source_tokens is not None and (
        source_tokens.device != hidden.device
        or source_tokens.dtype != torch.int32
        or source_tokens.shape != (routes,)
        or not source_tokens.is_contiguous()
    ):
        raise ValueError(
            f"source_tokens must be contiguous CUDA int32 {(routes,)}"
        )
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be a bool")
    load_library()
    torch.ops.mxfp6.qwen35_topk_quant_route_out(
        quantized,
        logical_scales,
        dense_scales,
        topk_weights,
        topk_ids,
        permuted_activation,
        grouped_scales,
        expert_cursors,
        expert_offsets,
        scale_offsets,
        inverse_permutation,
        hidden,
        routed_logits,
        grouped_output,
        weight,
        weight_scales,
        source_tokens,
        renormalize,
    )


def qwen35_w1_splitk_silu_mxfp8_out(
    output: torch.Tensor,
    output_scales: torch.Tensor,
    partial: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    include_shared: bool = True,
    shared_weight: torch.Tensor | None = None,
    shared_weight_scales: torch.Tensor | None = None,
) -> None:
    """Run the cooperative Qwen3.5 small-batch split-K W1 specialization."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.ndim != 2
        or activation.shape[0] not in (1, 2, 4, 8)
        or activation.shape[1] != 2048
        or not activation.is_contiguous()
    ):
        raise ValueError(
            "activation must be contiguous CUDA MXFP8 [B,2048], with B in {1,2,4,8}"
        )
    _require_sm120(activation.device)
    batch = activation.shape[0]
    if not isinstance(include_shared, bool):
        raise TypeError("include_shared must be a bool")
    if (shared_weight is None) != (shared_weight_scales is None):
        raise ValueError(
            "shared_weight and shared_weight_scales must be provided together"
        )
    separate_shared = shared_weight is not None
    if separate_shared and (not include_shared or batch not in (1, 2, 4)):
        raise ValueError(
            "separate shared W1 currently requires include_shared=True and "
            "B in {1,2,4}"
        )
    experts = 257 if include_shared and not separate_shared else 256
    routes_per_token = 9 if include_shared else 8
    routes = batch * routes_per_token
    expected = (
        (output, torch.uint8, (routes, 256), "output"),
        (output_scales, torch.uint8, (routes, 8), "output_scales"),
        (logical_scales, torch.uint8, (batch, 64), "logical_scales"),
        (weight, torch.uint8, (experts, 512, 1536), "weight"),
        (topk_ids, torch.int32, (batch, 8), "topk_ids"),
    )
    for tensor, dtype, shape, name in expected:
        if (
            tensor.device != activation.device
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA {dtype} {shape}")
    required_partial = (
        (288 if batch in (4, 8) else 144)
        if include_shared
        else (256 if batch in (4, 8) else 128)
    ) * 128
    if (
        partial.device != activation.device
        or partial.dtype != torch.float32
        or partial.numel() < required_partial
        or not partial.is_contiguous()
    ):
        raise ValueError(
            "partial must be contiguous CUDA float32 with enough split-K storage"
        )
    if separate_shared:
        assert shared_weight is not None
        assert shared_weight_scales is not None
        if (
            shared_weight.device != activation.device
            or shared_weight.dtype != torch.uint8
            or shared_weight.shape != (1, 512, 1536)
            or not shared_weight.is_contiguous()
        ):
            raise ValueError(
                "shared_weight must be contiguous CUDA uint8 [1,512,1536]"
            )
        if (
            shared_weight_scales.device != activation.device
            or shared_weight_scales.dtype != torch.uint8
            or shared_weight_scales.numel() < 512 * 64
            or not shared_weight_scales.is_contiguous()
        ):
            raise ValueError(
                "shared_weight_scales must be contiguous CUDA uint8 with "
                "at least 512*64 values"
            )
    if (
        weight_scales.device != activation.device
        or weight_scales.dtype != torch.uint8
        or weight_scales.numel() < experts * 512 * 64
        or not weight_scales.is_contiguous()
    ):
        raise ValueError(
            "weight_scales must be contiguous CUDA uint8 with "
            f"at least {experts}*512*64 values"
        )
    load_library()
    torch.ops.mxfp6.qwen35_w1_splitk_silu_mxfp8_out(
        output,
        output_scales,
        partial,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        include_shared,
        shared_weight,
        shared_weight_scales,
    )


def qwen35_w2_splitk_reduce_out(
    output: torch.Tensor,
    partial: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_weight: torch.Tensor | None = None,
    shared_weight_scales: torch.Tensor | None = None,
) -> None:
    """Run the cooperative Qwen3.5 B=1 split-K W2/reduce specialization."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.shape != (9, 256)
        or not activation.is_contiguous()
    ):
        raise ValueError("activation must be contiguous CUDA MXFP8 [9,256]")
    _require_sm120(activation.device)
    if (shared_weight is None) != (shared_weight_scales is None):
        raise ValueError(
            "shared_weight and shared_weight_scales must be provided together"
        )
    separate_shared = shared_weight is not None
    experts = 256 if separate_shared else 257
    expected = (
        (output, torch.bfloat16, (1, 2048), "output"),
        (partial, torch.float32, (256, 9, 16), "partial"),
        (logical_scales, torch.uint8, (9, 8), "logical_scales"),
        (weight, torch.uint8, (experts, 2048, 192), "weight"),
        (topk_ids, torch.int32, (1, 8), "topk_ids"),
        (topk_weights, torch.float32, (1, 8), "topk_weights"),
    )
    for tensor, dtype, shape, name in expected:
        if (
            tensor.device != activation.device
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA {dtype} {shape}")
    if (
        weight_scales.device != activation.device
        or weight_scales.dtype != torch.uint8
        or weight_scales.numel() < experts * 2048 * 8
        or not weight_scales.is_contiguous()
    ):
        raise ValueError(
            "weight_scales must be contiguous CUDA uint8 with "
            f"at least {experts}*2048*8 values"
        )
    if separate_shared:
        assert shared_weight is not None
        assert shared_weight_scales is not None
        if (
            shared_weight.device != activation.device
            or shared_weight.dtype != torch.uint8
            or shared_weight.shape != (1, 2048, 192)
            or not shared_weight.is_contiguous()
        ):
            raise ValueError(
                "shared_weight must be contiguous CUDA uint8 [1,2048,192]"
            )
        if (
            shared_weight_scales.device != activation.device
            or shared_weight_scales.dtype != torch.uint8
            or shared_weight_scales.numel() < 2048 * 8
            or not shared_weight_scales.is_contiguous()
        ):
            raise ValueError(
                "shared_weight_scales must be contiguous CUDA uint8 with "
                "at least 2048*8 values"
            )
    if (
        shared_gate.device != activation.device
        or shared_gate.dtype != torch.bfloat16
        or shared_gate.numel() != 1
        or not shared_gate.is_contiguous()
    ):
        raise ValueError("shared_gate must be contiguous CUDA bfloat16 [1]")
    load_library()
    torch.ops.mxfp6.qwen35_w2_splitk_reduce_out(
        output,
        partial,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        topk_weights,
        shared_gate,
        shared_weight,
        shared_weight_scales,
    )


def array_gemm_w6a8_reduce_out(
    output: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_output: torch.Tensor | None = None,
    shared_weight: torch.Tensor | None = None,
    shared_weight_scales: torch.Tensor | None = None,
    use_packed_vector_loads: bool = False,
) -> None:
    """Fuse Qwen top-8 W2 GEMMs with routed/shared weighted reduction."""
    if (
        activation.device.type != "cuda"
        or activation.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or activation.ndim != 2
        or not activation.is_contiguous()
    ):
        raise ValueError(
            "activation must be contiguous CUDA float8_e4m3fn/uint8 "
            "[tokens*(8 or 9),K]"
        )
    _require_sm120(activation.device)
    for tensor, name in (
        (logical_scales, "logical_scales"),
        (weight, "weight"),
        (weight_scales, "weight_scales"),
    ):
        _require_cuda_uint8_contiguous(tensor, name)
        if tensor.device != activation.device:
            raise ValueError(f"{name} must be on {activation.device}")
    if (
        topk_ids.device != activation.device
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_ids.ndim != 2
        or topk_ids.shape[1] != 8
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("topk_ids must be contiguous CUDA int32/int64 [tokens,8]")
    if (
        topk_weights.device != activation.device
        or topk_weights.dtype != torch.float32
        or topk_weights.shape != topk_ids.shape
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("topk_weights must be contiguous CUDA float32 [tokens,8]")
    if (
        shared_gate.device != activation.device
        or shared_gate.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not shared_gate.is_contiguous()
    ):
        raise ValueError("shared_gate must be contiguous CUDA float32/float16/bfloat16")
    if (
        output.device != activation.device
        or output.dtype not in (torch.float16, torch.bfloat16)
        or output.ndim != 2
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous CUDA float16/bfloat16 [tokens,N]")
    if shared_output is not None and (
        shared_output.device != output.device
        or shared_output.dtype != output.dtype
        or shared_output.shape != output.shape
        or not shared_output.is_contiguous()
    ):
        raise ValueError(
            "shared_output must be contiguous and match output's shape, "
            "dtype, and CUDA device"
        )
    if (shared_weight is None) != (shared_weight_scales is None):
        raise ValueError(
            "shared_weight and shared_weight_scales must be provided together"
        )
    if shared_output is not None and shared_weight is not None:
        raise ValueError(
            "separate shared weights cannot be combined with shared_output"
        )
    if shared_weight is not None:
        assert shared_weight_scales is not None
        if (
            weight.ndim != 3
            or weight.shape[0] != 256
            or shared_weight.device != activation.device
            or shared_weight.dtype != torch.uint8
            or shared_weight.shape != (1, weight.shape[1], weight.shape[2])
            or not shared_weight.is_contiguous()
        ):
            raise ValueError(
                "separate shared W2 requires 256 routed experts and contiguous "
                "CUDA uint8 shared_weight [1,N,packed_K]"
            )
        packed_k_blocks = ((activation.shape[1] // 32 + 3) // 4) * 4
        if (
            shared_weight_scales.device != activation.device
            or shared_weight_scales.dtype != torch.uint8
            or shared_weight_scales.numel()
            < weight.shape[1] * packed_k_blocks
            or not shared_weight_scales.is_contiguous()
        ):
            raise ValueError(
                "shared_weight_scales must be contiguous CUDA uint8 with "
                "enough values for one shared expert"
            )
        if output.dtype != torch.bfloat16 or topk_ids.dtype != torch.int32:
            raise ValueError(
                "separate shared W2 currently requires bfloat16 output and "
                "int32 ids"
            )
    load_library()
    torch.ops.mxfp6.array_gemm_w6a8_reduce_out(
        output,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        topk_weights,
        shared_gate,
        shared_output,
        shared_weight,
        shared_weight_scales,
        use_packed_vector_loads,
    )


def moe_reduce_array_out(
    output: torch.Tensor,
    combined_output: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_gate: torch.Tensor,
) -> None:
    """Reduce interleaved routed/shared rows emitted by the array path."""
    if (
        combined_output.device.type != "cuda"
        or combined_output.dtype not in (torch.float16, torch.bfloat16)
        or combined_output.ndim != 2
        or not combined_output.is_contiguous()
    ):
        raise ValueError(
            "combined_output must be contiguous CUDA float16/bfloat16 "
            "[tokens*(topk+1),H]"
        )
    if (
        output.device != combined_output.device
        or output.dtype != combined_output.dtype
        or output.ndim != 2
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous CUDA [tokens,H] with combined_output's dtype"
        )
    if (
        topk_weights.device != combined_output.device
        or topk_weights.dtype != torch.float32
        or topk_weights.ndim != 2
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("topk_weights must be contiguous CUDA float32 [tokens,topk]")
    if (
        shared_gate.device != combined_output.device
        or shared_gate.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not shared_gate.is_contiguous()
        or (
            shared_gate.numel() != output.shape[0]
            and (shared_gate.ndim != 2 or shared_gate.shape[0] != output.shape[0])
        )
    ):
        raise ValueError(
            "shared_gate must contain one contiguous floating value or row per token"
        )
    load_library()
    torch.ops.mxfp6.moe_reduce_array_out(
        output,
        combined_output,
        topk_weights,
        shared_gate,
    )


def grouped_gemm_w6a8(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return native SM120 grouped ``A8 @ W6.T`` for routed MoE rows."""
    out_dtype = _validate_output_dtype(out_dtype)
    if weight.ndim != 3:
        raise ValueError("weight must have shape [E,N,packed_K]")
    output = torch.empty(
        (activation.shape[0], weight.shape[1]),
        device=activation.device,
        dtype=out_dtype,
    )
    grouped_gemm_w6a8_out(
        output,
        activation,
        activation_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )
    return output


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


def is_tuned_shape(m: int, n: int, k: int) -> bool:
    """Return whether a problem has an exact target-shape override."""
    if m in TUNED_M and (n, k) in TUNED_NK:
        return True
    return m in (40, 48) and (n, k) == (5120, 3072)


def is_available() -> bool:
    """Return whether the library can load and the current GPU is SM120."""
    if not torch.cuda.is_available():
        return False
    try:
        load_library()
        return torch.cuda.get_device_capability() == (12, 0)
    except (ImportError, OSError, RuntimeError):
        return False
