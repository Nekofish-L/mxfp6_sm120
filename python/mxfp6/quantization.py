"""Packed operand types and native quantization/conversion operations."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from ._loader import load_library
from ._validation import (SCALE_VECTOR_SIZE, _require_cuda_uint8_contiguous, _require_sm120, _validate_float_activation)


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
    """A byte-aligned E4M3 matrix and UE8M0/32 scales for W6A8 or W8A8."""

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

    def as_float8(self) -> torch.Tensor:
        """Return the E4M3 payload viewed as a float8 matrix (without scales)."""
        return self.values.view(self.rows, self.k).view(torch.float8_e4m3fn)

    # Historical name retained for callers; this view does not apply scales.
    dequantized_values = as_float8


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


def quantize_mxfp6(input: torch.Tensor) -> PackedMXFP6Tensor:
    """Dynamically map a contiguous FP16/BF16 ``[M,K]`` matrix to MXFP6."""
    rows, k = _validate_float_activation(input)
    _require_sm120(input.device)
    load_library()
    values, scales = torch.ops.mxfp6.quantize_mxfp6(input)
    return PackedMXFP6Tensor(values, scales, rows, k)


def silu_and_mul_mxfp8(input: torch.Tensor) -> MXFP8Tensor:
    """Quantize BF16/FP16 SwiGLU with rounded SiLU/product and up + positive zero.

    Unlike the MoE packed-out producer, this dense interface also initializes
    every padding scale on each call and matches the separate activation's
    signed-zero boundary. Input is contiguous [M,2K], gate followed by up.
    """
    _require_sm120(input.device)
    load_library()
    values, scales = torch.ops.mxfp6.silu_and_mul_mxfp8(input)
    return MXFP8Tensor(values=values, scales=scales, rows=input.shape[0], k=input.shape[1]//2)


# Public compatibility alias for activation quantization.
quantize_activation = quantize_mxfp8
