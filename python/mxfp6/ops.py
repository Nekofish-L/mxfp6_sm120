from __future__ import annotations

from dataclasses import dataclass

import torch

from ._loader import load_library


SCALE_VECTOR_SIZE = 32
TUNED_NK = frozenset(
    {
        (5120, 8192),
        (3072, 5120),
        (5120, 7168),
        (5120, 17408),
        (8704, 5120),
    }
)


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


def _validate_problem(m: int, n: int, k: int) -> None:
    if not (1 <= m <= 32 or m == 2048):
        raise ValueError(f"m must be in [1, 32] or equal to 2048; got {m}")
    if n <= 0 or n % 128:
        raise ValueError(f"n must be a positive multiple of 128; got {n}")
    if k <= 0 or k % 128:
        raise ValueError(f"k must be a positive multiple of 128; got {k}")


@dataclass(frozen=True)
class PackedMXFP6Tensor:
    """An E3M2 matrix and its UE8M0/32 scales in SM120 physical layout."""

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


def pack_scales(logical: torch.Tensor) -> torch.Tensor:
    """Reorder logical UE8M0 scales to the SM120 CUTLASS physical layout.

    ``logical`` is a contiguous CUDA uint8 tensor of shape ``[rows, k / 32]``.
    Byte values are encoded UE8M0 exponents (``0x7f`` represents 1.0).
    """
    _require_cuda_uint8_contiguous(logical, "logical scales")
    if (
        logical.ndim != 2
        or logical.shape[0] <= 0
        or logical.shape[1] <= 0
        or logical.shape[1] % 4
    ):
        raise ValueError(
            "logical scales must have shape [rows, k / 32] with k divisible by 128"
        )
    rows, k_blocks = logical.shape
    load_library()
    return torch.ops.mxfp6.pack_scales(
        logical, int(rows), int(k_blocks * SCALE_VECTOR_SIZE)
    )


def unpack_scales(packed: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Convert packed SM120 scales back to logical ``[rows, k / 32]``."""
    _require_cuda_uint8_contiguous(packed, "packed scales")
    if rows <= 0 or k <= 0 or k % 128:
        raise ValueError("rows must be positive and k must be divisible by 128")
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
    if k % 128 or tuple(logical_scales.shape) != expected_scale_shape:
        raise ValueError(
            f"logical_scales must have shape {expected_scale_shape}; "
            f"got {tuple(logical_scales.shape)}"
        )
    if codes.device != logical_scales.device:
        raise ValueError("codes and logical_scales must be on the same device")
    return PackedMXFP6Tensor(
        pack_fp6(codes), pack_scales(logical_scales), rows, k
    )


def unpack_operand(
    operand: PackedMXFP6Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logical E3M2 codes and logical UE8M0 scales for an operand."""
    return (
        unpack_fp6(operand.values, operand.rows, operand.k),
        unpack_scales(operand.scales, operand.rows, operand.k),
    )


def gemm_packed(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Low-level ``A @ B.T`` for already packed values and scales."""
    _validate_problem(m, n, k)
    for tensor, name in ((a, "a"), (b, "b"), (sfa, "sfa"), (sfb, "sfb")):
        _require_cuda_uint8_contiguous(tensor, name)
    devices = {a.device, b.device, sfa.device, sfb.device}
    if len(devices) != 1:
        raise ValueError("a, b, sfa, and sfb must be on the same CUDA device")
    _require_sm120(a.device)
    load_library()
    return torch.ops.mxfp6.gemm(a, b, sfa, sfb, m, n, k, alpha)


def gemm(
    a: PackedMXFP6Tensor,
    b: PackedMXFP6Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute ``A @ B.T`` from two prepacked MXFP6 operands."""
    if not isinstance(a, PackedMXFP6Tensor) or not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("a and b must be PackedMXFP6Tensor instances")
    if a.k != b.k:
        raise ValueError(f"a.k and b.k must match; got {a.k} and {b.k}")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    return gemm_packed(
        a.values, b.values, a.scales, b.scales,
        a.rows, b.rows, a.k, alpha,
    )


def gemm_from_codes(
    a_codes: torch.Tensor,
    b_codes: torch.Tensor,
    sfa_logical: torch.Tensor,
    sfb_logical: torch.Tensor,
    alpha: float = 1.0,
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
    )


def is_tuned_shape(m: int, n: int, k: int) -> bool:
    """Return whether a problem is one of the 20 profiler-tuned shapes."""
    return m in (1, 16, 32, 2048) and (n, k) in TUNED_NK


def is_available() -> bool:
    """Return whether the library can load and the current GPU is SM120."""
    if not torch.cuda.is_available():
        return False
    try:
        load_library()
        return torch.cuda.get_device_capability() == (12, 0)
    except (ImportError, OSError, RuntimeError):
        return False
