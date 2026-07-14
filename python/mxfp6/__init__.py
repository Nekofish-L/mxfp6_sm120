"""Native SM120 MXFP6 E3M2 x E3M2 block-scaled GEMM."""

from ._loader import load_library
from .ops import (
    PackedMXFP6Tensor,
    SCALE_VECTOR_SIZE,
    TUNED_NK,
    gemm,
    gemm_from_codes,
    gemm_packed,
    is_available,
    is_tuned_shape,
    pack_fp6,
    pack_operand,
    pack_scales,
    unpack_fp6,
    unpack_operand,
    unpack_scales,
)

__version__ = "0.1.0"

__all__ = [
    "PackedMXFP6Tensor",
    "SCALE_VECTOR_SIZE",
    "TUNED_NK",
    "__version__",
    "gemm",
    "gemm_from_codes",
    "gemm_packed",
    "is_available",
    "is_tuned_shape",
    "load_library",
    "pack_fp6",
    "pack_operand",
    "pack_scales",
    "unpack_fp6",
    "unpack_operand",
    "unpack_scales",
]
