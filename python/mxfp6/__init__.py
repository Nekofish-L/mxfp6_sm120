"""Packed MXFP6 GEMM kernels for NVIDIA SM120 GPUs."""

from ._loader import load_library
from .humming_backend import (
    HummingMXFP6Weight,
    gemm_humming,
    is_humming_available,
    prepare_humming_weight,
)
from .ops import (
    PackedMXFP6Tensor,
    SCALE_VECTOR_SIZE,
    TUNED_NK,
    expand_fp6_to_fp8,
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
    "HummingMXFP6Weight",
    "SCALE_VECTOR_SIZE",
    "TUNED_NK",
    "__version__",
    "expand_fp6_to_fp8",
    "gemm",
    "gemm_from_codes",
    "gemm_humming",
    "gemm_packed",
    "is_available",
    "is_humming_available",
    "is_tuned_shape",
    "load_library",
    "pack_fp6",
    "pack_operand",
    "pack_scales",
    "prepare_humming_weight",
    "unpack_fp6",
    "unpack_operand",
    "unpack_scales",
]
