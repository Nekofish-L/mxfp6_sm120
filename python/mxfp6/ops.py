from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

import torch

from ._loader import load_library

if TYPE_CHECKING:
    from .autotune import W6A8Config
    from .humming_backend import HummingMXFP6Weight


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
            "out_dtype must be torch.float16 or torch.bfloat16; "
            f"got {out_dtype}"
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
        is_autotune_enabled,
        should_tune_exact_shapes,
    )

    if not is_autotune_enabled() or not can_autotune_now():
        return False
    if is_tuned_shape(m, n, k) and not should_tune_exact_shapes():
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
    from .autotune import ensure_w6a8_tuned

    config = ensure_w6a8_tuned(
        a, b, sfa, sfb, m, n, k, out_dtype=out_dtype, force=force
    )
    if config is not None:
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
            "input must have dtype torch.float16 or torch.bfloat16; "
            f"got {input.dtype}"
        )
    if input.ndim != 2:
        raise ValueError(f"input must have shape [M,K]; got {tuple(input.shape)}")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    rows, k = (int(value) for value in input.shape)
    if rows <= 0 or k <= 0 or k % 128:
        raise ValueError("M must be positive and K a positive multiple of 128")
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


def expand_fp6_to_fp8(
    packed: torch.Tensor, rows: int, k: int
) -> torch.Tensor:
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
        logical_scales = unpack_scales(
            operand.scales, operand.rows, operand.k
        )
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
    return torch.ops.mxfp6.gemm(
        a, b, sfa, sfb, m, n, k, alpha, out_dtype
    )


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
    if _needs_w6a8_autotune(
        a.device, a.rows, b.rows, a.k, out_dtype
    ):
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
    calls invoke the same tuner automatically; checked-in exact shapes keep
    their deterministic dispatch unless ``force=True``.
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
        from .autotune import should_tune_exact_shapes

        if not should_tune_exact_shapes():
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
            config = autotune_w6a8(
                a, b, out_dtype=out_dtype, force=force
            )
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
            config = autotune_w6a8(
                a, b, out_dtype=out_dtype, force=force
            )
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
    b: PackedMXFP6Tensor | HummingMXFP6Weight,
    alpha: float = 1.0,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute ``A @ B.T`` using the native current-repository kernels.

    FP16/BF16 activation tensors are dynamically mapped 16-to-8 and use W6A8
    for every batch size. :class:`MXFP8Tensor` skips that conversion. Packed
    MXFP6 activation inputs retain the legacy W6A6-compatible API.

    A :class:`~mxfp6.HummingMXFP6Weight` selects the optional Humming reference
    backend only when explicitly supplied; ordinary weights never route to it.
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
            "a must be an FP16/BF16 Tensor, MXFP8Tensor, or "
            "PackedMXFP6Tensor"
        )

    # Keep Humming optional and lazily imported. This also avoids starting its
    # background JIT launcher for users of the native W6A6 path.
    from .humming_backend import HummingMXFP6Weight, gemm_humming

    if isinstance(b, HummingMXFP6Weight):
        return gemm_humming(a, b, alpha, out_dtype=out_dtype)
    if not isinstance(b, PackedMXFP6Tensor):
        raise TypeError(
            "b must be a PackedMXFP6Tensor or HummingMXFP6Weight instance"
        )
    if a.k != b.k:
        raise ValueError(f"a.k and b.k must match; got {a.k} and {b.k}")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    return gemm_packed(
        a.values, b.values, a.scales, b.scales,
        a.rows, b.rows, a.k, alpha, out_dtype=out_dtype,
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
    return m in TUNED_M and (n, k) in TUNED_NK


def is_available() -> bool:
    """Return whether the library can load and the current GPU is SM120."""
    if not torch.cuda.is_available():
        return False
    try:
        load_library()
        return torch.cuda.get_device_capability() == (12, 0)
    except (ImportError, OSError, RuntimeError):
        return False
