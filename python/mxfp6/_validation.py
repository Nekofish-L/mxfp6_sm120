from __future__ import annotations
import torch

SCALE_VECTOR_SIZE = 32


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
            f"out_dtype must be torch.float16 or torch.bfloat16; got {out_dtype}"
        )
    return out_dtype


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
