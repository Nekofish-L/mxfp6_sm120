"""Optional Humming W6A8 reference backend for SM120 GEMMs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path
import sys
from typing import Any

import torch

from .ops import PackedMXFP6Tensor, expand_fp6_to_fp8, unpack_scales


@lru_cache(maxsize=1)
def _import_humming() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load an installed Humming package or the bundled source snapshot."""
    try:
        humming = importlib.import_module("humming")
    except ModuleNotFoundError as error:
        if error.name != "humming":
            raise RuntimeError(
                "The Humming backend is missing a runtime dependency. Install "
                "the project with `pip install 'mxfp6-sm120[humming]'`."
            ) from error

        package_dir = Path(__file__).resolve().parent
        candidates = (
            package_dir / "_vendor",
            package_dir.parents[1] / "third_party" / "humming",
        )
        source_root = next(
            (path for path in candidates if (path / "humming" / "__init__.py").is_file()),
            None,
        )
        if source_root is None:
            raise RuntimeError(
                "Humming sources are unavailable. Initialize third_party/humming "
                "or install the project with the `humming` extra."
            ) from error
        sys.path.insert(0, str(source_root))
        try:
            humming = importlib.import_module("humming")
        except ModuleNotFoundError as dependency_error:
            raise RuntimeError(
                "The Humming backend is missing a runtime dependency. Install "
                "the project with `pip install 'mxfp6-sm120[humming]'`."
            ) from dependency_error

    del humming
    dtypes = importlib.import_module("humming.dtypes")
    ops = importlib.import_module("humming.ops")
    kernel_module = importlib.import_module("humming.kernel.humming")
    layer_module = importlib.import_module("humming.layer")
    tune_module = importlib.import_module("humming.tune")
    weight_module = importlib.import_module("humming.utils.weight")
    return dtypes, ops, kernel_module, layer_module, tune_module, weight_module


def _logical_scales(operand: PackedMXFP6Tensor) -> torch.Tensor:
    if operand.logical_scales is not None:
        return operand.logical_scales
    return unpack_scales(operand.scales, operand.rows, operand.k)


@dataclass(frozen=True)
class HummingMXFP6Weight:
    """Persistent W6 weight in Humming's mixed W6A8 physical layout.

    The values remain bit-packed at six bits each. Only activations are
    losslessly expanded to FP8 at execution time.
    """

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

    @property
    def nbytes(self) -> int:
        return (
            self.values.numel() * self.values.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


def prepare_humming_weight(weight: PackedMXFP6Tensor) -> HummingMXFP6Weight:
    """Repack a persistent E3M2 weight once for the reference W6A8 kernel."""
    if not isinstance(weight, PackedMXFP6Tensor):
        raise TypeError("weight must be a PackedMXFP6Tensor instance")
    if weight.rows % 256:
        raise ValueError(
            f"Humming W6A8 requires N divisible by 256; got {weight.rows}"
        )
    if weight.k % 128:
        raise ValueError(f"Humming W6A8 requires K divisible by 128; got {weight.k}")

    dtypes, _, _, _, _, weight_utils = _import_humming()
    packed_words = weight.values.view(weight.rows, weight.k * 6 // 8).view(torch.int32)
    values = weight_utils.prepare_humming_weight(
        packed_words,
        b_dtype=dtypes.float6e3m2,
        a_dtype=dtypes.float8e4m3,
        packed=True,
    )
    logical_scales = _logical_scales(weight).contiguous()
    scales = weight_utils.prepare_humming_weight_scale(
        logical_scales.view(torch.float8_e8m0fnu),
        is_mxmma=True,
        mxmma_scale_vec=1,
    )
    return HummingMXFP6Weight(values, scales, weight.rows, weight.k)


def _layer_config(n: int, k: int) -> dict[str, object]:
    return {
        "shape_n": n,
        "shape_k": k,
        "a_dtype": "float8e4m3",
        "b_dtype": "float6e3m2",
        "c_dtype": "float16",
        "bs_dtype": "float8e8m0",
        "input_scale_group_size": 32,
        "weight_scale_group_size": 32,
        "weight_scale_type": "group",
        "mma_type": "mxmma",
    }


@lru_cache(maxsize=128)
def _kernel_configs(
    device_index: int, m: int, n: int, k: int
) -> torch.Tensor:
    # Humming chooses the tile and pipeline from its SM120 heuristic. Three
    # safety overrides avoid rare partial-output corruption observed in
    # multi-wave dense GEMMs in the pinned revision. Disabling overlapped
    # reduction also restores the two stages selected by Humming's base SM120
    # W6A8 configuration; its heuristic raises this to three only for the
    # overlapped schedule.
    with torch.cuda.device(device_index):
        _, _, kernel_module, layer_module, tune_module, _ = _import_humming()
        layer = _layer_config(n, k)
        meta = layer_module.HummingLayerMeta(**layer)
        tuning = dict(tune_module.get_heuristics_config(meta, shape_m=m))
        tuning.update(
            num_stages=2,
            use_stream_k=False,
            use_tma_c=False,
            reduce_overlap_last_stage_only=False,
        )
        return kernel_module.HummingKernel.prepare_kernels(layer, {}, tuning)


def gemm_humming(
    a: PackedMXFP6Tensor,
    b: HummingMXFP6Weight,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Run the Humming mixed W6A8 path for ``A @ B.T``.

    E3M2 is a subset of E4M3, so activation conversion is exact. The returned
    tensor is FP16, matching the native operator.
    """
    if not isinstance(a, PackedMXFP6Tensor):
        raise TypeError("a must be a PackedMXFP6Tensor instance")
    if not isinstance(b, HummingMXFP6Weight):
        raise TypeError("b must be a HummingMXFP6Weight instance")
    if alpha != 1.0:
        raise ValueError("the Humming backend currently requires alpha == 1.0")
    if a.k != b.k:
        raise ValueError(f"a.k and b.k must match; got {a.k} and {b.k}")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    device_index = a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (12, 0):
        raise RuntimeError(
            f"Humming W6A8 requires SM120; device {device_index} is "
            f"SM{capability[0]}{capability[1]}"
        )

    _, humming_ops, _, _, _, _ = _import_humming()
    configs = _kernel_configs(device_index, a.rows, b.rows, a.k)
    inputs = expand_fp6_to_fp8(a.values, a.rows, a.k)
    input_scale = _logical_scales(a).contiguous().view(torch.int32)
    output = torch.empty(
        (a.rows, b.rows), dtype=torch.float16, device=a.device
    )
    return humming_ops.launch_kernel(
        configs=configs,
        inputs=inputs,
        weight=b.values,
        outputs=output,
        input_scale=input_scale,
        weight_scale=b.scales,
    )


def is_humming_available() -> bool:
    """Return whether the optional backend and its dependencies can load."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        return False
    try:
        _import_humming()
    except (ImportError, OSError, RuntimeError):
        return False
    return True
