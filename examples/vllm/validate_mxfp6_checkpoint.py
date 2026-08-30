#!/usr/bin/env python3
"""Validate a model-scoped Quark/OCP MXFP6 checkpoint without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any


CONVERTER_COMMIT = "37242a6a1bf6869857084d2ac7ccb22d1af7168d"
MAX_HEADER_BYTES = 256 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class Profile:
    source_model: str
    source_revision: str
    model_type: str
    architecture: str
    quantized_weights: int
    total_tensors: int
    shards: int
    tensor_contract_sha256: str
    must_stay_dense: tuple[str, ...]

PROFILES = {
    "qwen3.5-27b": Profile(
        source_model="Qwen/Qwen3.5-27B",
        source_revision="fc05daec18b0a78c049392ed2e771dde82bdf654",
        model_type="qwen3_5",
        architecture="Qwen3_5ForConditionalGeneration",
        quantized_weights=408,
        total_tensors=1607,
        shards=11,
        tensor_contract_sha256=(
            "9996c487c1b7d503cfb917b33d6a6d77d8bf09c1c0d61f73a5c7e65db7d3942a"
        ),
        must_stay_dense=(
            "model.language_model.embed_tokens",
            "mtp.layers.0.embed_tokens",
            "lm_head",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "visual.",
        ),
    ),
    "qwen3.5-35b-a3b": Profile(
        source_model="Qwen/Qwen3.5-35B-A3B",
        source_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        model_type="qwen3_5_moe",
        architecture="Qwen3_5MoeForConditionalGeneration",
        quantized_weights=31746,
        total_tensors=64197,
        shards=14,
        tensor_contract_sha256=(
            "4bf3ac4923f8d9a8969c7e9eea98f6f9bdbafbd13e1a298e5e30df1152fa28a7"
        ),
        must_stay_dense=(
            "model.language_model.embed_tokens",
            "lm_head",
            ".mlp.gate",
            ".mlp.shared_expert_gate",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "visual.",
        ),
    ),
}


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        (header_length,) = struct.unpack("<Q", raw_length)
        if (
            header_length == 0
            or header_length > file_size - 8
            or header_length > MAX_HEADER_BYTES
        ):
            raise ValueError(f"{path}: invalid safetensors header length {header_length}")
        try:
            header = json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: invalid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be an object")
    header.pop("__metadata__", None)

    payload_size = file_size - 8 - header_length
    regions = []
    for name, metadata in header.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: `{name}` has invalid tensor metadata")
        shape = _shape(metadata, name)
        dtype = metadata.get("dtype")
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"{path}: `{name}` has unsupported dtype {dtype!r}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"{path}: `{name}` has invalid data offsets")
        start, end = offsets
        expected_bytes = prod(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > payload_size:
            raise ValueError(f"{path}: `{name}` has out-of-bounds data offsets")
        if end - start != expected_bytes:
            raise ValueError(
                f"{path}: `{name}` stores {end - start} bytes; expected {expected_bytes}"
            )
        regions.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(regions):
        if start != cursor:
            raise ValueError(
                f"{path}: `{name}` begins at {start}; expected contiguous offset {cursor}"
            )
        cursor = end
    if cursor != payload_size:
        raise ValueError(
            f"{path}: tensor data covers {cursor} bytes; payload has {payload_size}"
        )
    return header


def _require(mapping: dict[str, Any], path: str, expected: Any) -> None:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"config.json is missing `{path}`")
        value = value[component]
    if value != expected:
        raise ValueError(f"config.json `{path}` is {value!r}; expected {expected!r}")


def _validate_quantization_config(config: dict[str, Any]) -> None:
    required = {
        "quantization_config.quant_method": "quark",
        "quantization_config.global_quant_config.weight.dtype": "fp6_e3m2",
        "quantization_config.global_quant_config.weight.is_dynamic": False,
        "quantization_config.global_quant_config.weight.qscheme": "per_group",
        "quantization_config.global_quant_config.weight.group_size": 32,
        "quantization_config.global_quant_config.weight.scale_format": "e8m0",
        "quantization_config.global_quant_config.input_tensors.dtype": "fp8_e4m3",
        "quantization_config.global_quant_config.input_tensors.is_dynamic": True,
        "quantization_config.global_quant_config.input_tensors.qscheme": "per_group",
        "quantization_config.global_quant_config.input_tensors.group_size": 32,
        "quantization_config.global_quant_config.input_tensors.scale_format": "e8m0",
    }
    for path, expected in required.items():
        _require(config, path, expected)


def _shape(metadata: dict[str, Any], name: str) -> tuple[int, ...]:
    shape = metadata.get("shape")
    if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
        raise ValueError(f"{name}: invalid shape metadata")
    return tuple(shape)


def _tensor_contract_sha256(tensors: dict[str, dict[str, Any]]) -> str:
    contract = [
        [name, metadata.get("dtype"), list(_shape(metadata, name))]
        for name, metadata in sorted(tensors.items())
    ]
    encoded = json.dumps(contract, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint(checkpoint: Path, profile_name: str | None) -> dict[str, Any]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing {config_path}")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{config_path}: invalid JSON") from error
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: top-level value must be an object")
    _validate_quantization_config(config)

    shard_paths = sorted(checkpoint.glob("*.safetensors"))
    if not shard_paths:
        raise ValueError(f"no safetensors shards found in {checkpoint}")

    tensors: dict[str, dict[str, Any]] = {}
    for shard_path in shard_paths:
        for name, metadata in _read_safetensors_header(shard_path).items():
            if name in tensors:
                raise ValueError(f"duplicate tensor `{name}`")
            if not isinstance(metadata, dict):
                raise ValueError(f"{name}: invalid tensor metadata")
            tensors[name] = metadata

    quantized_weights = 0
    for name, metadata in tensors.items():
        if not name.endswith(".weight") or metadata.get("dtype") != "U8":
            continue
        weight_shape = _shape(metadata, name)
        if len(weight_shape) != 2:
            raise ValueError(f"{name}: packed MXFP6 weight must be two-dimensional")
        rows, packed_k = weight_shape
        if rows <= 0 or rows % 8 or packed_k <= 0 or packed_k % 3:
            raise ValueError(f"{name}: invalid packed MXFP6 shape {weight_shape}")
        k = packed_k * 4 // 3
        if k % 128:
            raise ValueError(f"{name}: decoded K={k} is not divisible by 128")
        scale_name = f"{name[:-7]}.weight_scale"
        scale = tensors.get(scale_name)
        if scale is None:
            raise ValueError(f"{name}: missing `{scale_name}`")
        if scale.get("dtype") != "U8" or _shape(scale, scale_name) != (rows, k // 32):
            raise ValueError(
                f"{scale_name}: expected U8 shape {(rows, k // 32)} for `{name}`"
            )
        quantized_weights += 1

    for name in tensors:
        if not name.endswith(".weight_scale"):
            continue
        weight_name = f"{name[:-13]}.weight"
        weight = tensors.get(weight_name)
        if weight is None or weight.get("dtype") != "U8":
            raise ValueError(f"{name}: missing packed U8 weight `{weight_name}`")

    tensor_contract_sha256 = _tensor_contract_sha256(tensors)
    profile = PROFILES.get(profile_name) if profile_name else None
    if profile:
        if config.get("model_type") != profile.model_type:
            raise ValueError(
                f"config.json model_type is {config.get('model_type')!r}; "
                f"expected {profile.model_type!r} for {profile_name}"
            )
        if config.get("architectures") != [profile.architecture]:
            raise ValueError(
                f"config.json architectures is {config.get('architectures')!r}; "
                f"expected {[profile.architecture]!r} for {profile_name}"
            )
        for name, metadata in tensors.items():
            if metadata.get("dtype") != "U8":
                continue
            if any(fragment in name for fragment in profile.must_stay_dense):
                raise ValueError(f"{name}: profile requires this tensor to stay dense")
        expected = {
            "quantized_weights": profile.quantized_weights,
            "total_tensors": profile.total_tensors,
            "shards": profile.shards,
        }
        actual = {
            "quantized_weights": quantized_weights,
            "total_tensors": len(tensors),
            "shards": len(shard_paths),
        }
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                raise ValueError(
                    f"{key} is {actual[key]}; expected {expected_value} for {profile_name}"
                )
        if profile.tensor_contract_sha256 and (
            tensor_contract_sha256 != profile.tensor_contract_sha256
        ):
            raise ValueError(
                "tensor name/dtype/shape contract is "
                f"{tensor_contract_sha256}; expected "
                f"{profile.tensor_contract_sha256} for {profile_name}"
            )

    return {
        "status": "ok",
        "checkpoint": str(checkpoint.resolve()),
        "profile": profile_name or "layout-only",
        "converter_commit": CONVERTER_COMMIT,
        "source_model": profile.source_model if profile else None,
        "recipe_source_revision": profile.source_revision if profile else None,
        "quantized_weights": quantized_weights,
        "total_tensors": len(tensors),
        "safetensors_shards": len(shard_paths),
        "tensor_contract_sha256": tensor_contract_sha256,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(checkpoint: Path, output: Path, result: dict[str, Any]) -> None:
    files = [checkpoint / "config.json", *sorted(checkpoint.glob("*.safetensors"))]
    manifest = {
        **result,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="write config/shard sizes and SHA-256 hashes after validation",
    )
    args = parser.parse_args()

    try:
        result = validate_checkpoint(args.checkpoint, args.profile)
        if args.write_manifest:
            _write_manifest(args.checkpoint, args.write_manifest, result)
            result["manifest"] = str(args.write_manifest.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ValueError) as error:
        parser.exit(1, f"validation failed: {error}\n")


if __name__ == "__main__":
    main()
