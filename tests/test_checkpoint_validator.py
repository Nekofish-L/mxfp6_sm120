import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_mxfp6_checkpoint",
    ROOT / "examples/vllm/validate_mxfp6_checkpoint.py",
)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], int]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _config() -> dict:
    return {
        "quantization_config": {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {
                    "dtype": "fp6_e3m2",
                    "is_dynamic": False,
                    "qscheme": "per_group",
                    "group_size": 32,
                    "scale_format": "e8m0",
                },
                "input_tensors": {
                    "dtype": "fp8_e4m3",
                    "is_dynamic": True,
                    "qscheme": "per_group",
                    "group_size": 32,
                    "scale_format": "e8m0",
                },
            },
        }
    }


class CheckpointValidatorTest(unittest.TestCase):
    def test_valid_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(json.dumps(_config()))
            _write_safetensors(
                checkpoint / "model.safetensors",
                {
                    "model.linear.weight": ("U8", [8, 96], 768),
                    "model.linear.weight_scale": ("U8", [8, 4], 32),
                    "model.embed_tokens.weight": ("BF16", [16, 128], 4096),
                },
            )
            result = VALIDATOR.validate_checkpoint(checkpoint, None)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["quantized_weights"], 1)
            self.assertEqual(result["total_tensors"], 3)

            manifest_path = checkpoint / "manifest.json"
            VALIDATOR._write_manifest(checkpoint, manifest_path, result)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(
                [entry["path"] for entry in manifest["files"]],
                ["config.json", "model.safetensors"],
            )
            self.assertTrue(
                all(len(entry["sha256"]) == 64 for entry in manifest["files"])
            )

    def test_rejects_wrong_scale_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(json.dumps(_config()))
            _write_safetensors(
                checkpoint / "model.safetensors",
                {
                    "model.linear.weight": ("U8", [8, 96], 768),
                    "model.linear.weight_scale": ("U8", [8, 3], 24),
                },
            )
            with self.assertRaisesRegex(ValueError, "expected U8 shape"):
                VALIDATOR.validate_checkpoint(checkpoint, None)

    def test_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(json.dumps(_config()))
            shard = checkpoint / "model.safetensors"
            _write_safetensors(
                shard,
                {
                    "model.linear.weight": ("U8", [8, 96], 768),
                    "model.linear.weight_scale": ("U8", [8, 4], 32),
                },
            )
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "out-of-bounds"):
                VALIDATOR.validate_checkpoint(checkpoint, None)

    def test_profile_checks_complete_tensor_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            config = _config()
            config.update({"model_type": "tiny", "architectures": ["TinyModel"]})
            (checkpoint / "config.json").write_text(json.dumps(config))
            shard = checkpoint / "model.safetensors"
            tensors = {
                "model.linear.weight": ("U8", [8, 96], 768),
                "model.linear.weight_scale": ("U8", [8, 4], 32),
                "model.embed_tokens.weight": ("BF16", [16, 128], 4096),
            }
            _write_safetensors(shard, tensors)
            contract = VALIDATOR.validate_checkpoint(checkpoint, None)[
                "tensor_contract_sha256"
            ]
            VALIDATOR.PROFILES["tiny"] = VALIDATOR.Profile(
                source_model="example/tiny",
                source_revision="abc123",
                model_type="tiny",
                architecture="TinyModel",
                quantized_weights=1,
                total_tensors=3,
                shards=1,
                tensor_contract_sha256=contract,
                must_stay_dense=("embed_tokens",),
            )
            self.addCleanup(VALIDATOR.PROFILES.pop, "tiny", None)
            VALIDATOR.validate_checkpoint(checkpoint, "tiny")

            tensors["model.token_embedding.weight"] = tensors.pop(
                "model.embed_tokens.weight"
            )
            _write_safetensors(shard, tensors)
            with self.assertRaisesRegex(ValueError, "tensor name/dtype/shape"):
                VALIDATOR.validate_checkpoint(checkpoint, "tiny")


if __name__ == "__main__":
    unittest.main()
