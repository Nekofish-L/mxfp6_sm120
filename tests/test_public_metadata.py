"""Publication copies must preserve measurements while removing local IDs."""
import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "benchmarks" / "public_metadata.py"
spec = importlib.util.spec_from_file_location("public_metadata", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_redaction_preserves_measurements_and_original_input():
    original = {
        "latency_us": 12.345,
        "source_sha256": "a" * 64,
        "torch": "2.13.0+cu130",
        "url": "https://github.com/NVIDIA/cutlass",
        "device": "GPU(name='Example', uuid=example-id, pci_bus_id=7, L2_cache_size=96MB)",
        "environment": {"hostname": "example-host", "physical_gpu": 7,
                        "PYTHONPATH": "/home/example/project:/tmp/example/site"},
        "runs": [{"checkpoint": "/mnt/example/model.safetensors", "relative": "csrc/gemm.cu"}],
    }
    redacted = module.redact_local_metadata(original)
    assert redacted["latency_us"] == original["latency_us"]
    assert redacted["source_sha256"] == original["source_sha256"]
    assert redacted["torch"] == original["torch"]
    assert redacted["url"] == original["url"]
    assert redacted["device"] == "GPU(name='Example', L2_cache_size=96MB)"
    assert redacted["environment"] == {"hostname": "<REDACTED>", "physical_gpu": "<REDACTED>",
                                       "PYTHONPATH": "<LOCAL_PATH>:<LOCAL_PATH>"}
    assert redacted["runs"] == [{"checkpoint": "<LOCAL_PATH>", "relative": "csrc/gemm.cu"}]
    assert original["environment"]["hostname"] == "example-host"
    assert module.redact_local_metadata(redacted) == redacted
