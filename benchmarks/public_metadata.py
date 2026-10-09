"""Remove machine-specific identities from metadata published with results."""
import re


_LOCAL_PATH = re.compile(
    r"(?<![\w/])/(?:data\d*|home|root|mnt|scratch|tmp|opt|usr|var)/"
    r"[^\s:\"'`<>;,\)\]\}]*"
)
_DEVICE_ID = re.compile(r",?\s*(?:uuid|pci_bus_id|pci_device_id|pci_domain_id)=[^,\)]+")
_PRIVATE_FIELDS = {
    "hostname", "host_name", "username", "user", "pid", "worker_pid",
    "uuid", "device_uuid", "gpu_uuid", "pci_bus_id", "pci_device_id",
    "pci_domain_id", "physical_gpu", "microbenchmark_gpu_index",
    "serving_gpu_index", "adapter_branch", "CUDA_VISIBLE_DEVICES",
    "api_key", "access_token", "password", "authorization", "secret",
}


def redact_local_metadata(value):
    """Preserve metrics, versions and content hashes; redact paths and IDs.

    Hashes continue to identify the original measured inputs and artifacts,
    rather than the redacted metadata representation. Raw local audit files
    should be retained separately from these publication copies.
    """
    if isinstance(value, dict):
        return {key: "<REDACTED>" if key in _PRIVATE_FIELDS else redact_local_metadata(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_local_metadata(item) for item in value]
    if isinstance(value, str):
        return _DEVICE_ID.sub("", _LOCAL_PATH.sub("<LOCAL_PATH>", value))
    return value
