# Checkpoint format status

## Current contract

Version 0.1.1 defines an in-memory operator representation, not a persistent
model checkpoint format.

`PackedMXFP6Tensor` contains:

- `values`: contiguous CUDA `uint8` bytes containing four E3M2 values in three
  bytes;
- `scales`: contiguous CUDA `uint8` UE8M0 scales in the physical layout used by
  the SM120 kernels;
- `rows` and `k`: the logical two-dimensional matrix shape;
- optional `logical_scales`: row-major `[rows, k / 32]` UE8M0 scales retained
  by `pack_operand` for round-trip inspection.

Each scale covers 32 consecutive values along K. `pack_operand` converts
logical E3M2 codes and row-major scales into the kernel layout;
`unpack_operand` reverses that transformation. These APIs are suitable for
tests and process-local weight preparation.

## What is deliberately not specified

The physical tensors above must not yet be treated as a stable safetensors or
Hugging Face checkpoint ABI. This release does not define:

- `config.json` recognition fields or a quantization method identifier;
- tensor names for packed values and scales;
- padding and sharding rules for tensor, pipeline or expert parallelism;
- fused projection ordering and model-specific tensor mapping;
- a converter, validator or deterministic round-trip manifest;
- forward compatibility across package minor versions.

The Qwen3.5 benchmark contains a model-specific reader for local research
checkpoints. It is benchmark code, not a supported loader.

The checked-in full-serving manifests were produced with pinned research
exports and an external runtime adapter. Publishing those measurements does not
promote that physical checkpoint layout to a stable or supported serialized
ABI.

## Acceptance bar for checkpoint format v1

A future persistent format must include all of the following before a vLLM or
SGLang loader is published:

1. a versioned quantization section declaring E3M2 weights, UE8M0 scales,
   group size 32, byte order, padding and fused-projection ordering;
2. deterministic tensor-name and TP/PP/EP sharding rules;
3. a converter from a public source model or a publicly reproducible recipe;
4. a CPU-readable metadata validator that does not require an SM120 GPU;
5. conversion and load round-trip tests with checksums;
6. an explicit policy for unknown versions and unsupported platforms.

Until that contract exists, runtime adapters must fail with a clear
unsupported-checkpoint error instead of guessing a layout.
