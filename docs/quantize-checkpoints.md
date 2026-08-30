# Quantize the validated Qwen checkpoints

Community users can create the two Qwen3.5 MXFP6 checkpoints used by the public
vLLM reproducer from the original Hugging Face weights. The conversion is
data-free: it quantizes model weights directly and does not require a calibration
dataset.

The recipes are model-scoped. They reproduce the tested Quark/OCP-MX layouts for
Qwen3.5-27B and Qwen3.5-35B-A3B; they are not a general MXFP6 checkpoint format.
There is currently no public Qwen3.8-27B recipe.

## Install the pinned converter

The MXFP6 converter is currently maintained at a pinned public
`llm-compressor` commit rather than a released package. Install that exact source
revision so the recipe, tensor mapping and emitted Quark metadata stay aligned:

```bash
git clone https://github.com/troycheng/llm-compressor.git
cd llm-compressor
git checkout 37242a6a1bf6869857084d2ac7ccb22d1af7168d
python3 -m pip install -e .

export MXFP6_MODELS_DIR=/path/to/models
mkdir -p "$MXFP6_MODELS_DIR"
```

Keep the source model and output directory separate. Both coexist during
conversion, so reserve disk space for both checkpoints. The default
`--max-workers 1` limits concurrent shard memory. Increase it only when every
worker's source shard and conversion temporaries fit on the selected device.

An SM120 GPU is not required to create the checkpoint. The converter has a
portable PyTorch implementation and accepts `--device cpu`, although CPU
conversion is slower. On SM120 with `mxfp6-sm120` installed, `--device cuda:0`
can use the native quantizer. Both paths target the same logical checkpoint
layout; validate the generated output as described below.

## Qwen3.5-27B Dense

From the checked-out `llm-compressor` repository:

```bash
python3 examples/quantization_w6a8_mxfp6/qwen35_27b_example.py \
  --model Qwen/Qwen3.5-27B \
  --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --output "$MXFP6_MODELS_DIR/Qwen3.5-27B-MXFP6" \
  --device cuda:0 --max-workers 1
```

This recipe converts 408 tested Dense projection weights. Embeddings, `lm_head`,
GDN `in_proj_a`/`in_proj_b`, MTP embeddings and visual modules remain dense.

## Qwen3.5-35B-A3B routed MoE

```bash
python3 examples/quantization_w6a8_mxfp6/qwen35_35b_example.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --output "$MXFP6_MODELS_DIR/Qwen3.5-35B-A3B-MXFP6" \
  --device cuda:0 --max-workers 1
```

The source checkpoint stores routed experts as fused three-dimensional tensors.
The recipe splits them into per-expert gate, up and down projections before
packing 31,746 weights. Embeddings, `lm_head`, router weights, the scalar
shared-expert gate, GDN `in_proj_a`/`in_proj_b` and visual modules remain dense.

## Validate without loading tensor payloads

Return to the `mxfp6_sm120` checkout and run the metadata validator. It uses only
the Python standard library and reads safetensors headers rather than weight
payloads, so validation does not require a GPU or enough RAM to load the model.

```bash
python3 examples/vllm/validate_mxfp6_checkpoint.py \
  "$MXFP6_MODELS_DIR/Qwen3.5-27B-MXFP6" --profile qwen3.5-27b

python3 examples/vllm/validate_mxfp6_checkpoint.py \
  "$MXFP6_MODELS_DIR/Qwen3.5-35B-A3B-MXFP6" --profile qwen3.5-35b-a3b
```

The validator checks:

- Quark weight and dynamic-activation metadata in `config.json`;
- packed E3M2 weight and UE8M0 scale dtype, shape and one-to-one pairing;
- model architecture and the complete tensor name/dtype/shape contract;
- safetensors data offsets, encoded byte sizes and payload coverage;
- the model-specific tensor and shard counts;
- that explicitly excluded modules did not become packed U8 weights.

The pinned recipes have these exact expected counts:

| Profile | Packed MXFP6 weights | Total tensors | Safetensors shards |
|---|---:|---:|---:|
| `qwen3.5-27b` | 408 | 1,607 | 11 |
| `qwen3.5-35b-a3b` | 31,746 | 64,197 | 14 |

For a durable local conversion record, add `--write-manifest`:

```bash
python3 examples/vllm/validate_mxfp6_checkpoint.py \
  "$MXFP6_MODELS_DIR/Qwen3.5-27B-MXFP6" --profile qwen3.5-27b \
  --write-manifest "$MXFP6_MODELS_DIR/Qwen3.5-27B-MXFP6.manifest.json"
```

Hashing reads every shard and therefore takes longer than the default metadata
validation. The manifest records the pinned converter and expected recipe source
revision plus the size and SHA-256 digest of `config.json` and every safetensors
shard. It is a local provenance record; the validator cannot infer the actual
source revision from the converted files, and the manifest does not prove
numerical quality by itself.

## Verify native execution in vLLM

Checkpoint validation proves the serialized layout, not the runtime path. Build
the [version-locked vLLM image](../examples/vllm/README.md), mount the validated
checkpoint and wait for `GET /health` to return HTTP 200. In another shell, set
the model and container names used by the launch command and send a deterministic
request:

```bash
# Qwen3.5-27B:
export MXFP6_CONTAINER=mxfp6-q27
export MXFP6_SERVED_MODEL=Qwen3.5-27B-MXFP6

# For Qwen3.5-35B-A3B, use this pair instead:
# export MXFP6_CONTAINER=mxfp6-q35
# export MXFP6_SERVED_MODEL=Qwen3.5-35B-A3B-MXFP6

set -o pipefail
curl -sS --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MXFP6_SERVED_MODEL\",\"prompt\":\"The capital of France is\",\"temperature\":0,\"max_tokens\":8}" \
  | python3 -c '
import json
import sys

response = json.load(sys.stdin)
choices = response.get("choices")
completion_tokens = (response.get("usage") or {}).get("completion_tokens", 0)
if not isinstance(choices, list) or not choices or completion_tokens <= 0:
    raise SystemExit(f"invalid completion response: {response}")
print(json.dumps(response, indent=2))
'

docker logs "$MXFP6_CONTAINER" 2>&1 | grep -F Mxfp6Sm120LinearKernel
```

The request must return HTTP 200 with a non-empty `choices` array and positive
`usage.completion_tokens`. Qwen3.5-35B-A3B must also pass:

```bash
docker logs "$MXFP6_CONTAINER" 2>&1 \
  | grep -F 'Using native mxfp6-sm120 grouped MoE backend'
```

A fallback to MXFP6 emulation, unsupported-layout error, failed CUDA Graph
capture or failed generation request is not a successful native reproduction.

## Current scope

These instructions intentionally cover only the two layouts that were converted
and served end to end. Qwen3.8-27B has published project measurements but does
not yet have a public conversion recipe, so it is not included in this community
checkpoint workflow.
