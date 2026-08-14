# Public vLLM reproduction

This directory reproduces the two Qwen3.5 configurations measured by this
project without the internally maintained vLLM environment:

- `Qwen/Qwen3.5-27B` with native dense SM120 W6A8 kernels;
- `Qwen/Qwen3.5-35B-A3B` with native dense and routed-MoE SM120 W6A8 kernels.

The runtime image pins vLLM `v0.25.1` and records the exact mxfp6_sm120
checkout used for each build. Conversion uses the pinned public llm-compressor
commit below. No internal image or source tree is required.

This is an exact two-model reproduction, not a general MXFP6 checkpoint ABI or
an assertion that unmodified vLLM already supports this package.

| Model and workload | FP8 | MXFP6 | Gain |
|---|---:|---:|---:|
| Qwen3.5-27B, 3000 input / 1000 output, c32 | 1229.90 tok/s | 1341.97 tok/s | **+9.11%** |
| Qwen3.5-35B-A3B, public `random-mm`, c4 | 678.37 tok/s | 793.33 tok/s | **+16.95%** |

The 35B-A3B P99 TTFT regressed by 21.24%. Full metrics are in the
[benchmark artifact](../../benchmarks/results/qwen35_public_vllm_tp2.json).

## 1. Build the public image

```bash
git clone https://github.com/Nekofish-L/mxfp6_sm120.git
cd mxfp6_sm120
git submodule update --init third_party/cutlass
docker build --build-arg REPRODUCER_REF="$(git rev-parse HEAD)" \
  -f examples/vllm/Dockerfile -t mxfp6-community:public-v1 .
```

Inspect the complete input plan:

```bash
docker run --rm mxfp6-community:public-v1 \
  mxfp6-reproduce plan --profile qwen3.5-27b
docker run --rm mxfp6-community:public-v1 \
  mxfp6-reproduce plan --profile qwen3.5-35b-a3b
```

## 2. Download and convert

Download the exact source and official FP8 revisions:

```bash
mkdir -p models
for spec in \
  'Qwen/Qwen3.5-27B fc05daec18b0a78c049392ed2e771dde82bdf654 Qwen3.5-27B' \
  'Qwen/Qwen3.5-27B-FP8 97f5941bf617e31c5e237364a8602ce3f03a551a Qwen3.5-27B-FP8' \
  'Qwen/Qwen3.5-35B-A3B 59d61f3ce65a6d9863b86d2e96597125219dc754 Qwen3.5-35B-A3B' \
  'Qwen/Qwen3.5-35B-A3B-FP8 9d1823d2dee688a6b25e77009dc727688c44936e Qwen3.5-35B-A3B-FP8'
do
  set -- $spec
  docker run --rm -v "$PWD/models:/models" mxfp6-community:public-v1 \
    hf download "$1" --revision "$2" --local-dir "/models/$3"
done
```

Install the pinned public converter and run its two model-specific recipes:

```bash
git clone https://github.com/troycheng/llm-compressor.git
git -C llm-compressor checkout 37242a6a1bf6869857084d2ac7ccb22d1af7168d
python3 -m venv llm-compressor/.venv
BUILD_TYPE=release llm-compressor/.venv/bin/pip install -e llm-compressor

llm-compressor/.venv/bin/python \
  llm-compressor/examples/quantization_w6a8_mxfp6/qwen35_27b_example.py \
  --model "$PWD/models/Qwen3.5-27B" \
  --output "$PWD/models/Qwen3.5-27B-MXFP6"

llm-compressor/.venv/bin/python \
  llm-compressor/examples/quantization_w6a8_mxfp6/qwen35_35b_example.py \
  --model "$PWD/models/Qwen3.5-35B-A3B" \
  --output "$PWD/models/Qwen3.5-35B-A3B-MXFP6"
```

The recipes reproduce the model-specific Quark layouts used by the published
measurements. The 27B recipe quantizes the same dense projections as the tested
checkpoint. The 35B-A3B recipe splits and quantizes the routed-expert bundles
while preserving the tested exclusions.

## 3. Start either format with the same runtime

The examples below use one PIX-connected GPU pair per TP2 service. Replace the
host model directory and GPU indices as needed.

```bash
# Dense MXFP6
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models:ro" mxfp6-community:public-v1 \
  mxfp6-reproduce serve --profile qwen3.5-27b --format mxfp6 \
  --model-path /models/Qwen3.5-27B-MXFP6 --gpus 0,1 --port 8251

# Dense official FP8 baseline
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models:ro" mxfp6-community:public-v1 \
  mxfp6-reproduce serve --profile qwen3.5-27b --format fp8 \
  --model-path /models/Qwen3.5-27B-FP8 --gpus 0,1 --port 8251

# MoE MXFP6
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models:ro" mxfp6-community:public-v1 \
  mxfp6-reproduce serve --profile qwen3.5-35b-a3b --format mxfp6 \
  --model-path /models/Qwen3.5-35B-A3B-MXFP6 --gpus 0,1 --port 8251

# MoE official FP8 baseline
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models:ro" mxfp6-community:public-v1 \
  mxfp6-reproduce serve --profile qwen3.5-35b-a3b --format fp8 \
  --model-path /models/Qwen3.5-35B-A3B-FP8 --gpus 0,1 --port 8251
```

Wait for `GET /health` to return HTTP 200 before sending requests. Both model
paths use CUDA Graph capture and TP2 in the public reproducer.

## 4. Public benchmark contracts

For publishable comparisons, run fresh service lifecycles in
FP8/MXFP6/MXFP6/FP8 order. Keep the runtime image, GPU pair, scheduler settings
and request contract identical between formats.

Create a result directory, then run the pinned client from the same image. The
27B contract is:

```bash
mkdir -p results
docker run --rm --network host -v "$PWD/models:/models:ro" \
  -v "$PWD/results:/results" -w /results mxfp6-community:public-v1 \
  vllm bench serve --backend openai \
  --base-url http://127.0.0.1:8251 --endpoint /v1/completions \
  --model Qwen3.5-27B-MXFP6 --tokenizer /models/Qwen3.5-27B-FP8 \
  --dataset-name random --random-input-len 3000 --random-output-len 1000 \
  --num-prompts 320 --request-rate 20 --max-concurrency 32 \
  --num-warmups 4 --ignore-eos --temperature 0 --seed 20260814 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 99 \
  --save-result --save-detailed
```

The public 35B-A3B contract is deterministic `random-mm`; it is not the private
workload behind the environment-bound `+13.58%` result:

```bash
docker run --rm --network host -v "$PWD/models:/models:ro" \
  -v "$PWD/results:/results" -w /results mxfp6-community:public-v1 \
  vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8251 --endpoint /v1/chat/completions \
  --model Qwen3.5-35B-A3B-MXFP6 \
  --tokenizer /models/Qwen3.5-35B-A3B-FP8 \
  --dataset-name random-mm --enable-multimodal-chat \
  --random-input-len 128 --random-output-len 512 \
  --random-mm-base-items-per-request 1 \
  --random-mm-bucket-config '{(256, 256, 1): 1.0}' \
  --num-prompts 64 --request-rate inf --max-concurrency 4 \
  --num-warmups 4 --ignore-eos --temperature 0 --seed 20260814 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 99 \
  --save-result --save-detailed
```

Always change `--model` to the served FP8 name for FP8 blocks. Retain the raw
JSON, complete server log, image digest, model revision, GPU topology and
preflight utilization for every block.

## Validation

Both converted checkpoints matched the tested safetensors shards, loaded under
TP2, captured CUDA Graphs and completed generation on RTX 5090. See the
[conversion artifact](../../benchmarks/results/qwen35_public_conversion_validation.json).
