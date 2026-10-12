# vLLM reproduction

This image runs the two MXFP6 checkpoints validated by this project on two
RTX 5090 GPUs:

- Qwen3.5-27B Dense;
- Qwen3.5-35B-A3B routed MoE.

The image pins vLLM `v0.28.0` and the FlashInfer/vLLM GDN and local-AllReduce patches in [Dockerfile](Dockerfile). It builds the checked-out native library and adapter from public source; checkpoints are mounted at runtime.

The adapter selects the native Dense kernel and the validated Qwen3.5-35B-A3B
TP2 MoE schedules directly from Quark checkpoint metadata. Unsupported GPU,
quantization, TP or expert geometry fails closed.

## Build

From a normal clone of this repository, run one command:

```bash
docker build -f examples/vllm/Dockerfile -t mxfp6-vllm:0.28.0 .
```

The Docker build fetches the pinned CUTLASS source itself; initializing the
repository submodule is not required.

## Checkpoints

Set the host directory containing the converted checkpoints. The default below
uses `models/` in the repository checkout:

```bash
export MXFP6_MODELS_DIR="$PWD/models"
```

The commands below expect these tested Quark/OCP-MX layouts under that directory:

```text
models/Qwen3.5-27B-MXFP6
models/Qwen3.5-35B-A3B-MXFP6
```

Follow the [checkpoint conversion guide](../../docs/quantize-checkpoints.md) to create and validate either tested Quark layout before starting vLLM.

The adapter rejects incompatible TP size, expert geometry, checkpoint layout,
or GPU architecture instead of silently using a different execution path.

## Serve Qwen3.5-27B

```bash
docker run --rm --name mxfp6-q27 --gpus all --ipc=host --network host \
  -e CUDA_VISIBLE_DEVICES=0,1 -v "$MXFP6_MODELS_DIR:/models:ro" \
  -v mxfp6-vllm-cache:/root/.cache \
  mxfp6-vllm:0.28.0 /models/Qwen3.5-27B-MXFP6 \
  --served-model-name Qwen3.5-27B-MXFP6 --tensor-parallel-size 2 \
  --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching --attention-backend FLASHINFER --async-scheduling \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,24,32],"pass_config":{"fuse_allreduce_rms":true}}'
```

## Serve Qwen3.5-35B-A3B

```bash
docker run --rm --name mxfp6-q35 --gpus all --ipc=host --network host \
  -e CUDA_VISIBLE_DEVICES=0,1 -v "$MXFP6_MODELS_DIR:/models:ro" \
  -v mxfp6-vllm-cache:/root/.cache \
  mxfp6-vllm:0.28.0 /models/Qwen3.5-35B-A3B-MXFP6 \
  --served-model-name Qwen3.5-35B-A3B-MXFP6 --tensor-parallel-size 2 \
  --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching --attention-backend FLASHINFER --async-scheduling \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,24,32],"pass_config":{"fuse_allreduce_rms":true}}'
```

Both commands preserve the measured TP2 execution profile: the listed CUDA
Graph buckets through batch 32, fused GDN where a validated registry row exists,
FlashInfer TRT-LLM AllReduce/RMSNorm on local consumer Blackwell, and the
current Dense/MoE MXFP6 dispatch.

Wait for `GET /health` to return HTTP 200 before sending requests. A successful
startup log should show `Mxfp6Sm120LinearKernel`, `Using FlashInfer GDN prefill
kernel`, `Using FlashInfer fused GDN decode step when supported`, `Using native
mxfp6-sm120 grouped MoE backend` for 35B-A3B, and a FlashInfer AllReduce
workspace using the `trtllm` backend.
Treat a fallback to MXFP6 emulation or a failed request as a failed
reproduction. vLLM uses its `FULL_AND_PIECEWISE` handling for GDN and captures
both graph sets at the listed sizes.

The first start compiles the SM120 Torch and FlashInfer extensions and can take
several minutes. The named cache volume makes later starts reuse those artifacts.

## Validation

The [v0.28 integration receipt](../../benchmarks/results/vllm_v028_public_reproducer_validation.json) records tested TP2 loads, native execution and request contracts. Use [vLLM Mach](https://github.com/troycheng/vllm-mach) for current serving profiles and [model performance](../../docs/performance.md) for the current results.
