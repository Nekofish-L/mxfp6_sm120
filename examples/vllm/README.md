# vLLM reproduction

This image runs the two MXFP6 checkpoints validated by this project on two
RTX 5090 GPUs:

- Qwen3.5-27B Dense;
- Qwen3.5-35B-A3B routed MoE.

It starts from the official vLLM `v0.28.0` image and builds every added
component from public source. The image contains the current mxfp6-sm120
checkout, the small vLLM adapter in this directory, and the following pinned
upstream changes used by the measured configuration:

- [FlashInfer #4634](https://github.com/flashinfer-ai/flashinfer/pull/4634),
  which permits local TRT-LLM IPC allocation without GPUDirect RDMA;
- [FlashInfer #4698](https://github.com/flashinfer-ai/flashinfer/pull/4698),
  which registers the validated Qwen3.5 TP2 fused-GDN geometries;
- [vLLM #50862](https://github.com/vllm-project/vllm/pull/50862), commit
  `f5d77dc04e2f61e21c5e6b3d48d2cc1b92d24616`, which enables the existing
  FlashInfer GDN prefill path on SM120;
- [vLLM #53645](https://github.com/vllm-project/vllm/pull/53645), commit
  `fbd402ef87ad4f0a79a8d18ad17ccc70e1c10a3b`, which connects supported plain
  decode steps to FlashInfer's fused GDN operation.

The exact vLLM image digest and FlashInfer commits are pinned in the
[Dockerfile](Dockerfile). Checkpoints are mounted at runtime and are not copied
into the image.

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

The public conversion recipes are maintained in
[`troycheng/llm-compressor`](https://github.com/troycheng/llm-compressor/tree/37242a6a1bf6869857084d2ac7ccb22d1af7168d/examples/quantization_w6a8_mxfp6).
The 27B recipe converts the tested Dense projections; the 35B-A3B recipe
converts the tested routed-expert bundles. Follow the complete
[checkpoint quantization guide](../../docs/quantize-checkpoints.md) to install
the pinned converter, create either checkpoint and validate its metadata before
starting vLLM. The guide also documents the current model scope: Qwen3.8-27B
does not yet have a public conversion recipe.

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

## Performance comparisons

The public runtime was checked on a dual-RTX-5090 TP2 host against the same
frozen MXFP6 request contracts used by the project. Before the SM120 GDN
prefill backport, Qwen3.5-27B reached 355.02 tok/s at c4 and 1342.33 tok/s at
c32; Qwen3.5-35B-A3B reached 788.11 tok/s at c4. With the backport enabled,
35B-A3B reached 2011.82 tok/s at c32, within 1.53% of the comparable internal
runtime without hybrid lm_head quantization.

The full internal 35B-A3B Champion also uses a separate hybrid NVFP4 lm_head
optimization. That optimization is not part of this MXFP6 template and its
additional gain is not attributed to MXFP6.

For format comparisons, run FP8 and MXFP6 in separate service lifecycles on the
same GPU pair and keep every setting and request token contract fixed. The
project-level [performance report](../../docs/benchmarks.md) records
the full internally optimized concurrency sweeps; this image is the minimal
public MXFP6 execution path, not a byte-identical build of that runtime.
