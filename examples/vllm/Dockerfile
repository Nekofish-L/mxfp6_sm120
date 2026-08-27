ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.28.0@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14

FROM ${VLLM_BASE_IMAGE} AS builder

USER root
ARG CUTLASS_REF=e6233cbac5d7c7a865c19c91cd684ceece19513c
ARG FLASHINFER_REF=6d5ef77f8c05cb76112c49517efcad60973384ed
ARG FLASHINFER_LOCAL_IPC_FIX=0dbb75cae89e1bb7cbd9c27f24cd42fc0ab78dd0
ARG FLASHINFER_QWEN35_GDN=dcdd78d76a7caa73ff1cc8e2f9d62817a7f73d12

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake cuda-nvrtc-dev-13-0 git libcusolver-dev-13-0 \
        libcusparse-dev-13-0 ninja-build patch \
    && rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt CHANGELOG.md CONTRIBUTING.md LICENSE MANIFEST.in README.md \
    THIRD_PARTY_NOTICES.md pyproject.toml setup.py /opt/mxfp6_sm120/
COPY csrc /opt/mxfp6_sm120/csrc
COPY patches /opt/mxfp6_sm120/patches
COPY python /opt/mxfp6_sm120/python
COPY scripts /opt/mxfp6_sm120/scripts

RUN mkdir -p /opt/mxfp6_sm120/third_party/cutlass \
    && git -C /opt/mxfp6_sm120/third_party/cutlass init \
    && git -C /opt/mxfp6_sm120/third_party/cutlass remote add origin \
        https://github.com/NVIDIA/cutlass.git \
    && git -C /opt/mxfp6_sm120/third_party/cutlass fetch --depth 1 origin \
        "${CUTLASS_REF}" \
    && git -C /opt/mxfp6_sm120/third_party/cutlass checkout --detach FETCH_HEAD \
    && cd /opt/mxfp6_sm120 \
    && ./scripts/build_wheel.sh

RUN git clone https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer \
    && git -C /tmp/flashinfer fetch origin \
        refs/pull/4634/head \
        refs/pull/4698/head \
    && git -C /tmp/flashinfer checkout --detach "${FLASHINFER_REF}" \
    && git -C /tmp/flashinfer config user.name mxfp6-build \
    && git -C /tmp/flashinfer config user.email mxfp6-build@localhost \
    && git -C /tmp/flashinfer cherry-pick \
        "${FLASHINFER_LOCAL_IPC_FIX}" \
        "${FLASHINFER_QWEN35_GDN}" \
    && git -C /tmp/flashinfer submodule update --init \
        3rdparty/cccl 3rdparty/cutlass 3rdparty/spdlog \
    && cd /tmp/flashinfer \
    && BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 \
        python3 -m pip wheel . --no-build-isolation --no-deps \
        --wheel-dir /tmp/wheels

FROM ${VLLM_BASE_IMAGE}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/mxfp6_sm120/dist/mxfp6_sm120-*.whl /tmp/
COPY --from=builder /tmp/wheels/flashinfer_python-*.whl /tmp/
COPY examples/vllm/adapter/mxfp6_sm120_moe.py /tmp/
COPY examples/vllm/adapter/mxfp6_sm120_utils.py /tmp/
COPY examples/vllm/adapter/mxfp6_tp2_vllm.py /tmp/
COPY examples/vllm/patches/vllm-v0.28.0-mxfp6.patch /tmp/
COPY examples/vllm/patches/vllm-pr-53645-fused-gdn.patch /tmp/

RUN python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
        /tmp/flashinfer_python-*.whl \
        /tmp/mxfp6_sm120-*.whl \
    && SP_DIR="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && install -m 0644 /tmp/mxfp6_sm120_moe.py "${SP_DIR}/mxfp6_sm120_moe.py" \
    && install -m 0644 /tmp/mxfp6_sm120_utils.py "${SP_DIR}/mxfp6_sm120_utils.py" \
    && install -m 0644 /tmp/mxfp6_tp2_vllm.py "${SP_DIR}/mxfp6_tp2_vllm.py" \
    && cd "${SP_DIR}" \
    && patch --batch --forward --strip=1 --fuzz=0 \
        < /tmp/vllm-v0.28.0-mxfp6.patch \
    && patch --batch --forward --strip=1 --fuzz=0 \
        < /tmp/vllm-pr-53645-fused-gdn.patch \
    && python3 -m py_compile \
        "${SP_DIR}/mxfp6_sm120_moe.py" \
        "${SP_DIR}/mxfp6_sm120_utils.py" \
        "${SP_DIR}/mxfp6_tp2_vllm.py" \
        "${SP_DIR}/vllm/config/compilation.py" \
        "${SP_DIR}/vllm/compilation/passes/fusion/allreduce_rms_fusion.py" \
        "${SP_DIR}/vllm/envs.py" \
        "${SP_DIR}/vllm/model_executor/kernels/linear/__init__.py" \
        "${SP_DIR}/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py" \
        "${SP_DIR}/vllm/model_executor/layers/quantization/quark/quark_moe.py" \
        "${SP_DIR}/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py" \
    && grep -Fq 'qwen_gdn_attention_core_fi' \
        "${SP_DIR}/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py" \
    && python3 -c 'import inspect, json; from importlib.metadata import version; from importlib.resources import files; from flashinfer.comm.mnnvl import SymmDeviceMemory; assert version("vllm") == "0.28.0"; assert version("flashinfer-python") == "0.6.18"; assert "gpu_direct_rdma_capable" in inspect.signature(SymmDeviceMemory).parameters; registry = json.loads(files("flashinfer.gdn_kernels.experimental").joinpath("gdn_fused_decode_registry.json").read_text()); rows = {(r["b"], r["hidden"], r["n_ba"], r["qkv_dim"], r["h_q"], r["hv"]) for r in registry["workloads"]}; assert (16, 5120, 48, 5120, 8, 24) in rows; assert (32, 2048, 32, 4096, 8, 16) in rows' \
    && rm -f /tmp/*.whl /tmp/*.patch /tmp/mxfp6_sm120_*.py \
        /tmp/mxfp6_tp2_vllm.py

ENV FLASHINFER_CUDA_ARCH_LIST=12.0f \
    MXFP6_AUTOTUNE=off \
    MXFP6_SM120_REQUIRE_NATIVE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all \
    VLLM_ALLREDUCE_USE_FLASHINFER=1 \
    VLLM_ENABLE_QWEN_GDN_FUSED_DECODE=1 \
    VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_USE_V2_MODEL_RUNNER=1
