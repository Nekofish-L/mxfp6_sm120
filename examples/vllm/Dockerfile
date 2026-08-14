ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089
FROM ${VLLM_BASE_IMAGE}

USER root
ARG LLM_COMPRESSOR_REF=37242a6a1bf6869857084d2ac7ccb22d1af7168d
ARG PYPI_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-nvrtc-dev-13-0=13.0.88-1 \
        git \
        libcusolver-dev-13-0=12.0.4.66-1 \
        libcusparse-dev-13-0=12.6.3.3-1 \
        ninja-build \
        patch \
    && rm -rf /var/lib/apt/lists/* \
    && UV_HTTP_TIMEOUT=300 uv pip install --system --no-cache \
        --index-url "${PYPI_INDEX_URL}" cmake==3.31.10

COPY . /opt/mxfp6_sm120

RUN cd /opt/mxfp6_sm120 \
    && ./scripts/build_wheel.sh \
    && python3 -m pip install --no-cache-dir --no-deps dist/mxfp6_sm120-*.whl

COPY --from=llmcompressor . /opt/llm-compressor
RUN cd /opt/llm-compressor \
    && sha256sum -c \
        /opt/mxfp6_sm120/examples/vllm/llm-compressor.sha256 \
    && python3 -m pip install --no-deps -e /opt/llm-compressor

ARG REPRODUCER_REF
RUN test -n "${REPRODUCER_REF}"

COPY examples/vllm/patches/vllm-v0.25.1-mxfp6.patch /tmp/
COPY examples/vllm/adapter/mxfp6_tp2_vllm.py /tmp/

RUN SP_DIR="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && cd "${SP_DIR}" \
    && patch --batch --forward --strip=1 \
        < /tmp/vllm-v0.25.1-mxfp6.patch \
    && install -m 0644 /tmp/mxfp6_tp2_vllm.py "${SP_DIR}/mxfp6_tp2_vllm.py" \
    && python3 -m py_compile \
        "${SP_DIR}/mxfp6_tp2_vllm.py" \
        "${SP_DIR}/vllm/model_executor/layers/quantization/utils/mxfp6_sm120_utils.py" \
        "${SP_DIR}/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py" \
        "${SP_DIR}/vllm/model_executor/models/qwen3_next.py"

COPY examples/vllm/reproduce.py /opt/mxfp6-repro/reproduce.py
RUN chmod +x /opt/mxfp6-repro/reproduce.py \
    && ln -s /opt/mxfp6-repro/reproduce.py /usr/local/bin/mxfp6-reproduce

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all \
    MXFP6_LLM_COMPRESSOR_REF=${LLM_COMPRESSOR_REF} \
    MXFP6_REPRODUCER_REF=${REPRODUCER_REF} \
    VLLM_USE_DEEP_GEMM=0

ENTRYPOINT []
