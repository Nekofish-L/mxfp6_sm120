from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "vllm"


def test_image_inputs_are_public_and_pinned():
    dockerfile = (EXAMPLE / "Dockerfile").read_text()

    assert (
        "vllm/vllm-openai:v0.28.0@"
        "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
        in dockerfile
    )
    assert "6d5ef77f8c05cb76112c49517efcad60973384ed" in dockerfile
    assert "0dbb75cae89e1bb7cbd9c27f24cd42fc0ab78dd0" in dockerfile
    assert "dcdd78d76a7caa73ff1cc8e2f9d62817a7f73d12" in dockerfile
    assert "e6233cbac5d7c7a865c19c91cd684ceece19513c" in dockerfile
    assert "COPY csrc /opt/mxfp6_sm120/csrc" in dockerfile
    assert "flashinfer-cubin flashinfer-jit-cache" in dockerfile
    assert "VLLM_ALLREDUCE_USE_FLASHINFER=1" in dockerfile
    assert "VLLM_ENABLE_QWEN_GDN_FUSED_DECODE=1" in dockerfile
    assert "VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm" in dockerfile
    assert "vllm-pr-53645-fused-gdn.patch" in dockerfile
    assert "mxfp6_sm120_moe.py" in dockerfile
    assert "mxfp6_sm120_utils.py" in dockerfile


def test_example_has_one_build_path_and_direct_vllm_serve_commands():
    readme = (EXAMPLE / "README.md").read_text()

    assert "docker build -f examples/vllm/Dockerfile" in readme
    assert readme.count("mxfp6-vllm:0.28.0 /models/") == 2
    assert "mxfp6-reproduce" not in readme
    assert not (EXAMPLE / "reproduce.py").exists()
    assert '"cudagraph_mode":"FULL"' in readme
    assert '"cudagraph_capture_sizes":[1,2,4,8,16,24,32]' in readme
    assert '"fuse_allreduce_rms":true' in readme
    assert "--async-scheduling" in readme
    assert "--max-num-batched-tokens 4096" in readme
    assert readme.count("--no-enable-prefix-caching") == 2


def test_public_example_contains_no_internal_resource_reference():
    forbidden = (
        "git.yukework.com",
        "image-docker.yukework.com",
        "vllm_template_0730",
        "/data/lxy",
        "/data/models",
        "rubric-27b",
    )
    text = "\n".join(
        path.read_text(errors="replace")
        for path in EXAMPLE.rglob("*")
        if path.is_file()
    )
    for marker in forbidden:
        assert marker not in text


def test_vllm_patch_is_limited_to_required_integration_points():
    patch = (EXAMPLE / "patches" / "vllm-v0.28.0-mxfp6.patch").read_text()
    changed = {
        line.removeprefix("diff --git a/").split(" b/", 1)[0]
        for line in patch.splitlines()
        if line.startswith("diff --git a/")
    }
    assert changed == {
        "vllm/compilation/passes/fusion/allreduce_rms_fusion.py",
        "vllm/model_executor/kernels/linear/__init__.py",
        "vllm/model_executor/layers/quantization/quark/quark_moe.py",
        "vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py",
        "vllm/model_executor/warmup/kernel_warmup.py",
        "vllm/v1/worker/gpu/cudagraph_utils.py",
    }
    assert "Mxfp6Sm120LinearKernel" in patch
    assert '"mxfp8_e4m3": kMxfp8Dynamic' in patch
    assert "make_mxfp6_sm120_moe_kernel" in patch
    assert "try_enable_qwen35_moe_small_batch" in patch
    assert "warmup_mxfp6_sm120_stream" in patch
    assert "stream=torch.cuda.current_stream()" in patch
    assert "120: {\n+        2: 64" in patch
    assert "120: {\n+        2: 32" in patch

    gdn_patch = (
        EXAMPLE / "patches" / "vllm-pr-53645-fused-gdn.patch"
    ).read_text()
    assert "fbd402ef87ad4f0a79a8d18ad17ccc70e1c10a3b" in gdn_patch
    assert "qwen_gdn_attention_core_fi" in gdn_patch
    assert "VLLM_ENABLE_QWEN_GDN_FUSED_DECODE" in gdn_patch


def test_adapter_fails_closed_on_the_tested_dense_and_moe_boundaries():
    adapter = (EXAMPLE / "adapter" / "mxfp6_tp2_vllm.py").read_text()
    moe = (EXAMPLE / "adapter" / "mxfp6_sm120_moe.py").read_text()

    assert "requires an SM120 GPU" in adapter
    assert "requires static MXFP6 E3M2 weights" in adapter
    assert "requires dynamic MXFP8 E4M3 activations" in adapter
    assert "moe.tp_size != 2" in moe
    assert "moe.hidden_dim != 2048" in moe
    assert "moe.num_experts != 256" in moe
    assert "_QWEN35_SMALL_BATCH_MAX_TOKENS = 4" in moe
    assert "_QWEN35_GROUPED_MIN_TOKENS = 5" in moe
    assert "use_packed_vector_loads=padded_tokens == 4" in moe
