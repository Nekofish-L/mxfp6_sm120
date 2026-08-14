import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = ROOT / "examples" / "vllm" / "reproduce.py"


def _load_reproduce():
    spec = importlib.util.spec_from_file_location("mxfp6_vllm_reproduce", REPRODUCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_reproduction_profiles_cover_both_published_models():
    reproduce = _load_reproduce()

    assert set(reproduce.PROFILES) == {
        "qwen3.5-27b",
        "qwen3.5-35b-a3b",
    }
    assert reproduce.PROFILES["qwen3.5-27b"]["source_model"] == (
        "Qwen/Qwen3.5-27B"
    )
    assert reproduce.PROFILES["qwen3.5-27b"]["fp8_model"] == (
        "Qwen/Qwen3.5-27B-FP8"
    )
    assert reproduce.PROFILES["qwen3.5-27b"]["source_revision"] == (
        "fc05daec18b0a78c049392ed2e771dde82bdf654"
    )
    assert reproduce.PROFILES["qwen3.5-27b"]["fp8_revision"] == (
        "97f5941bf617e31c5e237364a8602ce3f03a551a"
    )
    assert reproduce.PROFILES["qwen3.5-35b-a3b"]["source_model"] == (
        "Qwen/Qwen3.5-35B-A3B"
    )
    assert reproduce.PROFILES["qwen3.5-35b-a3b"]["fp8_model"] == (
        "Qwen/Qwen3.5-35B-A3B-FP8"
    )
    assert reproduce.PROFILES["qwen3.5-35b-a3b"]["source_revision"] == (
        "59d61f3ce65a6d9863b86d2e96597125219dc754"
    )
    assert reproduce.PROFILES["qwen3.5-35b-a3b"]["fp8_revision"] == (
        "9d1823d2dee688a6b25e77009dc727688c44936e"
    )


def test_plan_is_public_and_builds_the_runtime_from_source(capsys):
    reproduce = _load_reproduce()

    assert reproduce.main(["plan", "--profile", "qwen3.5-27b"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["container"]["base_image"].startswith(
        "vllm/vllm-openai:v0.25.1@sha256:"
    )
    assert plan["runtime"]["mxfp6_source"] == (
        "https://github.com/Nekofish-L/mxfp6_sm120.git"
    )
    assert plan["runtime"]["mxfp6_commit"] == (
        "efb3a7968ffa61d8521843600721dda3c93a5854"
    )
    assert plan["quantizer"]["source"] == (
        "https://github.com/troycheng/llm-compressor.git"
    )
    assert plan["quantizer"]["commit"] == (
        "37242a6a1bf6869857084d2ac7ccb22d1af7168d"
    )
    assert len(plan["models"]["source_revision"]) == 40
    assert len(plan["models"]["fp8_revision"]) == 40


def test_runtime_image_and_converter_are_pinned():
    dockerfile = (ROOT / "examples/vllm/Dockerfile").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text()
    readme = (ROOT / "examples/vllm/README.md").read_text()

    assert "ARG REPRODUCER_REF" in dockerfile
    assert "COPY --from=llmcompressor" not in dockerfile
    assert "COPY . /opt/mxfp6_sm120" in dockerfile
    assert "git clone https://github.com/Nekofish-L" not in dockerfile
    assert "**/.git" in dockerignore
    assert "mxfp6-community:public-v1" in readme
    assert "37242a6a1bf6869857084d2ac7ccb22d1af7168d" in readme
    assert "BUILD_TYPE=release" in readme
    assert "vllm bench serve" in readme


def test_public_launch_specs_keep_the_measured_runtime_contract():
    reproduce = _load_reproduce()

    dense = reproduce.build_launch_spec(
        "qwen3.5-27b", "/models/Qwen3.5-27B-MXFP6", "4,5", 8251
    )
    assert dense["environment"]["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert dense["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert dense["environment"]["MXFP6_AUTOTUNE"] == "off"
    assert dense["environment"]["MXFP6_SM120_REQUIRE_NATIVE"] == "1"
    assert "--max-num-batched-tokens" in dense["argv"]
    assert "--speculative-config" not in dense["argv"]

    moe = reproduce.build_launch_spec(
        "qwen3.5-35b-a3b", "/models/Qwen3.5-35B-A3B-MXFP6", "6,7", 8251
    )
    assert moe["environment"]["QWEN35_MXFP6_TP2"] == "1"
    assert moe["environment"]["QWEN35_MXFP6_B4_VECTOR_LOADS"] == "1"
    assert moe["environment"]["MXFP6_SM120_REQUIRE_NATIVE"] == "1"
    assert "--compilation-config" in moe["argv"]
    graph_config = json.loads(
        moe["argv"][moe["argv"].index("--compilation-config") + 1]
    )
    assert graph_config == {"cudagraph_capture_sizes": [1, 2, 4]}
    assert "--speculative-config" not in moe["argv"]


def test_public_launch_specs_can_start_the_official_fp8_baselines():
    reproduce = _load_reproduce()

    dense = reproduce.build_launch_spec(
        "qwen3.5-27b",
        "/models/Qwen3.5-27B-FP8",
        "4,5",
        8251,
        format_name="fp8",
    )
    assert dense["environment"] == {
        "CUDA_VISIBLE_DEVICES": "4,5",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    assert dense["argv"][dense["argv"].index("--served-model-name") + 1] == (
        "Qwen3.5-27B-FP8"
    )
    assert "--max-num-batched-tokens" in dense["argv"]

    moe = reproduce.build_launch_spec(
        "qwen3.5-35b-a3b",
        "/models/Qwen3.5-35B-A3B-FP8",
        "6,7",
        8251,
        format_name="fp8",
    )
    assert "QWEN35_MXFP6_TP2" not in moe["environment"]
    assert "QWEN35_MXFP6_B4_VECTOR_LOADS" not in moe["environment"]
    assert moe["argv"][moe["argv"].index("--served-model-name") + 1] == (
        "Qwen3.5-35B-A3B-FP8"
    )
    assert "--compilation-config" in moe["argv"]


def test_public_example_contains_no_internal_resource_reference():
    forbidden = (
        "git.yukework.com",
        "image-docker.yukework.com",
        "vllm_template_0730",
        "/data/lxy",
        "/data/models",
        "rubric-27b",
    )
    example_root = ROOT / "examples" / "vllm"
    text = "\n".join(
        path.read_text(errors="replace")
        for path in example_root.rglob("*")
        if path.is_file()
    )
    for marker in forbidden:
        assert marker not in text


def test_vllm_patch_is_limited_to_the_two_model_integration_surface():
    patch = (ROOT / "examples/vllm/patches/vllm-v0.25.1-mxfp6.patch").read_text()
    changed = {
        line.removeprefix("diff --git a/").split(" b/", 1)[0]
        for line in patch.splitlines()
        if line.startswith("diff --git a/")
    }
    assert changed == {
        "vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py",
        "vllm/model_executor/layers/quantization/utils/mxfp6_sm120_utils.py",
        "vllm/model_executor/models/qwen3_next.py",
        "vllm/model_executor/warmup/kernel_warmup.py",
        "vllm/v1/worker/gpu/cudagraph_utils.py",
    }
    assert "stream=torch.cuda.current_stream()" in patch


def test_public_runtime_fails_closed_when_native_mxfp6_is_unavailable():
    reproduce = _load_reproduce()
    patch = (ROOT / "examples/vllm/patches/vllm-v0.25.1-mxfp6.patch").read_text()

    for profile in reproduce.PROFILES:
        launch = reproduce.build_launch_spec(
            profile,
            "/models/candidate",
            "0,1",
            8251,
            format_name="mxfp6",
        )
        assert launch["environment"]["MXFP6_SM120_REQUIRE_NATIVE"] == "1"

    assert "MXFP6_SM120_REQUIRE_NATIVE" in patch
    assert "requires the native mxfp6-sm120 backend" in patch
