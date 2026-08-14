#!/usr/bin/env python3
"""Build a public reproduction plan for the two measured Qwen3.5 profiles."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


VLLM_BASE_IMAGE = (
    "vllm/vllm-openai:v0.25.1@"
    "sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089"
)
MXFP6_SOURCE = "https://github.com/Nekofish-L/mxfp6_sm120.git"
MXFP6_REF = "efb3a7968ffa61d8521843600721dda3c93a5854"
LLM_COMPRESSOR_SOURCE = "https://github.com/troycheng/llm-compressor.git"
LLM_COMPRESSOR_REF = "37242a6a1bf6869857084d2ac7ccb22d1af7168d"

PROFILES: dict[str, dict[str, Any]] = {
    "qwen3.5-27b": {
        "source_model": "Qwen/Qwen3.5-27B",
        "source_revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "fp8_model": "Qwen/Qwen3.5-27B-FP8",
        "fp8_revision": "97f5941bf617e31c5e237364a8602ce3f03a551a",
        "mxfp6_output": "Qwen3.5-27B-MXFP6",
        "quantization_recipe": "qwen35_27b_example.py",
        "served_names": {
            "fp8": "Qwen3.5-27B-FP8",
            "mxfp6": "Qwen3.5-27B-MXFP6",
        },
        "max_num_batched_tokens": 4096,
    },
    "qwen3.5-35b-a3b": {
        "source_model": "Qwen/Qwen3.5-35B-A3B",
        "source_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "fp8_model": "Qwen/Qwen3.5-35B-A3B-FP8",
        "fp8_revision": "9d1823d2dee688a6b25e77009dc727688c44936e",
        "mxfp6_output": "Qwen3.5-35B-A3B-MXFP6",
        "quantization_recipe": "qwen35_35b_example.py",
        "served_names": {
            "fp8": "Qwen3.5-35B-A3B-FP8",
            "mxfp6": "Qwen3.5-35B-A3B-MXFP6",
        },
        "cuda_graph_capture_sizes": [1, 2, 4],
    },
}


def build_plan(profile_name: str) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    return {
        "profile": profile_name,
        "models": {
            "source": profile["source_model"],
            "source_revision": profile["source_revision"],
            "fp8_baseline": profile["fp8_model"],
            "fp8_revision": profile["fp8_revision"],
            "mxfp6_output": profile["mxfp6_output"],
        },
        "quantizer": {
            "source": LLM_COMPRESSOR_SOURCE,
            "commit": LLM_COMPRESSOR_REF,
            "recipe": (
                "examples/quantization_w6a8_mxfp6/"
                + profile["quantization_recipe"]
            ),
        },
        "runtime": {
            "mxfp6_source": MXFP6_SOURCE,
            "mxfp6_commit": MXFP6_REF,
            "reproducer_commit": os.environ.get(
                "MXFP6_REPRODUCER_REF", "uncommitted"
            ),
            "build": "./scripts/build_wheel.sh",
        },
        "container": {
            "base_image": VLLM_BASE_IMAGE,
            "dockerfile": "examples/vllm/Dockerfile",
        },
    }


def build_launch_spec(
    profile_name: str,
    model_path: str,
    gpu_pair: str,
    port: int,
    *,
    format_name: str = "mxfp6",
) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    if format_name not in ("fp8", "mxfp6"):
        raise ValueError(f"unsupported format: {format_name}")
    environment = {
        "CUDA_VISIBLE_DEVICES": gpu_pair,
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    if format_name == "mxfp6":
        environment["MXFP6_SM120_REQUIRE_NATIVE"] = "1"
    argv = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "64",
        "--gpu-memory-utilization",
        "0.9",
        "--served-model-name",
        profile["served_names"][format_name],
        "--attention-backend",
        "FLASHINFER",
    ]
    if profile_name == "qwen3.5-27b":
        if format_name == "mxfp6":
            environment["MXFP6_AUTOTUNE"] = "off"
        argv.extend(
            [
                "--max-num-batched-tokens",
                str(profile["max_num_batched_tokens"]),
                "--reasoning-parser",
                "qwen3",
            ]
        )
    else:
        if format_name == "mxfp6":
            environment["QWEN35_MXFP6_TP2"] = "1"
            environment["QWEN35_MXFP6_B4_VECTOR_LOADS"] = "1"
        argv.extend(
            [
                "--compilation-config",
                json.dumps(
                    {
                        "cudagraph_capture_sizes": profile[
                            "cuda_graph_capture_sizes"
                        ]
                    },
                    separators=(",", ":"),
                ),
            ]
        )
    return {"profile": profile_name, "environment": environment, "argv": argv}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print the frozen public inputs")
    plan.add_argument("--profile", required=True, choices=sorted(PROFILES))
    serve = commands.add_parser("serve", help="start a public MXFP6 profile")
    serve.add_argument("--profile", required=True, choices=sorted(PROFILES))
    serve.add_argument("--format", required=True, choices=("fp8", "mxfp6"))
    serve.add_argument("--model-path", required=True)
    serve.add_argument("--gpus", required=True)
    serve.add_argument("--port", type=int, default=8251)
    serve.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps(build_plan(args.profile), indent=2, sort_keys=True))
        return 0
    if args.command == "serve":
        launch = build_launch_spec(
            args.profile,
            args.model_path,
            args.gpus,
            args.port,
            format_name=args.format,
        )
        if args.dry_run:
            print(json.dumps(launch, indent=2, sort_keys=True))
            return 0
        environment = os.environ.copy()
        environment.update(launch["environment"])
        os.execvpe(launch["argv"][0], launch["argv"], environment)
        raise AssertionError("unreachable")
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
