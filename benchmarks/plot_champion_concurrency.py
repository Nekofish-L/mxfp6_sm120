#!/usr/bin/env python3
"""Plot the Champion full-service throughput sweep from its JSON artifact."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_PANELS = (
    ("qwen3_5_27b", "Qwen3.5-27B", "3000 input / 1000 output"),
    ("qwen3_5_35b_a3b", "Qwen3.5-35B-A3B", "frozen real multimodal workload"),
)
FP8_COLOR = "#5F6875"
MXFP6_COLOR = "#2563EB"
INK = "#20242A"
GRID = "#D9DEE5"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "benchmarks/results/qwen35_champion_concurrency_tp2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs/assets/full-service-throughput.svg",
    )
    return parser.parse_args()


def repeat_ranges(blocks: list[dict]) -> dict[tuple[str, int], tuple[float, float]]:
    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    for block in blocks:
        key = block["variant"], block["concurrency"]
        samples[key].append(block["output_throughput_tokens_per_s"])
    return {key: (min(values), max(values)) for key, values in samples.items()}


def plot_panel(ax: plt.Axes, model: dict, title: str, workload: str) -> None:
    points = model["points"]
    repeats = repeat_ranges(model["blocks"])
    x = list(range(len(points)))
    concurrency = [point["concurrency"] for point in points]
    fp8 = [point["fp8"]["output_throughput_tokens_per_s"] for point in points]
    mxfp6 = [point["mxfp6"]["output_throughput_tokens_per_s"] for point in points]

    def error(series: str, means: list[float]) -> list[list[float]]:
        lower, upper = [], []
        for c, mean in zip(concurrency, means):
            minimum, maximum = repeats[series, c]
            lower.append(mean - minimum)
            upper.append(maximum - mean)
        return [lower, upper]

    ax.errorbar(
        x,
        fp8,
        yerr=error("fp8", fp8),
        color=FP8_COLOR,
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2,
        capsize=3,
        label="Official FP8",
        zorder=2,
    )
    ax.errorbar(
        x,
        mxfp6,
        yerr=error("mxfp6", mxfp6),
        color=MXFP6_COLOR,
        linestyle="-",
        marker="o",
        markerfacecolor=MXFP6_COLOR,
        markeredgewidth=1.5,
        linewidth=2.4,
        capsize=3,
        label="MXFP6",
        zorder=3,
    )

    for position, value, point in zip(x, mxfp6, points):
        gain = point["changes_percent"]["output_throughput_gain"]
        ax.annotate(
            f"+{gain:.1f}%",
            (position, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=MXFP6_COLOR,
            fontsize=8,
            fontweight="semibold",
        )

    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=20)
    ax.text(0, 1.025, workload, transform=ax.transAxes, color=FP8_COLOR, fontsize=9)
    ax.set_xticks(x, concurrency)
    ax.set_xlabel("Request concurrency", color=INK)
    ax.set_ylim(0, max(mxfp6) * 1.17)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BD")
    ax.tick_params(colors=INK)


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "mxfp6-champion-concurrency",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=False)
    for ax, (key, title, workload) in zip(axes, MODEL_PANELS):
        plot_panel(ax, data["models"][key], title, workload)
    axes[0].set_ylabel("Output throughput (tokens/s)", color=INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.955),
        frameon=False,
        ncol=2,
    )
    fig.suptitle(
        "Full-service output throughput",
        x=0.07,
        y=0.97,
        ha="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.915,
        "Internal Champion, TP2 on 2 x RTX 5090; means of two opposite-order service lifecycles",
        ha="left",
        color=FP8_COLOR,
        fontsize=9.5,
    )
    fig.text(
        0.07,
        0.02,
        "Error bars show the two-run range. Labels show MXFP6 throughput gain over FP8 at the same concurrency.",
        ha="left",
        color=FP8_COLOR,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.14, wspace=0.20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white", metadata={"Date": None})
    if args.output.suffix.lower() == ".svg":
        lines = args.output.read_text().splitlines()
        args.output.write_text("\n".join(line.rstrip() for line in lines) + "\n")


if __name__ == "__main__":
    main()
