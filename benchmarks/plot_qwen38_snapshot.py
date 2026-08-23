#!/usr/bin/env python3
"""Render the Qwen3.8-27B publication figures from checked-in JSON data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "bf16": "#98A2B3",
    "fp8": "#AAB4C4",
    "mxfp6": "#4057E3",
    "nvfp4": "#202B3C",
    "model": "#172033",
    "kv": "#5268F5",
}
LABELS = {"bf16": "BF16", "fp8": "FP8", "mxfp6": "MXFP6", "nvfp4": "NVFP4"}
INK = "#141A24"
MUTED = "#667085"
GRID = "#E5EAF1"
BACKGROUND = "#FBFCFE"


def load(path: Path):
    return json.loads(path.read_text())


def style_axis(ax, grid_axis: str):
    ax.set_facecolor(BACKGROUND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD3DF")
    ax.tick_params(colors=INK, labelsize=9.5, length=0, pad=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def save(fig, output_dir: Path, preview_dir: Path | None, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "mxfp6_sm120", "Date": None}
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor=BACKGROUND, metadata=metadata)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor=BACKGROUND, metadata={"Software": "mxfp6_sm120"})
    plt.close(fig)


def plot_quality(quality, output_dir: Path, preview_dir: Path | None):
    variants = ["fp8", "mxfp6", "nvfp4"]
    values = [quality["summaries"][v]["sample_mean_mae"] for v in variants]
    intervals = [quality["sample_mean_mae_ci95"][v] for v in variants]
    y = [2, 1, 0]
    fig, ax = plt.subplots(figsize=(8.2, 2.75), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    for row_y in (0.5, 1.5):
        ax.axhline(row_y, color=GRID, linewidth=0.8, zorder=0)
    for row_y, variant, value, interval in zip(y, variants, values, intervals):
        ax.hlines(row_y, interval[0], interval[1], color=COLORS[variant], linewidth=3.0 if variant == "mxfp6" else 2.2, zorder=2)
        ax.scatter(value, row_y, s=130 if variant == "mxfp6" else 95, marker="o" if variant != "nvfp4" else "D", color=COLORS[variant], edgecolor=BACKGROUND, linewidth=1.8, zorder=3)
        ax.text(interval[1] + 0.006, row_y, f"{value:.5f}", va="center", ha="left", color=INK, fontsize=10.5, fontweight="bold")
    ax.set_yticks(y, [LABELS[v] for v in variants])
    ax.set_xlim(0, 0.225)
    ax.set_xlabel("Sample-mean token-logprob MAE vs BF16  ·  95% CI  ·  lower is better", color=INK, fontsize=10, labelpad=9)
    style_axis(ax, "x")
    reduction = 1 - values[1] / values[2]
    ax.text(0.99, 0.92, f"MXFP6 error  {reduction * 100:.1f}% lower than NVFP4", transform=ax.transAxes, ha="right", va="top", color=COLORS["mxfp6"], fontsize=10.5, fontweight="bold")
    ax.text(0.99, 0.80, "256 samples · 10,479 target tokens", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    save(fig, output_dir, preview_dir, "qwen38-quality-fidelity")


def plot_throughput(serving, output_dir: Path, preview_dir: Path | None):
    scenarios = [
        ("interactive_c1", "1024→256 · c1"),
        ("online_c16", "1024→256 · c16"),
        ("long_c32", "3000→1000 · c32"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 3.45), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    y = [2, 1, 0]
    mx_gains = [serving["scenarios"][name]["ratios"]["mxfp6_vs_fp8_output_throughput"] * 100 for name, _ in scenarios]
    nv_gains = [serving["scenarios"][name]["ratios"]["nvfp4_vs_fp8_output_throughput"] * 100 for name, _ in scenarios]
    fp8_values = [serving["scenarios"][name]["variants"]["fp8"]["mean_output_tokens_per_s"] for name, _ in scenarios]

    for row_y, mx_gain, nv_gain in zip(y, mx_gains, nv_gains):
        ax.hlines(row_y, 0, nv_gain, color="#DCE3ED", linewidth=3.0, zorder=1)
        ax.scatter(mx_gain, row_y, s=125, marker="o", color=COLORS["mxfp6"], edgecolor=BACKGROUND, linewidth=1.8, zorder=3)
        ax.scatter(nv_gain, row_y, s=105, marker="D", color=COLORS["nvfp4"], edgecolor=BACKGROUND, linewidth=1.6, zorder=3)
        ax.text(mx_gain + 1.2, row_y + 0.14, f"+{mx_gain:.1f}%", color=COLORS["mxfp6"], fontsize=10, fontweight="bold", ha="left", va="bottom")
        ax.text(nv_gain + 1.2, row_y - 0.14, f"+{nv_gain:.1f}%", color=COLORS["nvfp4"], fontsize=10, fontweight="bold", ha="left", va="top")

    y_labels = [f"{label}\nFP8 {fp8:.1f} tok/s" for (_, label), fp8 in zip(scenarios, fp8_values)]
    ax.set_yticks(y, y_labels)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks([0, 10, 20, 30, 40, 50], ["0%", "+10%", "+20%", "+30%", "+40%", "+50%"])
    ax.set_xlabel("Output-throughput gain relative to FP8", color=INK, fontsize=10, labelpad=9)
    style_axis(ax, "x")
    ax.axvline(0, color="#AAB4C4", linewidth=1.4, zorder=1)
    ax.scatter([], [], s=90, marker="o", color=COLORS["mxfp6"], label="MXFP6")
    ax.scatter([], [], s=75, marker="D", color=COLORS["nvfp4"], label="NVFP4")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9.5, handletextpad=0.6, columnspacing=1.8)
    save(fig, output_dir, preview_dir, "qwen38-serving-throughput")


def plot_tradeoff(quality, serving, output_dir: Path, preview_dir: Path | None):
    variants = ["fp8", "mxfp6", "nvfp4"]
    scenario_names = ["interactive_c1", "online_c16", "long_c32"]
    mae = {variant: quality["summaries"][variant]["sample_mean_mae"] for variant in variants}
    mae_ci = {variant: quality["sample_mean_mae_ci95"][variant] for variant in variants}
    gains = {
        "fp8": [0.0, 0.0, 0.0],
        "mxfp6": [serving["scenarios"][name]["ratios"]["mxfp6_vs_fp8_output_throughput"] * 100 for name in scenario_names],
        "nvfp4": [serving["scenarios"][name]["ratios"]["nvfp4_vs_fp8_output_throughput"] * 100 for name in scenario_names],
    }
    gain_mean = {variant: sum(gains[variant]) / len(gains[variant]) for variant in variants}

    fig, ax = plt.subplots(figsize=(8.2, 4.35), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    style_axis(ax, "both")
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color="#AAB4C4", linewidth=1.3, linestyle=(0, (4, 4)), zorder=1)

    ax.plot([mae[v] for v in variants], [gain_mean[v] for v in variants], color="#CCD4E0", linewidth=1.7, linestyle=(0, (3, 4)), zorder=1)
    markers = {"fp8": "o", "mxfp6": "o", "nvfp4": "D"}
    sizes = {"fp8": 115, "mxfp6": 180, "nvfp4": 145}
    for variant in variants:
        x = mae[variant]
        y = gain_mean[variant]
        interval = mae_ci[variant]
        y_min, y_max = min(gains[variant]), max(gains[variant])
        ax.errorbar(
            x,
            y,
            xerr=[[x - interval[0]], [interval[1] - x]],
            yerr=[[y - y_min], [y_max - y]],
            fmt="none",
            ecolor=COLORS[variant],
            elinewidth=2.2 if variant == "mxfp6" else 1.7,
            capsize=4,
            capthick=1.5,
            alpha=0.78,
            zorder=2,
        )
        if variant != "fp8":
            ax.scatter([x] * 3, gains[variant], s=28, marker=markers[variant], color=COLORS[variant], alpha=0.30, edgecolor="none", zorder=2)
        ax.scatter(x, y, s=sizes[variant], marker=markers[variant], color=COLORS[variant], edgecolor=BACKGROUND, linewidth=2.0, zorder=4)

    ax.text(mae["fp8"] + 0.008, gain_mean["fp8"] + 2.1, "FP8\n0.057 MAE · baseline", color=MUTED, fontsize=9.7, fontweight="bold", ha="left", va="bottom")
    ax.text(mae["mxfp6"] + 0.010, gain_mean["mxfp6"] - 0.4, f"MXFP6\n0.091 MAE · +{gain_mean['mxfp6']:.1f}% avg", color=COLORS["mxfp6"], fontsize=10.2, fontweight="bold", ha="left", va="center")
    ax.text(mae["nvfp4"] - 0.009, gain_mean["nvfp4"] + 1.0, f"NVFP4\n0.177 MAE · +{gain_mean['nvfp4']:.1f}% avg", color=COLORS["nvfp4"], fontsize=9.8, fontweight="bold", ha="right", va="bottom")

    ax.set_xlim(0.035, 0.215)
    ax.set_ylim(-6, 51)
    ax.set_xticks([0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
    ax.set_yticks([0, 10, 20, 30, 40, 50], ["0%", "+10%", "+20%", "+30%", "+40%", "+50%"])
    ax.set_xlabel("Sample-mean token-logprob MAE vs BF16  ·  lower is better", color=INK, fontsize=10, labelpad=9)
    ax.set_ylabel("Output-throughput gain vs FP8  ·  higher is better", color=INK, fontsize=10, labelpad=9)
    ax.text(0.01, 0.98, "horizontal: 95% quality CI   ·   vertical: range across 3 workloads", transform=ax.transAxes, color=MUTED, fontsize=9, ha="left", va="top")
    save(fig, output_dir, preview_dir, "qwen38-quality-throughput-tradeoff")


def plot_capacity(capacity, output_dir: Path, preview_dir: Path | None):
    rows = {row["variant"]: row for row in capacity["variants"]}
    variants = ["fp8", "mxfp6", "nvfp4"]
    y = [2, 1, 0]
    fig, ax = plt.subplots(figsize=(9.2, 3.3), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    model = [rows[v]["model_loading_gib"] for v in variants]
    kv = [max(rows[v]["available_kv_cache_gib"] or 0, 0) for v in variants]
    ax.barh(y, [29.4] * 3, height=0.54, color="#EEF2F7", edgecolor="none", zorder=1)
    model_bars = ax.barh(y, model, height=0.54, color=COLORS["model"], edgecolor="none", label="Model load", zorder=2)
    ax.barh(y, kv, left=model, height=0.54, color=COLORS["kv"], edgecolor="none", label="Available KV cache", zorder=2)
    ax.set_yticks(y, [LABELS[v] for v in variants])
    ax.set_xlim(0, 31.8)
    ax.set_ylim(-0.65, 2.75)
    ax.set_xlabel("GPU memory (GiB)  ·  TP1  ·  gpu_memory_utilization = 0.90", color=INK, fontsize=10, labelpad=9)
    style_axis(ax, "x")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9.5, handlelength=1.4, columnspacing=1.8)
    for row_y, variant, model_value, kv_value, bar in zip(y, variants, model, kv, model_bars):
        ax.text(model_value / 2, row_y, f"{model_value:.2f} GiB", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
        if kv_value > 0:
            ax.text(model_value + kv_value / 2, row_y, f"{kv_value:.2f}", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
            concurrency = rows[variant]["maximum_concurrency_4096"]
            ax.text(model_value + kv_value + 0.45, row_y, f"{concurrency:.2f}× @ 4096", ha="left", va="center", color=INK, fontsize=9.5, fontweight="bold")
        else:
            ax.text(model_value + 0.45, row_y, "no KV block", ha="left", va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.text(0.995, 0.94, "BF16 checkpoint 51.77 GiB  →  OOM", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9.5, fontweight="bold")
    save(fig, output_dir, preview_dir, "qwen38-tp1-capacity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelcolor": INK,
        "text.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "qwen38-publication",
    })
    quality = load(args.results_dir / "qwen38_27b_quality_fidelity.json")
    serving = load(args.results_dir / "qwen38_27b_serving_tp2.json")
    capacity = load(args.results_dir / "qwen38_27b_capacity_tp1.json")
    plot_quality(quality, args.output_dir, args.preview_dir)
    plot_throughput(serving, args.output_dir, args.preview_dir)
    plot_tradeoff(quality, serving, args.output_dir, args.preview_dir)
    plot_capacity(capacity, args.output_dir, args.preview_dir)


if __name__ == "__main__":
    main()
