"""Render single-run, resolution, or generation-length latency reports."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


STAGES = (
    ("image_preprocessing", "Image preprocessing", "#4C78A8"),
    ("vision_encoder", "Vision encoder", "#F58518"),
    ("multimodal_projection", "Multimodal projection", "#54A24B"),
    ("llm_prefill", "LLM prefill", "#E45756"),
    ("token_generation", "Token generation", "#B279A2"),
)
SHARED_COMPARISON_FIELDS = (
    "model",
    "task",
    "device",
    "dtype",
    "attention_backend",
    "prompt",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help=(
            "JSON produced by pipeline_profile.py. Repeat --input for a "
            "resolution or generation-length comparison."
        ),
    )
    parser.add_argument(
        "--comparison",
        choices=("auto", "resolution", "generation"),
        default="auto",
        help=(
            "Comparison type for multiple inputs. 'auto' detects whether "
            "max_image_side or max_new_tokens changes (default: auto)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path. Defaults to pipeline_stage_latency.png for "
            "one input, otherwise a comparison-specific filename."
        ),
    )
    return parser.parse_args()


def validate_profile(data, path):
    summary = data.get("summary")
    images = data.get("images")

    if not isinstance(summary, dict):
        raise ValueError(f"{path}: missing summary")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{path}: missing measured images")
    if summary.get("count") != len(images):
        raise ValueError(f"{path}: summary count does not match image count")

    stage_summary = summary.get("stages")
    if not isinstance(stage_summary, dict):
        raise ValueError(f"{path}: missing stage statistics")

    for key, _, _ in STAGES:
        if key not in stage_summary:
            raise ValueError(f"{path}: missing stage statistics for {key}")
        values = stage_summary[key]
        for field in (
            "mean_seconds",
            "median_seconds",
            "p90_seconds",
            "fraction_percent",
        ):
            value = values.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: invalid {key}.{field}")
        if values["mean_seconds"] <= 0:
            raise ValueError(f"{path}: non-positive mean for {key}")

    fraction_sum = sum(
        stage_summary[key]["fraction_percent"]
        for key, _, _ in STAGES
    )
    if abs(fraction_sum - 100.0) > 1e-3:
        raise ValueError(f"{path}: stage fractions sum to {fraction_sum:.6f}%")

    mean_sum = sum(
        stage_summary[key]["mean_seconds"]
        for key, _, _ in STAGES
    )
    if abs(mean_sum - summary["mean_seconds"]) > 1e-6:
        raise ValueError(f"{path}: stage means do not match total mean")

    max_image_side = data.get("max_image_side")
    if not isinstance(max_image_side, int) or max_image_side < 0:
        raise ValueError(f"{path}: invalid max_image_side")

    max_new_tokens = data.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError(f"{path}: invalid max_new_tokens")


def load_profiles(paths):
    profiles = []
    for path in paths:
        data = json.loads(path.read_text())
        validate_profile(data, path)
        profiles.append({"path": path, "data": data})
    return profiles


def image_identity(data):
    return tuple(Path(item["image"]).name for item in data["images"])


def resolution_label(data):
    value = data["max_image_side"]
    return "Original" if value == 0 else f"{value} px"


def resolution_sort_key(profile):
    value = profile["data"]["max_image_side"]
    return (value == 0, value)


def generation_label(data):
    return f"{data['max_new_tokens']} max"


def generation_sort_key(profile):
    return profile["data"]["max_new_tokens"]


def validate_shared_comparison(profiles):
    reference = profiles[0]
    reference_data = reference["data"]
    reference_images = image_identity(reference_data)

    for profile in profiles:
        data = profile["data"]
        path = profile["path"]
        for field in SHARED_COMPARISON_FIELDS:
            if data.get(field) != reference_data.get(field):
                raise ValueError(
                    f"{path}: {field} differs from {reference['path']}"
                )
        if image_identity(data) != reference_images:
            raise ValueError(f"{path}: measured image set or order differs")


def infer_comparison(profiles):
    resolutions = {profile["data"]["max_image_side"] for profile in profiles}
    token_limits = {profile["data"]["max_new_tokens"] for profile in profiles}

    if len(resolutions) > 1 and len(token_limits) == 1:
        return "resolution"
    if len(resolutions) == 1 and len(token_limits) > 1:
        return "generation"
    raise ValueError(
        "Cannot infer comparison type: vary exactly one of max_image_side "
        "or max_new_tokens, or pass --comparison explicitly."
    )


def validate_comparison(profiles, comparison):
    validate_shared_comparison(profiles)
    reference = profiles[0]
    reference_data = reference["data"]

    if comparison == "resolution":
        if any(
            profile["data"]["max_new_tokens"]
            != reference_data["max_new_tokens"]
            for profile in profiles
        ):
            raise ValueError(
                "Resolution comparison requires the same max_new_tokens "
                "for every input."
            )
        values = set()
        field = "max_image_side"
        label = "resolution setting"
    else:
        if any(
            profile["data"]["max_image_side"]
            != reference_data["max_image_side"]
            for profile in profiles
        ):
            raise ValueError(
                "Generation comparison requires the same max_image_side "
                "for every input."
            )
        values = set()
        field = "max_new_tokens"
        label = "generation-length setting"

    for profile in profiles:
        value = profile["data"][field]
        if value in values:
            raise ValueError(f"Duplicate {label}: {value}")
        values.add(value)


def default_output(profiles, comparison=None):
    if len(profiles) == 1:
        filename = "pipeline_stage_latency.png"
    elif comparison == "resolution":
        filename = "pipeline_resolution_comparison.png"
    else:
        filename = "pipeline_generation_comparison.png"
    return profiles[0]["path"].with_name(filename)


def stage_arrays(data):
    stage_summary = data["summary"]["stages"]
    means_ms = [
        stage_summary[key]["mean_seconds"] * 1000
        for key, _, _ in STAGES
    ]
    medians_ms = [
        stage_summary[key]["median_seconds"] * 1000
        for key, _, _ in STAGES
    ]
    p90_ms = [
        stage_summary[key]["p90_seconds"] * 1000
        for key, _, _ in STAGES
    ]
    fractions = [
        stage_summary[key]["fraction_percent"]
        for key, _, _ in STAGES
    ]
    return means_ms, medians_ms, p90_ms, fractions


def save_figure(figure, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Saved: {output}")


def render_single(data, output):
    summary = data["summary"]
    images = data["images"]
    labels = [label for _, label, _ in STAGES]
    colors = [color for _, _, color in STAGES]
    means_ms, medians_ms, p90_ms, fractions = stage_arrays(data)

    figure = plt.figure(figsize=(15, 8))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.25, 1.55),
        height_ratios=(3.0, 1.0),
        hspace=0.42,
        wspace=0.22,
    )
    latency_axis = figure.add_subplot(grid[0, 0])
    fraction_axis = figure.add_subplot(grid[1, 0])
    table_axis = figure.add_subplot(grid[:, 1])

    y_positions = list(range(len(STAGES)))
    latency_axis.barh(y_positions, means_ms, color=colors, alpha=0.88)
    latency_axis.scatter(
        p90_ms,
        y_positions,
        color="#222222",
        marker="|",
        s=250,
        linewidths=2,
        zorder=3,
    )
    latency_axis.set_yticks(y_positions, labels=labels)
    latency_axis.invert_yaxis()
    latency_axis.set_xscale("log")
    latency_axis.set_xlabel("Latency (ms, logarithmic scale)")
    latency_axis.set_title(
        "Mean stage latency with P90 markers",
        loc="left",
        fontweight="bold",
    )
    latency_axis.grid(axis="x", linestyle=":", alpha=0.45)

    for index, mean_ms in enumerate(means_ms):
        latency_axis.annotate(
            f"{mean_ms:.1f} ms  ({fractions[index]:.1f}%)",
            (mean_ms, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    latency_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#222222",
                marker="|",
                linestyle="None",
                markersize=12,
                markeredgewidth=2,
                label="P90",
            )
        ],
        loc="lower right",
        frameon=False,
    )

    left = 0.0
    for fraction, color in zip(fractions, colors):
        fraction_axis.barh(
            [0],
            [fraction],
            left=left,
            color=color,
            height=0.5,
        )
        if fraction >= 4.0:
            fraction_axis.text(
                left + fraction / 2,
                0,
                f"{fraction:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        left += fraction

    fraction_axis.set_xlim(0, 100)
    fraction_axis.set_yticks([])
    fraction_axis.set_xlabel("Fraction of total measured latency (%)")
    fraction_axis.set_title(
        "Mean latency composition",
        loc="left",
        fontweight="bold",
    )

    table_axis.axis("off")
    table_rows = []
    for index, (_, label, _) in enumerate(STAGES):
        table_rows.append([
            label,
            f"{means_ms[index]:.1f}",
            f"{medians_ms[index]:.1f}",
            f"{p90_ms[index]:.1f}",
            f"{fractions[index]:.2f}%",
        ])
    table_rows.append([
        "Total",
        f"{summary['mean_seconds'] * 1000:.1f}",
        f"{summary['median_seconds'] * 1000:.1f}",
        f"{summary['p90_seconds'] * 1000:.1f}",
        "100.00%",
    ])
    table = table_axis.table(
        cellText=table_rows,
        colLabels=("Stage", "Mean (ms)", "Median (ms)", "P90 (ms)", "Fraction"),
        cellLoc="right",
        colLoc="center",
        loc="center",
        colWidths=(0.34, 0.17, 0.17, 0.17, 0.15),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.75)
    for column in range(5):
        table[(0, column)].set_facecolor("#E9EEF5")
        table[(0, column)].set_text_props(fontweight="bold")
    total_row = len(table_rows)
    for column in range(5):
        table[(total_row, column)].set_facecolor("#F0F0F0")
        table[(total_row, column)].set_text_props(fontweight="bold")
    for row in range(1, total_row + 1):
        table[(row, 0)].set_text_props(ha="left")

    generated_tokens = [
        item["generated_tokens_including_special_tokens"]
        for item in images
    ]
    figure.suptitle(
        "Pipeline-Level Latency Profile",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"{data['model']} | {data['device']} | {data['dtype']} | "
            f"{data['task']} | n={summary['count']}"
        ),
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.025,
        (
            "Instrumented stage latency; one unmeasured warm-up request. "
            f"Resolution limit: {resolution_label(data)}. "
            f"Generated tokens: {min(generated_tokens)}–{max(generated_tokens)}."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save_figure(figure, output)


def render_resolution_comparison(profiles, output):
    profiles = sorted(profiles, key=resolution_sort_key)
    labels = [resolution_label(profile["data"]) for profile in profiles]
    y_positions = list(range(len(profiles)))
    stage_means_ms = {
        key: [
            profile["data"]["summary"]["stages"][key]["mean_seconds"] * 1000
            for profile in profiles
        ]
        for key, _, _ in STAGES
    }
    stage_fractions = {
        key: [
            profile["data"]["summary"]["stages"][key]["fraction_percent"]
            for profile in profiles
        ]
        for key, _, _ in STAGES
    }
    total_means_ms = [
        profile["data"]["summary"]["mean_seconds"] * 1000
        for profile in profiles
    ]
    total_medians_ms = [
        profile["data"]["summary"]["median_seconds"] * 1000
        for profile in profiles
    ]
    total_p90_ms = [
        profile["data"]["summary"]["p90_seconds"] * 1000
        for profile in profiles
    ]

    figure = plt.figure(figsize=(17, 11))
    grid = figure.add_gridspec(2, 2, hspace=0.36, wspace=0.24)
    absolute_axis = figure.add_subplot(grid[0, 0])
    trend_axis = figure.add_subplot(grid[0, 1])
    fraction_axis = figure.add_subplot(grid[1, 0])
    table_axis = figure.add_subplot(grid[1, 1])

    left = [0.0] * len(profiles)
    for key, stage_label, color in STAGES:
        values = stage_means_ms[key]
        absolute_axis.barh(
            y_positions,
            values,
            left=left,
            color=color,
            label=stage_label,
            height=0.62,
        )
        left = [current + value for current, value in zip(left, values)]
    absolute_axis.scatter(
        total_p90_ms,
        y_positions,
        color="#222222",
        marker="|",
        s=260,
        linewidths=2,
        label="Total P90",
        zorder=3,
    )
    for index, total in enumerate(total_means_ms):
        absolute_axis.annotate(
            f"{total:.0f} ms",
            (total, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    absolute_axis.set_yticks(y_positions, labels=labels)
    absolute_axis.invert_yaxis()
    absolute_axis.set_xlabel("Mean latency (ms); marker shows total P90")
    absolute_axis.set_title(
        "Absolute stage latency by resolution",
        loc="left",
        fontweight="bold",
    )
    absolute_axis.grid(axis="x", linestyle=":", alpha=0.4)

    x_positions = list(range(len(profiles)))
    for key, stage_label, color in STAGES:
        trend_axis.plot(
            x_positions,
            stage_means_ms[key],
            marker="o",
            linewidth=2,
            color=color,
            label=stage_label,
        )
    trend_axis.set_xticks(x_positions, labels=labels)
    trend_axis.set_yscale("log")
    trend_axis.set_ylabel("Mean latency (ms, logarithmic scale)")
    trend_axis.set_title(
        "Stage scaling trend",
        loc="left",
        fontweight="bold",
    )
    trend_axis.grid(axis="y", linestyle=":", alpha=0.4)
    trend_axis.legend(fontsize=8, frameon=False, ncol=2)

    left = [0.0] * len(profiles)
    for key, stage_label, color in STAGES:
        values = stage_fractions[key]
        fraction_axis.barh(
            y_positions,
            values,
            left=left,
            color=color,
            label=stage_label,
            height=0.62,
        )
        left = [current + value for current, value in zip(left, values)]
    fraction_axis.set_yticks(y_positions, labels=labels)
    fraction_axis.invert_yaxis()
    fraction_axis.set_xlim(0, 100)
    fraction_axis.set_xlabel("Fraction of total measured latency (%)")
    fraction_axis.set_title(
        "Latency composition by resolution",
        loc="left",
        fontweight="bold",
    )
    fraction_axis.grid(axis="x", linestyle=":", alpha=0.35)

    table_axis.axis("off")
    table_rows = []
    for index, profile in enumerate(profiles):
        data = profile["data"]
        input_tokens = [
            item["input_tokens_including_visual_placeholders"]
            for item in data["images"]
        ]
        generated_tokens = [
            item["generated_tokens_including_special_tokens"]
            for item in data["images"]
        ]
        table_rows.append([
            labels[index],
            f"{total_means_ms[index]:.1f}",
            f"{total_medians_ms[index]:.1f}",
            f"{total_p90_ms[index]:.1f}",
            f"{sum(input_tokens) / len(input_tokens):.1f}",
            f"{sum(generated_tokens) / len(generated_tokens):.1f}",
        ])
    table = table_axis.table(
        cellText=table_rows,
        colLabels=(
            "Resolution",
            "Mean (ms)",
            "Median (ms)",
            "P90 (ms)",
            "Mean input\ntokens",
            "Mean output\ntokens",
        ),
        cellLoc="right",
        colLoc="center",
        loc="center",
        colWidths=(0.18, 0.17, 0.17, 0.17, 0.16, 0.16),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.75)
    for column in range(6):
        table[(0, column)].set_facecolor("#E9EEF5")
        table[(0, column)].set_text_props(fontweight="bold")
    for row in range(1, len(table_rows) + 1):
        table[(row, 0)].set_text_props(ha="left")

    reference = profiles[0]["data"]
    counts = {profile["data"]["summary"]["count"] for profile in profiles}
    count_text = str(next(iter(counts))) if len(counts) == 1 else "varied"
    figure.suptitle(
        "Pipeline Latency vs Image Resolution",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.945,
        (
            f"{reference['model']} | {reference['device']} | "
            f"{reference['dtype']} | {reference['task']} | "
            f"max new tokens={reference['max_new_tokens']} | n={count_text} per run"
        ),
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.018,
        (
            "Instrumented stage latency; one unmeasured warm-up request per run. "
            "All inputs use the same model, task, backend, prompt, and image order."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save_figure(figure, output)


def render_generation_comparison(profiles, output):
    profiles = sorted(profiles, key=generation_sort_key)
    labels = [generation_label(profile["data"]) for profile in profiles]
    y_positions = list(range(len(profiles)))
    x_positions = list(range(len(profiles)))
    stage_means_ms = {
        key: [
            profile["data"]["summary"]["stages"][key]["mean_seconds"] * 1000
            for profile in profiles
        ]
        for key, _, _ in STAGES
    }
    total_means_ms = [
        profile["data"]["summary"]["mean_seconds"] * 1000
        for profile in profiles
    ]
    total_medians_ms = [
        profile["data"]["summary"]["median_seconds"] * 1000
        for profile in profiles
    ]
    total_p90_ms = [
        profile["data"]["summary"]["p90_seconds"] * 1000
        for profile in profiles
    ]
    mean_output_tokens = []
    for profile in profiles:
        generated_tokens = [
            item["generated_tokens_including_special_tokens"]
            for item in profile["data"]["images"]
        ]
        mean_output_tokens.append(
            sum(generated_tokens) / len(generated_tokens)
        )

    figure = plt.figure(figsize=(17, 11))
    grid = figure.add_gridspec(2, 2, hspace=0.36, wspace=0.26)
    absolute_axis = figure.add_subplot(grid[0, 0])
    trend_axis = figure.add_subplot(grid[0, 1])
    generation_axis = figure.add_subplot(grid[1, 0])
    table_axis = figure.add_subplot(grid[1, 1])

    left = [0.0] * len(profiles)
    for key, stage_label, color in STAGES:
        values = stage_means_ms[key]
        absolute_axis.barh(
            y_positions,
            values,
            left=left,
            color=color,
            label=stage_label,
            height=0.62,
        )
        left = [current + value for current, value in zip(left, values)]
    absolute_axis.scatter(
        total_p90_ms,
        y_positions,
        color="#222222",
        marker="|",
        s=260,
        linewidths=2,
        label="Total P90",
        zorder=3,
    )
    for index, total in enumerate(total_means_ms):
        absolute_axis.annotate(
            f"{total:.0f} ms",
            (total, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    absolute_axis.set_yticks(y_positions, labels=labels)
    absolute_axis.invert_yaxis()
    absolute_axis.set_xlabel("Mean latency (ms); marker shows total P90")
    absolute_axis.set_title(
        "Absolute stage latency by generation limit",
        loc="left",
        fontweight="bold",
    )
    absolute_axis.grid(axis="x", linestyle=":", alpha=0.4)

    for key, stage_label, color in STAGES:
        trend_axis.plot(
            x_positions,
            stage_means_ms[key],
            marker="o",
            linewidth=2,
            color=color,
            label=stage_label,
        )
    trend_axis.set_xticks(x_positions, labels=labels)
    trend_axis.set_yscale("log")
    trend_axis.set_ylabel("Mean latency (ms, logarithmic scale)")
    trend_axis.set_title(
        "Stage scaling trend",
        loc="left",
        fontweight="bold",
    )
    trend_axis.grid(axis="y", linestyle=":", alpha=0.4)
    trend_axis.legend(fontsize=8, frameon=False, ncol=2)

    generation_ms = stage_means_ms["token_generation"]
    generation_axis.scatter(
        mean_output_tokens,
        generation_ms,
        color="#B279A2",
        s=65,
        zorder=3,
    )
    for index, token_limit in enumerate(labels):
        generation_axis.annotate(
            token_limit,
            (mean_output_tokens[index], generation_ms[index]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    x_mean = sum(mean_output_tokens) / len(mean_output_tokens)
    y_mean = sum(generation_ms) / len(generation_ms)
    denominator = sum(
        (value - x_mean) ** 2 for value in mean_output_tokens
    )
    if denominator > 0:
        slope = sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(mean_output_tokens, generation_ms)
        ) / denominator
        intercept = y_mean - slope * x_mean
        fit_x = [min(mean_output_tokens), max(mean_output_tokens)]
        fit_y = [slope * value + intercept for value in fit_x]
        fitted_values = [
            slope * value + intercept for value in mean_output_tokens
        ]
        total_variation = sum(
            (value - y_mean) ** 2 for value in generation_ms
        )
        residual_variation = sum(
            (actual - fitted) ** 2
            for actual, fitted in zip(generation_ms, fitted_values)
        )
        r_squared = (
            1.0 - residual_variation / total_variation
            if total_variation > 0
            else None
        )
        fit_label = f"Linear fit: {slope:.1f} ms/token"
        if r_squared is not None:
            fit_label += f", R²={r_squared:.3f}"
        generation_axis.plot(
            fit_x,
            fit_y,
            color="#444444",
            linestyle="--",
            linewidth=1.5,
            label=fit_label,
        )
        generation_axis.legend(frameon=False, fontsize=9)
    else:
        generation_axis.text(
            0.02,
            0.95,
            "Linear fit unavailable: mean output length did not vary",
            transform=generation_axis.transAxes,
            va="top",
            fontsize=9,
            color="#555555",
        )
    generation_axis.set_xlabel("Mean generated tokens (including special tokens)")
    generation_axis.set_ylabel("Mean token-generation latency (ms)")
    generation_axis.set_title(
        "Generation cost vs actual output length",
        loc="left",
        fontweight="bold",
    )
    generation_axis.grid(linestyle=":", alpha=0.4)

    table_axis.axis("off")
    table_rows = []
    for index, profile in enumerate(profiles):
        generation_fraction = profile["data"]["summary"]["stages"][
            "token_generation"
        ]["fraction_percent"]
        table_rows.append([
            str(profile["data"]["max_new_tokens"]),
            f"{mean_output_tokens[index]:.1f}",
            f"{generation_ms[index]:.1f}",
            f"{total_means_ms[index]:.1f}",
            f"{total_medians_ms[index]:.1f}",
            f"{total_p90_ms[index]:.1f}",
            f"{generation_fraction:.1f}%",
        ])
    table = table_axis.table(
        cellText=table_rows,
        colLabels=(
            "Max new\ntokens",
            "Mean actual\ntokens",
            "Generation\nmean (ms)",
            "Total mean\n(ms)",
            "Total median\n(ms)",
            "Total P90\n(ms)",
            "Generation\nfraction",
        ),
        cellLoc="right",
        colLoc="center",
        loc="center",
        colWidths=(0.13, 0.15, 0.16, 0.14, 0.14, 0.14, 0.14),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.85)
    for column in range(7):
        table[(0, column)].set_facecolor("#E9EEF5")
        table[(0, column)].set_text_props(fontweight="bold")

    reference = profiles[0]["data"]
    counts = {profile["data"]["summary"]["count"] for profile in profiles}
    count_text = str(next(iter(counts))) if len(counts) == 1 else "varied"
    figure.suptitle(
        "Pipeline Latency vs Generation Length",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.945,
        (
            f"{reference['model']} | {reference['device']} | "
            f"{reference['dtype']} | {reference['task']} | "
            f"resolution={resolution_label(reference)} | n={count_text} per run"
        ),
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.018,
        (
            "Instrumented stage latency; one unmeasured warm-up request per run. "
            "The fit uses actual generated-token counts, not configured limits."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save_figure(figure, output)


def main():
    args = parse_args()
    profiles = load_profiles(args.input)

    if len(profiles) == 1:
        output = (
            args.output
            if args.output is not None
            else default_output(profiles)
        )
        render_single(profiles[0]["data"], output)
    else:
        comparison = (
            infer_comparison(profiles)
            if args.comparison == "auto"
            else args.comparison
        )
        validate_comparison(profiles, comparison)
        output = (
            args.output
            if args.output is not None
            else default_output(profiles, comparison)
        )
        if comparison == "resolution":
            render_resolution_comparison(profiles, output)
        else:
            render_generation_comparison(profiles, output)


if __name__ == "__main__":
    main()
