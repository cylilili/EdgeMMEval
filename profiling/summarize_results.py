"""Analyze profiling sweep results and render an English PNG report."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "results/sweep"
DEFAULT_STABILITY_INPUT = SCRIPT_DIR / "results/stability"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/summary_metrics.png"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            index = 1 if bold and candidate.endswith("PingFang.ttc") else 0
            try:
                return ImageFont.truetype(candidate, size=size, index=index)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - text_box[1]),
        text,
        font=font,
        fill=fill,
        spacing=8,
        align="center",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def read_number(path: Path, pattern: str) -> float:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ValueError(f"Missing companion file: {path}") from error
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse {path}")
    return float(match.group(1))


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = [item for item in data["images"] if item.get("measured") is True]
    if not images:
        raise ValueError(f"No measured samples in {path}")

    prefix = path.with_suffix("")
    memory = data.get("summary", {}).get("memory", {})

    def memory_mean(field: str) -> float | None:
        value = memory.get(field, {}).get("mean_mb")
        return float(value) if isinstance(value, (int, float)) else None

    is_resolution_sweep = path.stem.startswith("resolution_")
    if is_resolution_sweep:
        swap_before = read_number(
            Path(f"{prefix}_swap_before.txt"), r"used\s*=\s*([0-9.]+)M"
        )
        swap_after = read_number(
            Path(f"{prefix}_swap_after.txt"), r"used\s*=\s*([0-9.]+)M"
        )
        pressure_before = read_number(
            Path(f"{prefix}_pressure_before.txt"), r"free percentage:\s*(\d+)%"
        )
        pressure_after = read_number(
            Path(f"{prefix}_pressure_after.txt"), r"free percentage:\s*(\d+)%"
        )
    else:
        # Swap is system-wide and was already active before these experiments,
        # so it cannot be attributed to an individual hardware or token run.
        swap_before = swap_after = None
        pressure_before = pressure_after = None
    original_max_side = max(max(item["original_image_size"]) for item in images)
    latencies = [float(item["latency_seconds"]) for item in images]
    input_tokens = [
        float(item["input_tokens_including_visual_placeholders"]) for item in images
    ]
    after_allocated = [
        float(item["memory"]["after_allocated_mb"])
        for item in images
        if isinstance(item.get("memory"), dict)
        and isinstance(item["memory"].get("after_allocated_mb"), (int, float))
    ]
    after_driver = [
        float(item["memory"]["after_driver_allocated_mb"])
        for item in images
        if isinstance(item.get("memory"), dict)
        and isinstance(item["memory"].get("after_driver_allocated_mb"), (int, float))
    ]

    def slope(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        mean_x = (len(values) - 1) / 2
        mean_y = statistics.fmean(values)
        denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
        return sum(
            (index - mean_x) * (value - mean_y)
            for index, value in enumerate(values)
        ) / denominator

    return {
        "name": path.stem,
        "group": path.parent.name,
        "task": data["task"],
        "device": data["device"],
        "resolution": data["max_image_side"],
        "max_new_tokens": data["max_new_tokens"],
        "sample_count": len(images),
        "mean_input_tokens": statistics.fmean(
            input_tokens
        ),
        "input_token_stdev": statistics.stdev(input_tokens) if len(input_tokens) > 1 else 0.0,
        "mean_generated_tokens": statistics.fmean(
            item["generated_tokens_including_special_tokens"] for item in images
        ),
        "mean_seconds": float(data["summary"]["mean_seconds"]),
        "p90_seconds": float(data["summary"]["p90_seconds"]),
        "p99_seconds": sorted(latencies)[math.ceil(0.99 * len(latencies)) - 1],
        "latency_stdev_seconds": (
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        ),
        "mean_peak_memory_mb": memory_mean("peak_driver_allocated_mb"),
        "mean_memory_increase_mb": memory_mean("peak_increment_driver_mb"),
        "swap_before_mb": swap_before,
        "swap_after_mb": swap_after,
        "swap_delta_mb": (
            swap_after - swap_before
            if swap_before is not None and swap_after is not None
            else None
        ),
        "pressure_before": pressure_before,
        "pressure_after": pressure_after,
        "original_max_side": original_max_side,
        "after_allocated_delta_mb": (
            after_allocated[-1] - after_allocated[0]
            if len(after_allocated) == len(images)
            else None
        ),
        "after_driver_delta_mb": (
            after_driver[-1] - after_driver[0]
            if len(after_driver) == len(images)
            else None
        ),
        "late_driver_slope_mb_per_request": (
            slope(after_driver[len(after_driver) // 2 :])
            if len(after_driver) == len(images)
            else None
        ),
    }


def linear_regression(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    xs = [row["mean_generated_tokens"] for row in rows]
    ys = [row["mean_seconds"] for row in rows]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1 - residual / total
    return intercept, slope, r_squared


def fmt(value: float | None, decimals: int = 2, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}{suffix}"


def draw_table(
    image: Image.Image,
    y: int,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    note: str,
) -> int:
    draw = ImageDraw.Draw(image)
    margin = 50
    section_font = load_font(29, bold=True)
    header_font = load_font(19, bold=True)
    body_font = load_font(19)
    note_font = load_font(19)
    header_height = 86
    row_height = 62
    table_width = sum(widths)

    draw.text((margin, y), title, font=section_font, fill="#172033")
    y += 48
    table_top = y
    x = margin
    for header, width in zip(headers, widths):
        draw.rectangle((x, y, x + width, y + header_height), fill="#1F4E78")
        draw_centered_text(
            draw, (x, y, x + width, y + header_height), header, header_font, "#FFFFFF"
        )
        x += width

    for row_index, values in enumerate(rows):
        top = table_top + header_height + row_index * row_height
        bottom = top + row_height
        background = "#FFFFFF" if row_index % 2 == 0 else "#EAF1F8"
        x = margin
        for value, width in zip(values, widths):
            draw.rectangle((x, top, x + width, bottom), fill=background)
            draw_centered_text(draw, (x, top, x + width, bottom), value, body_font, "#1E293B")
            x += width

    table_bottom = table_top + header_height + len(rows) * row_height
    x = margin
    for width in widths[:-1]:
        x += width
        draw.line((x, table_top, x, table_bottom), fill="#CBD5E1", width=2)
    for index in range(len(rows) + 1):
        line_y = table_top + header_height + index * row_height
        draw.line((margin, line_y, margin + table_width, line_y), fill="#CBD5E1", width=2)
    draw.rectangle(
        (margin, table_top, margin + table_width, table_bottom),
        outline="#CBD5E1",
        width=2,
    )
    draw.text((margin, table_bottom + 14), note, font=note_font, fill="#526175")
    return table_bottom + 55


def render_report(results: list[dict[str, Any]]) -> Image.Image:
    resolution = [row for row in results if row["name"].startswith("resolution_")]
    resolution.sort(
        key=lambda row: (
            0 if row["task"] == "classification" else 1,
            9999 if row["resolution"] == 0 else row["resolution"],
        )
    )
    hardware = [row for row in results if row["name"].startswith("hardware_")]
    hardware.sort(key=lambda row: (row["task"], row["device"]), reverse=False)
    generation = [row for row in results if row["name"].startswith("generation_")]
    generation.sort(key=lambda row: row["max_new_tokens"])
    stability = [row for row in results if row["group"] == "stability"]
    stability.sort(
        key=lambda row: (
            0 if row["task"] == "classification" else 1,
            row["sample_count"],
        )
    )

    width = 2380
    height = 2320
    image = Image.new("RGB", (width, height), "#F4F7FB")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    subtitle_font = load_font(21)
    draw.text((50, 34), "Qwen3-VL Profiling Analysis", font=title_font, fill="#172033")
    draw.text(
        (50, 88),
        "Warm-up excluded; sample counts are shown per experiment; memory values are MPS sampling estimates.",
        font=subtitle_font,
        fill="#526175",
    )

    resolution_rows = []
    for row in resolution:
        resolution_name = "Original" if row["resolution"] == 0 else f"{row['resolution']} px"
        task_name = "Classification" if row["task"] == "classification" else "Caption"
        pressure = f"{row['pressure_before']:.0f}% -> {row['pressure_after']:.0f}%"
        swap_delta = (
            fmt(row["swap_delta_mb"], 1)
            if row["swap_before_mb"] == 0
            else "N/A*"
        )
        resolution_rows.append(
            [
                task_name,
                resolution_name,
                fmt(row["mean_input_tokens"], 1),
                fmt(row["mean_generated_tokens"], 1),
                fmt(row["mean_seconds"], 3, " s"),
                fmt(row["p90_seconds"], 3, " s"),
                fmt(row["mean_peak_memory_mb"], 1),
                fmt(row["mean_memory_increase_mb"], 1),
                fmt(row["swap_before_mb"], 1),
                fmt(row["swap_after_mb"], 1),
                swap_delta,
                pressure,
            ]
        )
    y = draw_table(
        image,
        135,
        "Resolution scaling",
        [
            "Task",
            "Resolution",
            "Mean input\ntokens",
            "Mean output\ntokens",
            "Mean\nlatency",
            "P90\nlatency",
            "Mean peak\nmemory (MB)",
            "Mean request\nincrease (MB)",
            "Swap before\n(MB)",
            "Swap after\n(MB)",
            "Swap delta\n(MB)",
            "Free memory\npercentage",
        ],
        resolution_rows,
        [185, 155, 175, 175, 170, 170, 205, 220, 170, 170, 160, 245],
        (
            "System swap rose during the 768 px Caption run; this does not establish a size threshold. "
            "The source images top out at 640 px, so 768 px and Original use the same visual inputs. "
            "*Swap delta is not attributed when swap existed before the run."
        ),
    )

    hardware_rows = []
    for task in ("classification", "caption"):
        task_rows = [row for row in hardware if row["task"] == task]
        cpu = next(row for row in task_rows if row["device"] == "cpu")
        mps = next(row for row in task_rows if row["device"] == "mps")
        speedup = cpu["mean_seconds"] / mps["mean_seconds"]
        for row in sorted(task_rows, key=lambda item: item["device"], reverse=True):
            hardware_rows.append(
                [
                    "Classification" if task == "classification" else "Caption",
                    row["device"].upper(),
                    fmt(row["mean_seconds"], 3, " s"),
                    fmt(row["p90_seconds"], 3, " s"),
                    fmt(speedup, 2, "x") if row["device"] == "mps" else "1.00x",
                    fmt(row["mean_peak_memory_mb"], 1),
                ]
            )
    y = draw_table(
        image,
        y,
        "Hardware backend",
        ["Task", "Backend", "Mean latency", "P90 latency", "Speedup vs CPU", "Mean peak\nmemory (MB)"],
        hardware_rows,
        [285, 190, 260, 260, 270, 310],
        (
            "MPS used FP16 and CPU used FP32; the speedup includes both backend and precision differences. "
            "System swap is excluded because it pre-dated these runs."
        ),
    )

    intercept, slope, r_squared = linear_regression(generation)
    generation_rows = [
        [
            str(row["max_new_tokens"]),
            fmt(row["mean_generated_tokens"], 1),
            fmt(row["mean_seconds"], 3, " s"),
            fmt(row["p90_seconds"], 3, " s"),
            fmt(row["mean_peak_memory_mb"], 1),
            fmt(row["mean_memory_increase_mb"], 1),
        ]
        for row in generation
    ]
    y = draw_table(
        image,
        y,
        "Generation length",
        ["Token limit", "Mean actual\ntokens", "Mean latency", "P90 latency", "Mean peak\nmemory (MB)", "Mean request\nincrease (MB)"],
        generation_rows,
        [230, 260, 270, 270, 320, 340],
        (
            f"Fit using actual output tokens: latency = {intercept:.3f} + "
            f"{slope:.4f} x tokens seconds (R-squared = {r_squared:.4f}). "
            "This slope is for total timed requests, not isolated decode time. "
            "System swap is excluded because it pre-dated these runs."
        ),
    )
    stability_rows = [
        [
            "Classification" if row["task"] == "classification" else "Caption",
            str(row["sample_count"]),
            fmt(row["mean_input_tokens"], 1),
            fmt(row["input_token_stdev"], 1),
            fmt(row["mean_seconds"], 3, " s"),
            fmt(row["p90_seconds"], 3, " s"),
            fmt(row["p99_seconds"], 3, " s"),
            fmt(row["latency_stdev_seconds"], 3, " s"),
            fmt(row["mean_peak_memory_mb"], 1),
            (
                f"{fmt(row['after_driver_delta_mb'], 1)} / "
                f"{fmt(row['late_driver_slope_mb_per_request'], 2)}"
            ),
            fmt(row["after_allocated_delta_mb"], 1),
        ]
        for row in stability
    ]
    draw_table(
        image,
        y,
        "Sample scaling and stability",
        [
            "Task",
            "n",
            "Mean input\ntokens",
            "Input token\nSD",
            "Mean\nlatency",
            "P90\nlatency",
            "P99\nlatency",
            "Latency\nSD",
            "Mean peak\nmemory (MB)",
            "Driver delta / late slope\n(MB, MB/request)",
            "Post-request allocated\ndelta (MB)",
        ],
        stability_rows,
        [190, 90, 175, 170, 170, 170, 170, 170, 215, 280, 260],
        (
            "At n=50, Caption P99 reaches 21.303 s. Post-request allocated memory changes by about "
            "1 MB; driver growth may include MPS pool expansion and does not alone establish a leak."
        ),
    )
    return image


def main() -> None:
    args = parse_args()
    paths = sorted(path for path in args.input_dir.glob("*.json") if not path.name.startswith("._"))
    stability_paths = sorted(
        path for path in args.stability_dir.glob("*.json") if not path.name.startswith("._")
    )
    if len(paths) != 18:
        raise SystemExit(f"error: expected 18 JSON files in {args.input_dir}, found {len(paths)}")
    if len(stability_paths) != 4:
        raise SystemExit(
            f"error: expected 4 JSON files in {args.stability_dir}, found {len(stability_paths)}"
        )
    try:
        results = [load_result(path) for path in paths + stability_paths]
        image = render_report(results)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"Saved summary image: {args.output.resolve()}")
    if not args.no_show:
        image.show(title="Profiling sweep analysis")


if __name__ == "__main__":
    main()
