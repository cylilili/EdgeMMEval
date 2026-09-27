"""Render the four classification/caption profiling results as an image.

By default, the script reads the four ``coco_*`` JSON files in the sibling
``results`` directory, writes a PNG table, and opens it in the system image
viewer.  Only entries whose ``measured`` field is true are included, so
warm-up requests are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = (
    SCRIPT_DIR / "results/coco_cls_res384_2.json",
    SCRIPT_DIR / "results/coco_cls_resOrig_2.json",
    SCRIPT_DIR / "results/coco_cap_res384_2.json",
    SCRIPT_DIR / "results/coco_cap_resOrig_2.json",
)

TOKEN_FIELDS = (
    "input_tokens_including_visual_placeholders",
    "generated_tokens_including_special_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=list(DEFAULT_FILES),
        help="Result JSON files (default: the four profiling/results/coco_* files)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "results/summary_metrics.png",
        help="PNG output path (default: profiling/results/summary_metrics.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the PNG without opening the system image viewer",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for means and latency values (default: 3)",
    )
    args = parser.parse_args()
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def require_number(item: dict[str, Any], field: str, path: Path) -> float:
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: image entry has no numeric {field!r}")
    return float(value)


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile (P90 is item ceil(0.9 * n))."""
    if not values:
        raise ValueError("cannot calculate a percentile for an empty list")
    rank = max(1, math.ceil(percentile * len(values)))
    return sorted(values)[rank - 1]


def summarize(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"result file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError(f"{path}: expected an object containing an 'images' list")

    measured = [
        item
        for item in data["images"]
        if isinstance(item, dict) and item.get("measured") is True
    ]
    if not measured:
        raise ValueError(f"{path}: no measured image entries found")

    task = data.get("task")
    if task not in {"classification", "caption"}:
        raise ValueError(f"{path}: unsupported task {task!r}")

    input_tokens = [require_number(item, TOKEN_FIELDS[0], path) for item in measured]
    generated_tokens = [require_number(item, TOKEN_FIELDS[1], path) for item in measured]
    latencies = [require_number(item, "latency_seconds", path) for item in measured]
    memory_records = [
        item["memory"]
        for item in measured
        if isinstance(item.get("memory"), dict)
    ]

    def mean_memory(*fields: str) -> float | None:
        for field in fields:
            values = [
                float(record[field])
                for record in memory_records
                if isinstance(record.get(field), (int, float))
                and not isinstance(record.get(field), bool)
            ]
            if values:
                return statistics.fmean(values)
        return None

    resolution = data.get("max_image_side")
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
        raise ValueError(f"{path}: missing numeric 'max_image_side'")

    task_name = "Classification" if task == "classification" else "Caption"
    resolution_name = "Original" if resolution == 0 else f"{resolution:g}px"
    return {
        "sort_key": (0 if task == "classification" else 1, 1 if resolution == 0 else 0),
        "experiment": f"{task_name} ({'Original' if resolution == 0 else 'Small Image'})",
        "task_output": f"{'Classification' if task == 'classification' else 'Caption'} / {data.get('max_new_tokens', 'N/A')}",
        "resolution": f"{resolution_name} (0)" if resolution == 0 else resolution_name,
        "mean_input_tokens": statistics.fmean(input_tokens),
        "mean_generated_tokens": statistics.fmean(generated_tokens),
        "mean_seconds": statistics.fmean(latencies),
        "p90_seconds": percentile_nearest_rank(latencies, 0.90),
        "mean_peak_memory_mb": mean_memory(
            "peak_driver_allocated_mb", "peak_reserved_mb", "peak_allocated_mb"
        ),
        "mean_peak_increment_mb": mean_memory(
            "peak_increment_driver_mb",
            "peak_increment_reserved_mb",
            "peak_increment_allocated_mb",
        ),
        "count": len(measured),
    }


def format_value(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


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
            # PingFang contains several faces; face 1 is semibold on macOS.
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


def render_image(rows: list[dict[str, Any]], decimals: int) -> Image.Image:
    headers = [
        "Experiment",
        "Task / Output\nToken Limit",
        "Resolution\nLimit",
        "Mean Visual + Text\nInput Tokens",
        "Mean Generated\nTokens",
        "Mean Latency\n(mean_s)",
        "P90 Latency",
    ]
    column_widths = [250, 260, 190, 300, 240, 220, 190]
    has_memory = any(row["mean_peak_memory_mb"] is not None for row in rows)
    if has_memory:
        headers.extend(("Mean Peak\nMemory (MB)", "Mean Request\nIncrease (MB)"))
        column_widths.extend((240, 260))
    margin = 52
    title_height = 105
    header_height = 112
    row_height = 88
    footer_height = 62
    table_width = sum(column_widths)
    image_width = table_width + margin * 2
    image_height = title_height + header_height + row_height * len(rows) + footer_height + margin

    image = Image.new("RGB", (image_width, image_height), "#F4F7FB")
    draw = ImageDraw.Draw(image)
    title_font = load_font(36, bold=True)
    header_font = load_font(25, bold=True)
    body_font = load_font(25)
    footer_font = load_font(20)

    draw.text((margin, 38), "Qwen3-VL-2B-Instruct Performance Summary", font=title_font, fill="#172033")
    table_top = title_height
    draw.rounded_rectangle(
        (margin, table_top, margin + table_width, table_top + header_height + row_height * len(rows)),
        radius=16,
        fill="#FFFFFF",
        outline="#CBD5E1",
        width=2,
    )

    x = margin
    for header, width in zip(headers, column_widths):
        draw.rectangle((x, table_top, x + width, table_top + header_height), fill="#1F4E78")
        draw_centered_text(
            draw,
            (x, table_top, x + width, table_top + header_height),
            header,
            header_font,
            "#FFFFFF",
        )
        x += width

    for row_index, row in enumerate(rows):
        top = table_top + header_height + row_index * row_height
        bottom = top + row_height
        background = "#FFFFFF" if row_index % 2 == 0 else "#EAF1F8"
        experiment_label = row["experiment"].replace(" (", "\n(")
        values = [
            f"{row_index + 1}. {experiment_label}",
            row["task_output"],
            row["resolution"],
            format_value(row["mean_input_tokens"], decimals),
            format_value(row["mean_generated_tokens"], decimals),
            f"{format_value(row['mean_seconds'], decimals)} s",
            f"{format_value(row['p90_seconds'], decimals)} s",
        ]
        if has_memory:
            values.extend(
                (
                    "N/A" if row["mean_peak_memory_mb"] is None else
                    format_value(row["mean_peak_memory_mb"], decimals),
                    "N/A" if row["mean_peak_increment_mb"] is None else
                    format_value(row["mean_peak_increment_mb"], decimals),
                )
            )
        x = margin
        for value, width in zip(values, column_widths):
            draw.rectangle((x, top, x + width, bottom), fill=background)
            draw_centered_text(draw, (x, top, x + width, bottom), value, body_font, "#1E293B")
            x += width

    x = margin
    table_bottom = table_top + header_height + row_height * len(rows)
    for width in column_widths[:-1]:
        x += width
        draw.line((x, table_top, x, table_bottom), fill="#CBD5E1", width=2)
    for index in range(len(rows) + 1):
        y = table_top + header_height + index * row_height
        draw.line((margin, y, margin + table_width, y), fill="#CBD5E1", width=2)

    sample_counts = sorted({row["count"] for row in rows})
    count_text = str(sample_counts[0]) if len(sample_counts) == 1 else "/".join(map(str, sample_counts))
    footer = (
        f"Measured samples only; n={count_text} per experiment; P90 uses the nearest-rank method."
    )
    if has_memory:
        footer += " MPS peak memory is sampled during inference."
    draw.text(
        (margin, table_bottom + 22),
        footer,
        font=footer_font,
        fill="#526175",
    )
    return image


def main() -> None:
    args = parse_args()
    try:
        rows = sorted((summarize(path) for path in args.files), key=lambda row: row["sort_key"])
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    image = render_image(rows, args.decimals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"Saved table image: {args.output.resolve()}")
    if not args.no_show:
        image.show(title="Profiling summary")


if __name__ == "__main__":
    main()
