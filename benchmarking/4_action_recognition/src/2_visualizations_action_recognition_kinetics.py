from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

BAR_COLORS = [
  "#b94984", 
  "#cd80a5", 
  "#ddb4c7",
  "#7b3559",
  "#422131", 
]

def plot_latency_boxplot(
    latencies_by_model: Dict[str, List[float]],
    path: str,
    title: str,
    yscale: str = "log",
):
    

    labels, data = [], []
    for name, vals in latencies_by_model.items():
        v = [float(x) for x in vals if x is not None and np.isfinite(x)]
        labels.append(f"{name.lower()}\n(med={np.median(v):.3f}s)" if v else f"{name}\n(med=n/a)")
        data.append(v)

    plt.figure(figsize=(5, 4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Latency per sample (seconds)")
    plt.yscale(yscale)

    if yscale == "log":
        all_vals = [x for v in data for x in v if x > 0]
        if all_vals:
            plt.ylim(bottom=max(min(all_vals) * 0.8, 1e-4))

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_accuracy_bar(
    acc_by_model: Dict[str, float],
    path: str,
    title: str,
):
    names = [str(k).lower() for k in acc_by_model.keys()]
    vals = [acc_by_model[k] for k in acc_by_model.keys()]

    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(names))]

    plt.figure(figsize=(5, 4))
    plt.bar(
        range(len(names)),
        vals,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
    )
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Top-1 Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def _ordered_prefixes_from_columns(columns: List[str]) -> List[str]:
    suffixes = ["_latency_sec", "_correct"]
    seen = set()
    ordered: List[str] = []

    for c in columns:
        for suf in suffixes:
            if c.endswith(suf):
                p = c[: -len(suf)]
                if p not in seen:
                    seen.add(p)
                    ordered.append(p)
                break
    return ordered


def compute_metrics_from_dataframe(
    df,
    model_name_map: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    prefixes = _ordered_prefixes_from_columns(list(df.columns))

    def disp(prefix: str) -> str:
        if model_name_map and prefix in model_name_map:
            return model_name_map[prefix]
        return prefix

    latencies_by_model: Dict[str, List[float]] = {}
    acc_by_model: Dict[str, float] = {}

    for p in prefixes:
        dname = disp(p)

        lat_col = f"{p}_latency_sec"
        if lat_col in df.columns:
            lat_vals = []
            for x in df[lat_col].tolist():
                try:
                    x = float(x)
                    if np.isfinite(x):
                        lat_vals.append(x)
                except Exception:
                    continue
        else:
            lat_vals = []
        latencies_by_model[dname] = lat_vals

        cor_col = f"{p}_correct"
        if cor_col in df.columns:
            cors = []
            for x in df[cor_col].tolist():
                try:
                    cors.append(int(x))
                except Exception:
                    continue
            acc = float(np.mean(cors)) if cors else 0.0
        else:
            acc = 0.0
        acc_by_model[dname] = acc

    return latencies_by_model, acc_by_model


def make_titles_from_dataframe(df) -> Tuple[str, str]:
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns and len(df) else ""
    split = df["split"].iloc[0] if "split" in df.columns and len(df) else ""
    n = int(len(df))
    return (
        f"Action latency (kinetics/mini, val, n={n})".strip(),
        f"Action top-1 accuracy (kinetics/mini, val, n={n})".strip(),
    )


def render_plots_from_csv(
    log_csv: str,
    latency_plot_path: str,
    acc_plot_path: str,
    latency_scale: str = "log",
    model_name_map: Optional[Dict[str, str]] = None,
):
    import pandas as pd

    df = pd.read_csv(log_csv)
    latencies_by_model, acc_by_model = compute_metrics_from_dataframe(df, model_name_map=model_name_map)
    lat_title, acc_title = make_titles_from_dataframe(df)

    plot_latency_boxplot(latencies_by_model, latency_plot_path, lat_title, yscale=latency_scale)
    plot_accuracy_bar(acc_by_model, acc_plot_path, acc_title)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", required=True, help="Path to CSV produced by 1ar.py")
    ap.add_argument("--latency_plot", default="ar_latency_kinetics.png")
    ap.add_argument("--acc_plot", default="ar_accuracy_kinetics.png")
    ap.add_argument("--latency_scale", default="log", choices=["linear", "log"])
    args = ap.parse_args()

    render_plots_from_csv(
        log_csv=args.log_csv,
        latency_plot_path=args.latency_plot,
        acc_plot_path=args.acc_plot,
        latency_scale=args.latency_scale,
    )
    print(f"Saved latency plot to: {args.latency_plot}")
    print(f"Saved accuracy plot to: {args.acc_plot}")

if __name__ == "__main__":
    main()