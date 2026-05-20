import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = "results_figures_coco_10kimages"

MODELS = [
    ("yolo", "YOLO"),
    ("detr", "DETR"),
    ("frcnn", "FRCNN"),
    ("blip2", "BLIP2"),
    ("gemma_e2b", "GEMMA_E2B"),
    ("gemma_e4b", "GEMMA_E4B"),
]

style = {
    "YOLO":      {"marker": "o", "color": "#216154"},
    "DETR":      {"marker": "s", "color": "#20ae96"},
    "FRCNN":     {"marker": "D", "color": "#679aae"},
    "BLIP2":     {"marker": "^", "color": "#d16c73"},
    "GEMMA_E2B": {"marker": "P", "color": "#9c3ec3"},
    "GEMMA_E4B": {"marker": "X", "color": "#c991dc"},
}

def compute_accuracy_by_gt(df: pd.DataFrame, pred_col: str, gt_col: str = "gt"):
    grouped = df.groupby(gt_col)
    xs, ys, ns = [], [], []
    for gt_val, g in grouped:
        xs.append(int(gt_val))
        ys.append(float((g[pred_col] == g[gt_col]).mean()))
        ns.append(int(len(g)))

    order = np.argsort(xs)
    xs = np.array(xs, dtype=int)[order]
    ys = np.array(ys, dtype=float)[order]
    ns = np.array(ns, dtype=int)[order]
    return xs, ys, ns

def bin_by_gt(df: pd.DataFrame, bins: int = 12, gt_col: str = "gt"):
    gmin, gmax = int(df[gt_col].min()), int(df[gt_col].max())
    out = df.copy()

    if gmin == gmax:
        out["gt_bin"] = float(gmin)
        return out

    edges = np.linspace(gmin, gmax + 1e-9, bins + 1)
    b = np.digitize(out[gt_col].values, edges, right=False) - 1
    b = np.clip(b, 0, bins - 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    out["gt_bin"] = mids[b]
    return out

def plot_all_models(xs_map, ys_map, title, out_path):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()

    for pred_key, label in MODELS:
        xs = xs_map[pred_key]
        ys = ys_map[pred_key]
        st = style.get(label, {})
        ax.plot(
            xs, ys,
            label=label,
            linewidth=1.8,
            marker=st.get("marker", "o"),
            markersize=5,
            color=st.get("color", None),
        )

    ax.set_xlabel("Number of people in image (GT count)")
    ax.set_ylabel("Accuracy (exact match)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_single_model(xs, ys, label, title, out_path):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()

    st = style.get(label, {})
    ax.plot(
        xs, ys,
        linewidth=2.0,
        marker=st.get("marker", "o"),
        markersize=6,
        color=st.get("color", None),
    )

    ax.set_xlabel("Number of people in image (GT count)")
    ax.set_ylabel("Accuracy (exact match)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--max_count", type=int, default=None)
    ap.add_argument("--binned", action="store_true")
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args()

    csv_path = os.path.join(args.results_dir, f"round_{args.round}_per_image.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Could not find: {csv_path}")

    df = pd.read_csv(csv_path)

    df = df[df["gt"] >= args.min_count]
    if args.max_count is not None:
        df = df[df["gt"] <= args.max_count]

    xs_map, ys_map = {}, {}
    for pred_key, label in MODELS:
        xs, ys, _ns = compute_accuracy_by_gt(df, pred_col=pred_key, gt_col="gt")
        xs_map[pred_key] = xs
        ys_map[pred_key] = ys

        out_ind = os.path.join(
            args.results_dir,
            f"accuracy_vs_gt_count_round_{args.round}_{label}.png"
        )
        plot_single_model(
            xs, ys,
            label=label,
            title=f"{label} — Accuracy vs GT crowd size",
            out_path=out_ind
        )
        print(f"Saved: {out_ind}")

    out_comb = os.path.join(args.results_dir, f"accuracy_vs_gt_count_round_{args.round}.png")
    plot_all_models(
        xs_map, ys_map,
        title=f"Accuracy vs GT crowd size",
        out_path=out_comb
    )
    print(f"Saved: {out_comb}")

    if args.binned:
        dfb = bin_by_gt(df, bins=args.bins, gt_col="gt")
        grouped = dfb.groupby("gt_bin")
        bin_centers = np.array(sorted(grouped.groups.keys()), dtype=float)

        xs_map_b, ys_map_b = {}, {}

        for pred_key, label in MODELS:
            accs = []
            for bc in bin_centers:
                chunk = dfb[dfb["gt_bin"] == bc]
                accs.append(float((chunk[pred_key] == chunk["gt"]).mean()))

            xs_map_b[pred_key] = bin_centers
            ys_map_b[pred_key] = np.array(accs, dtype=float)

            out_ind_b = os.path.join(
                args.results_dir,
                f"accuracy_vs_gt_count_round_{args.round}_{label}_binned.png"
            )
            plot_single_model(
                bin_centers,
                ys_map_b[pred_key],
                label=label,
                title=f"{label} — Accuracy vs GT crowd size (binned, bins={args.bins})",
                out_path=out_ind_b
            )
            print(f"Saved: {out_ind_b}")

        out_comb_b = os.path.join(args.results_dir, f"accuracy_vs_gt_count_round_{args.round}_binned.png")
        plot_all_models(
            xs_map_b, ys_map_b,
            title=f"Accuracy vs GT crowd size (binned, bins={args.bins}) — Round {args.round}",
            out_path=out_comb_b
        )
        print(f"Saved: {out_comb_b}")

if __name__ == "__main__":
    main()