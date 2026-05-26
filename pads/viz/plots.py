"""All plotting routines."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pads.data.schema import N_TIME_OFFSETS
from pads.eval.metrics import evaluate
from pads.eval.thresholds import ThresholdMethod, optimal_threshold

# A clean, modern matplotlib look shared by every generated figure. Applied once
# at import; plots.py is only used by the pipeline so global rcParams are fine.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelcolor": "#2b2f36",
    "axes.edgecolor": "#d0d4da",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e8eaed",
    "grid.linewidth": 0.9,
    "xtick.color": "#5b626b",
    "ytick.color": "#5b626b",
    "legend.frameon": False,
    "lines.linewidth": 2.4,
})

COLOR_MAP_GROUPS = {
    "EXITUS <48h": "#f43543",
    "EXITUS >48h": "#ff9e30",
    "ALIVE <48h": "#3ab830",
    "ALIVE >48h": "#1097d8",
}
COLOR_MAP_ERROR = {0: "#3ab06a", 1: "#f4c430", 2: "#ef8a3a", 3: "#e4572e"}


def plot_roc_combined(
    mort_pred: np.ndarray,
    mort_gt: np.ndarray,
    disch_pred: np.ndarray,
    disch_gt: np.ndarray,
    *,
    out_dir: str | Path,
    inference_type: str | None = None,
    threshold_method: ThresholdMethod = "min_distance",
    save: bool = True,
) -> tuple[float, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    th_m = optimal_threshold(threshold_method, mort_gt, mort_pred)
    th_d = optimal_threshold(threshold_method, disch_gt, disch_pred)
    m = evaluate(mort_gt, mort_pred, th_m)
    d = evaluate(disch_gt, disch_pred, th_d)

    c_mort, c_disch = "#e4572e", "#3a7ca5"
    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.plot([0, 100], [0, 100], color="#c3c8cf", ls="--", lw=1.2, zorder=1)  # chance line

    for x, color, name in ((m, c_mort, "Mortality"), (d, c_disch, "Discharge")):
        fpc, tpc = 100 * x.fp_curve, 100 * x.tp_curve
        ax.fill_between(fpc, tpc, color=color, alpha=0.08, zorder=2)
        ax.plot(fpc, tpc, color=color, label=f"{name}  ·  AUC {x.auc:.2f}", zorder=3)
        ax.scatter(100 * x.opt_fp, 100 * x.opt_tp, color=color, edgecolor="white",
                   s=95, lw=1.6, zorder=5)
        ax.annotate(f"thr {x.threshold:.2f}", (100 * x.opt_fp, 100 * x.opt_tp),
                    textcoords="offset points", xytext=(9, -4), fontsize=9, color=color)

    ax.set_title(f"ROC — Mortality & Discharge   ·   threshold: {threshold_method}", pad=14)
    ax.set_xlabel("False Positives [%]")
    ax.set_ylabel("True Positives [%]")
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=11, handlelength=1.4)

    # Clean metrics table below the plot (replaces the old monospace block).
    cols = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Threshold"]
    cells = [
        [f"{x.accuracy:.2f}", f"{x.precision:.2f}", f"{x.recall:.2f}",
         f"{x.f1:.2f}", f"{x.auc:.2f}", f"{x.threshold:.3f}"]
        for x in (m, d)
    ]
    tbl = ax.table(cellText=cells, rowLabels=["Mortality", "Discharge"], colLabels=cols,
                   loc="bottom", bbox=[0.0, -0.26, 1.0, 0.16])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    row_color = {1: c_mort, 2: c_disch}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e6e9ee")
        cell.set_linewidth(0.8)
        if r == 0:  # header
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(weight="bold", color="#2b2f36")
        if c == -1 and r in row_color:  # row labels coloured per model
            cell.set_text_props(weight="bold", color=row_color[r])
    plt.subplots_adjust(bottom=0.2)

    if save:
        suffix = f"_{inference_type}" if inference_type else ""
        plt.savefig(out_dir / f"roc_combined{suffix}.png")
    plt.close()
    return th_m, th_d


def plot_inference(
    df_prob: pd.DataFrame,
    *,
    out_dir: str | Path,
    test_type: str = "last_96h",
    save: bool = False,
) -> None:
    """Per-stay scatter of adjusted mortality probability over time."""
    out_dir = Path(out_dir)
    for stay_id, group in df_prob.groupby("stay_id"):
        group = group.copy()
        group["normalized"] = group["normalized"].fillna(50)
        group["color_group"] = group["color_group"].fillna("GRIS")
        plt.figure(figsize=(10, 5))
        plt.scatter(
            group["hr"], group["normalized"],
            c=group["color_group"].map(COLOR_MAP_GROUPS), s=50, alpha=0.85,
        )
        plt.axvline(
            x=group["hr"].max() - N_TIME_OFFSETS, color="black",
            linestyle="--", lw=1.5, label="48h before discharge",
        )
        plt.axhline(y=50, color="gray", linestyle=":", lw=1.2)
        plt.xlabel("hr (relative to discharge)", fontsize=12)
        plt.ylabel("Adjusted Mortality Probability (%)", fontsize=12)
        alive = group["mortality_groundtruth"].iloc[-1] == 0
        plt.title(
            f"Stay {stay_id} — {test_type}",
            fontsize=14, color="#3ab830" if alive else "#f43543",
        )
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", lw=0.5, alpha=0.6)
        plt.tight_layout()
        plt.legend()
        if save:
            folder = out_dir / test_type
            folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(folder / f"{stay_id}.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_error(df: pd.DataFrame, *, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = (
        df["error"].value_counts(normalize=True)
        .reset_index()
    )
    counts.columns = ["group", "proportion"]
    counts["group"] = counts["group"].astype(int)
    counts = counts.sort_values("group")
    counts["color"] = counts["group"].map(COLOR_MAP_ERROR).fillna("gray")
    mean_err = df["error"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        counts["group"].astype(str), counts["proportion"],
        color=counts["color"].tolist(), edgecolor="white", linewidth=1.2, alpha=0.92,
    )
    ax.set_title("Error proportion per severity group", pad=12)
    ax.set_xlabel("Error severity group")
    ax.set_ylabel("Proportion")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    y_max = max(b.get_height() for b in bars)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10, color="#2b2f36")
    ax.text(0.98, 0.95, f"Mean: {mean_err:.4f}", transform=ax.transAxes,
            fontsize=11, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f6f8", edgecolor="#d0d4da"))
    ax.set_ylim(0, y_max * 1.15)
    plt.savefig(out_dir / "barplot_error.png")
    plt.close()

    plot_data = (
        df.groupby(["real_color_group", "color_group", "error"])
        .size().reset_index(name="count")
    )
    heatmap = plot_data.pivot_table(
        index="real_color_group", columns="color_group", values="count", aggfunc="sum",
    )
    err_pivot = plot_data.pivot_table(
        index="real_color_group", columns="color_group", values="error", aggfunc="first",
    )
    sizes = heatmap / heatmap.max().max() * 12000

    _, ax = plt.subplots(figsize=(10, 8))
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            cnt = heatmap.iloc[i, j]
            if pd.notna(cnt):
                ax.scatter(
                    j, i, s=sizes.iloc[i, j],
                    color=COLOR_MAP_ERROR.get(int(err_pivot.iloc[i, j]), "gray"),
                    alpha=0.6, edgecolor="black",
                )
                ax.text(j, i, int(cnt), color="black", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_xlim(-0.5, len(heatmap.columns) - 0.5)
    ax.set_ylim(len(heatmap.index) - 0.5, -0.5)
    ax.set_title("Predicted vs. real category (bubble size = count)", pad=12)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Real category")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(out_dir / "heatmap_error.png")
    plt.close()
