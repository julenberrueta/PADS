"""All plotting routines."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pads.data.schema import N_TIME_OFFSETS
from pads.eval.metrics import evaluate
from pads.eval.thresholds import ThresholdMethod, optimal_threshold

COLOR_MAP_GROUPS = {
    "EXITUS <48h": "#f43543",
    "EXITUS >48h": "#ff9e30",
    "ALIVE <48h": "#3ab830",
    "ALIVE >48h": "#1097d8",
}
COLOR_MAP_ERROR = {0: "lightgreen", 1: "lightyellow", 2: "lightsalmon", 3: "lightcoral"}


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

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(100 * m.fp_curve, 100 * m.tp_curve, color="#d4453b", lw=2, label=f"Mortality (AUC={m.auc:.2f})")
    ax.scatter(100 * m.opt_fp, 100 * m.opt_tp, color="#d4453b", edgecolor="black", zorder=5)
    ax.plot(100 * d.fp_curve, 100 * d.tp_curve, color="#298fcf", lw=2, label=f"Discharge (AUC={d.auc:.2f})")
    ax.scatter(100 * d.opt_fp, 100 * d.opt_tp, color="#298fcf", edgecolor="black", zorder=5)
    ax.set_title(
        f"ROC — Mortality & Discharge  |  threshold: {threshold_method}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("False Positives [%]")
    ax.set_ylabel("True Positives [%]")
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.grid(True)
    ax.set_aspect("equal")
    ax.legend(loc="center right", fontsize=10)

    def _row(label, x):
        return f"{label:<10} {x.accuracy:8.2f}  {x.precision:9.2f}  {x.recall:6.2f}  {x.f1:8.2f}  {x.auc:4.2f}  {x.threshold:.4f}"

    table = (
        "           Accuracy  Precision  Recall  F1 Score  AUC  Threshold\n"
        + _row("Mortality", m) + "\n"
        + _row("Discharge", d)
    )
    fig.text(0.5, -0.01, table, ha="center", va="bottom", fontsize=10,
             family="monospace", bbox=dict(facecolor="#f0f0f0", edgecolor="gray"))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save:
        suffix = f"_{inference_type}" if inference_type else ""
        plt.savefig(out_dir / f"roc_combined{suffix}.png", bbox_inches="tight")
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

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        counts["group"].astype(str), counts["proportion"],
        color=counts["color"].tolist(), edgecolor="black", alpha=0.6,
    )
    plt.title("Error Percentage per Error Group")
    plt.xlabel("Group")
    plt.ylabel("Proportion")
    y_max = max(b.get_height() for b in bars)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom")
    plt.text(len(counts) - 1.1, y_max * 1.05, f"Mean: {mean_err:.4f}", fontsize=10, ha="right")
    plt.ylim(0, y_max * 1.15)
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
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_xlim(-0.5, len(heatmap.columns) - 0.5)
    ax.set_ylim(len(heatmap.index) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(out_dir / "heatmap_error.png")
    plt.close()
