"""
visualisation.py
================
All plotting and table-export functions for the pipeline.

Outputs
-------
1. results/similarity_histogram_<method>.png
2. results/coverage_vs_threshold_<method>.png
3. results/ablation_comparison.png
4. results/summary_table.csv
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — safe on all platforms
import matplotlib.pyplot as plt



RESULTS_DIR = Path("results")
PALETTE = {
    "primary": "#2563EB",      # blue
    "secondary": "#10B981",    # emerald
    "accent": "#F59E0B",       # amber
    "danger": "#EF4444",       # red
    "bg": "#0F172A",           # slate-900
    "panel": "#1E293B",        # slate-800
    "text": "#E2E8F0",         # slate-200
    "grid": "#334155",         # slate-700
}


def _style_axis(ax: plt.Axes) -> None:
    """Apply dark-theme styling to a single axes."""
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, linestyle="--", alpha=0.6)
    if ax.get_title():
        ax.title.set_color(PALETTE["text"])


def save_similarity_histogram(
    nn_scores: np.ndarray,
    method_label: str,
    threshold: float,
    filename: str = "similarity_histogram",
) -> Path:
    """
    Plot the distribution of nearest-neighbour cosine similarities.

    Parameters
    ----------
    nn_scores    : array (n1,) of similarity values
    method_label : e.g. 'K-mer (k=4)' or 'Projected Embedding'
    threshold    : the chosen coverage threshold (drawn as a vertical line)
    filename     : output filename stem
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{filename}.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE["bg"])
    _style_axis(ax)

    n, bins, patches = ax.hist(
        nn_scores, bins=50, color=PALETTE["primary"], edgecolor=PALETTE["bg"],
        alpha=0.85, density=True,
    )

    # Colour bars above threshold in green
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= threshold:
            patch.set_facecolor(PALETTE["secondary"])

    ax.axvline(threshold, color=PALETTE["accent"], linewidth=1.8,
               linestyle="--", label=f"Threshold = {threshold:.2f}")
    ax.axvline(nn_scores.mean(), color=PALETTE["danger"], linewidth=1.4,
               linestyle=":", label=f"Mean = {nn_scores.mean():.3f}")

    covered_pct = (nn_scores >= threshold).mean() * 100
    ax.set_xlabel("Cosine Similarity (D1 → nearest neighbour in D2)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Nearest-Neighbour Similarity Distribution  [{method_label}]\n"
        f"Coverage @ {threshold:.0%} threshold: {covered_pct:.1f}% of D1",
        fontsize=11, pad=10,
    )
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_coverage_curve(
    thresholds_kmer: np.ndarray,
    coverage_kmer: np.ndarray,
    thresholds_embed: np.ndarray | None = None,
    coverage_embed: np.ndarray | None = None,
    filename: str = "coverage_vs_threshold",
) -> Path:
    """
    Coverage vs threshold curve for one or two representation methods.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{filename}.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE["bg"])
    _style_axis(ax)

    ax.plot(thresholds_kmer, coverage_kmer * 100, color=PALETTE["primary"],
            linewidth=2.2, label="K-mer vectors")
    if thresholds_embed is not None and coverage_embed is not None:
        ax.plot(thresholds_embed, coverage_embed * 100, color=PALETTE["secondary"],
                linewidth=2.2, linestyle="--", label="Projected embedding")

    ax.set_xlabel("Similarity Threshold τ", fontsize=11)
    ax.set_ylabel("Coverage of D1 (%)", fontsize=11)
    ax.set_title("Coverage of D1 by D2 vs Similarity Threshold", fontsize=12, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

    # Mark 90% and 50% threshold guide lines
    for y in (90, 50):
        ax.axhline(y, color=PALETTE["grid"], linewidth=0.8, linestyle=":")

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_ablation_plot(
    ablation_results: Dict[str, Dict[str, Any]],
    filename: str = "ablation_comparison",
) -> Path:
    """
    Bar chart comparing key metrics across ablation conditions and methods.

    ablation_results : dict keyed by label (e.g. 'K-mer', 'Embedding'),
                       values are the output of ablation_information_retained()
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{filename}.png"

    labels = list(ablation_results.keys())
    coverage_vals = [ablation_results[l]["nn_result"]["coverage_pct"] for l in labels]
    mean_sim_vals = [ablation_results[l]["nn_result"]["mean_similarity"] * 100 for l in labels]
    redundancy_vals = [ablation_results[l]["redundancy_score"] * 100 for l in labels]
    recon_r2_vals = [
        (1 - ablation_results[l]["norm_mse"]) * 100 for l in labels
    ]

    x = np.arange(len(labels))
    width = 0.2
    colours = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["danger"]]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["bg"])
    _style_axis(ax)

    metrics = [
        ("Coverage @ τ=0.9 (%)", coverage_vals),
        ("Mean Similarity × 100", mean_sim_vals),
        ("Redundancy Score (%)", redundancy_vals),
        ("Recon. Quality (1-norm_MSE) %", recon_r2_vals),
    ]
    for i, (metric_name, vals) in enumerate(metrics):
        bars = ax.bar(x + i * width, vals, width, label=metric_name,
                      color=colours[i], alpha=0.85, edgecolor=PALETTE["bg"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=PALETTE["text"])

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Ablation: Information Retained in D2 After Removing D1",
                 fontsize=12, pad=10)
    ax.set_ylim(0, 115)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_size_sensitivity_plot(
    size_results: List[Tuple[int, float, float]],
    filename: str = "size_sensitivity",
) -> Path:
    """
    How does redundancy score change as D1 size grows?

    size_results : list of (n_d1, redundancy_kmer, redundancy_embed)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{filename}.png"

    ns, red_kmer, red_embed = zip(*size_results)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE["bg"])
    _style_axis(ax)

    ax.plot(ns, [r * 100 for r in red_kmer], "o-", color=PALETTE["primary"],
            linewidth=2, markersize=6, label="K-mer")
    ax.plot(ns, [r * 100 for r in red_embed], "s--", color=PALETTE["secondary"],
            linewidth=2, markersize=6, label="Projected embedding")

    ax.set_xlabel("D1 Sample Size (n)", fontsize=11)
    ax.set_ylabel("Redundancy Score (%)", fontsize=11)
    ax.set_title("Redundancy Score vs D1 Dataset Size", fontsize=12, pad=10)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_validation_plot(
    score_high: float,
    score_low: float,
    filename: str = "validation_sanity_check",
) -> Path:
    """
    Bar chart comparing redundancy scores for the two sanity-check conditions:
      HIGH (D1 is a subset of D2) vs LOW (D1 disjoint from D2).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{filename}.png"

    labels = ["HIGH\n(D1 subset of D2)\nexpect ~1.0",
              "LOW\n(D1 disjoint D2)\nexpect ~0.0"]
    values = [score_high * 100, score_low * 100]
    colours = [PALETTE["secondary"], PALETTE["danger"]]

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=PALETTE["bg"])
    _style_axis(ax)

    bars = ax.bar(labels, values, color=colours, width=0.45,
                  edgecolor=PALETTE["bg"], alpha=0.9)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=PALETTE["text"])

    ax.set_ylim(0, 115)
    ax.set_ylabel("Redundancy Score (%)", fontsize=11)
    ax.set_title("Metric Validation: HIGH vs LOW Redundancy Conditions",
                 fontsize=11, pad=10)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_summary_table(
    ablation_results: Dict[str, Dict[str, Any]],
    dataset_sizes: Dict[str, int],
) -> pd.DataFrame:
    """
    Assemble the summary CSV table.

    Columns: Method | D1 size | D2 size | Coverage@0.9 | Mean Sim | MSE | Redundancy Score
    """
    rows = []
    for method, res in ablation_results.items():
        rows.append({
            "Method": method,
            "D1 Size": dataset_sizes["n_d1"],
            "D2 Size": dataset_sizes["n_d2"],
            "Coverage @ t=0.90 (%)": round(res["nn_result"]["coverage_pct"], 2),
            "Mean NN Similarity": round(res["nn_result"]["mean_similarity"], 4),
            "Std NN Similarity": round(res["nn_result"]["std_similarity"], 4),
            "Mean Recon MSE": round(res["recon_result"]["mean_mse"], 6),
            "Baseline MSE": round(res["recon_result"]["baseline_mse"], 6),
            "Normalised MSE": round(res["norm_mse"], 4),
            "Redundancy Score": round(res["redundancy_score"], 4),
        })
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    return df
