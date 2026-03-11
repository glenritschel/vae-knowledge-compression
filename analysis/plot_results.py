"""
plot_results.py

Generate all figures for the paper from saved CSV results.
Run this after run_experiment.py has completed.

Usage:
    python analysis/plot_results.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.utils import load_config

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
FACTOR_COLORS = {
    "F1_dominant":       "#2166ac",
    "F2_secondary":      "#4dac26",
    "F3_minority_signal":"#d6604d",
    "F4_rare_subtype":   "#b2182b",
    "F5_correlated_pair":"#8073ac",
}
FACTOR_LABELS = {
    "F1_dominant":        "F1 Dominant",
    "F2_secondary":       "F2 Secondary",
    "F3_minority_signal": "F3 Minority Signal",
    "F4_rare_subtype":    "F4 Rare Subtype (5%)",
    "F5_correlated_pair": "F5 Correlated Pair",
}


def load_results(results_dir: str):
    raw = pd.read_csv(os.path.join(results_dir, "raw_results.csv"))
    summary = pd.read_csv(os.path.join(results_dir, "summary_results.csv"))
    with open(os.path.join(results_dir, "training_histories.json")) as f:
        histories = json.load(f)
    return raw, summary, histories


def _mean_std(raw: pd.DataFrame, col: str):
    """Return (latent_dims, means, stds) for a column grouped by latent_dim."""
    g = raw.groupby("latent_dim")[col]
    dims = sorted(raw["latent_dim"].unique(), reverse=True)
    means = [g.get_group(d).mean() for d in dims]
    stds = [g.get_group(d).std() for d in dims]
    return np.array(dims), np.array(means), np.array(stds)


def fig1_overview(raw, figures_dir):
    """
    Figure 1: Four-panel overview.
      A — Overall MSE vs latent dim
      B — ELBO (val_loss) vs latent dim
      C — Active dimensions vs latent dim
      D — Silhouette (rare subtype) vs latent dim
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Figure 1: VAE Knowledge Compression — Overview",
        fontsize=14, fontweight="bold", y=1.01
    )

    panels = [
        ("mse_overall",        "MSE (Overall Reconstruction)",    "A", axes[0, 0], False),
        ("val_loss",           "ELBO Validation Loss",            "B", axes[0, 1], False),
        ("n_active_dims",      "Active Latent Dimensions",        "C", axes[1, 0], False),
        ("silhouette_rare",    "Silhouette Score (Rare Subtype)", "D", axes[1, 1], False),
    ]

    for col, ylabel, label, ax, invert in panels:
        if col not in raw.columns:
            ax.set_visible(False)
            continue
        dims, means, stds = _mean_std(raw, col)
        ax.plot(dims, means, "o-", color="#1a1a2e", linewidth=2, markersize=7)
        ax.fill_between(dims, means - stds, means + stds, alpha=0.2, color="#1a1a2e")
        ax.set_xlabel("Latent Dimension", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"({label})", loc="left", fontsize=12, fontweight="bold")
        ax.invert_xaxis()
        if col == "n_active_dims":
            # Overlay the y=x line (ideal: all dims active)
            ax.plot(dims, dims, "--", color="gray", linewidth=1.2, label="y=x (ideal)")
            ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(figures_dir, "fig1_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig2_factor_recovery(raw, figures_dir):
    """
    Figure 2: R² per factor vs. latent dim.
    Shows which factor is lost first.
    """
    factor_cols = [c for c in raw.columns if c.startswith("r2_F")]

    if not factor_cols:
        print("No factor R² columns found, skipping fig2.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    dims_sorted = sorted(raw["latent_dim"].unique(), reverse=True)

    for col in sorted(factor_cols):
        fname = col.replace("r2_", "")
        label = FACTOR_LABELS.get(fname, fname)
        color = FACTOR_COLORS.get(fname, "gray")

        g = raw.groupby("latent_dim")[col]
        means = [g.get_group(d).mean() for d in dims_sorted]
        stds = [g.get_group(d).std() for d in dims_sorted]

        ax.plot(dims_sorted, means, "o-", color=color, linewidth=2,
                markersize=7, label=label)
        ax.fill_between(
            dims_sorted,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.15, color=color
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("Factor Recovery R²", fontsize=12)
    ax.set_title(
        "Figure 2: Factor Recovery by Latent Dimension\n"
        "R² from linear probe: latent space → true factor score",
        fontsize=12, fontweight="bold"
    )
    ax.invert_xaxis()
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    path = os.path.join(figures_dir, "fig2_factor_recovery.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig3_mse_per_group(raw, figures_dir):
    """
    Figure 3: Per-group MSE vs. latent dim.
    Shows which feature group's reconstruction degrades first.
    """
    mse_cols = [c for c in raw.columns if c.startswith("mse_F")]

    if not mse_cols:
        print("No per-group MSE columns found, skipping fig3.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    dims_sorted = sorted(raw["latent_dim"].unique(), reverse=True)

    for col in sorted(mse_cols):
        fname = col.replace("mse_", "")
        label = FACTOR_LABELS.get(fname, fname)
        color = FACTOR_COLORS.get(fname, "gray")

        g = raw.groupby("latent_dim")[col]
        means = [g.get_group(d).mean() for d in dims_sorted]
        stds = [g.get_group(d).std() for d in dims_sorted]

        ax.plot(dims_sorted, means, "o-", color=color, linewidth=2,
                markersize=7, label=label)
        ax.fill_between(
            dims_sorted,
            np.clip(np.array(means) - np.array(stds), 0, None),
            np.array(means) + np.array(stds),
            alpha=0.15, color=color
        )

    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("MSE (per feature group)", fontsize=12)
    ax.set_title(
        "Figure 3: Reconstruction Error by Feature Group\n"
        "Which factor's features are reconstructed worst as capacity shrinks?",
        fontsize=12, fontweight="bold"
    )
    ax.invert_xaxis()
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    path = os.path.join(figures_dir, "fig3_mse_per_group.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig4_collapse_heatmap(raw, figures_dir):
    """
    Figure 4: Posterior collapse — n_active_dims vs. nominal latent dim.
    Also plots active_fraction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dims_sorted = sorted(raw["latent_dim"].unique(), reverse=True)

    # Panel A: active dims vs nominal
    ax = axes[0]
    g = raw.groupby("latent_dim")["n_active_dims"]
    means = [g.get_group(d).mean() for d in dims_sorted]
    stds = [g.get_group(d).std() for d in dims_sorted]

    ax.plot(dims_sorted, means, "o-", color="#e08214", linewidth=2, markersize=7,
            label="Active dims (mean)")
    ax.fill_between(dims_sorted,
                    np.clip(np.array(means) - np.array(stds), 0, None),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color="#e08214")
    ax.plot(dims_sorted, dims_sorted, "--", color="gray", linewidth=1.2,
            label="y=x (all active)")
    ax.set_xlabel("Nominal Latent Dimension", fontsize=11)
    ax.set_ylabel("Active Latent Dimensions", fontsize=11)
    ax.set_title("(A) Active vs. Nominal Latent Dims", fontweight="bold")
    ax.invert_xaxis()
    ax.legend(fontsize=9)

    # Panel B: active fraction
    ax = axes[1]
    g2 = raw.groupby("latent_dim")["active_fraction"]
    means2 = [g2.get_group(d).mean() for d in dims_sorted]
    stds2 = [g2.get_group(d).std() for d in dims_sorted]

    ax.plot(dims_sorted, means2, "o-", color="#542788", linewidth=2, markersize=7)
    ax.fill_between(dims_sorted,
                    np.clip(np.array(means2) - np.array(stds2), 0, None),
                    np.clip(np.array(means2) + np.array(stds2), None, 1),
                    alpha=0.2, color="#542788")
    ax.axhline(1.0, color="gray", linewidth=1.2, linestyle="--", label="100% active")
    ax.set_xlabel("Nominal Latent Dimension", fontsize=11)
    ax.set_ylabel("Fraction of Active Dimensions", fontsize=11)
    ax.set_title("(B) Active Dimension Fraction", fontweight="bold")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)

    fig.suptitle("Figure 4: Posterior Collapse as Latent Capacity Shrinks",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(figures_dir, "fig4_posterior_collapse.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig5_elbo_blindspot(raw, figures_dir):
    """
    Figure 5: The ELBO blindspot.
    ELBO (val_loss) vs. R² for rare subtype.
    Shows ELBO stays flat while rare subtype knowledge is lost.
    """
    if "r2_F4_rare_subtype" not in raw.columns:
        print("No F4 R² column found, skipping fig5.")
        return

    dims_sorted = sorted(raw["latent_dim"].unique(), reverse=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    g_elbo = raw.groupby("latent_dim")["val_loss"]
    elbo_means = [g_elbo.get_group(d).mean() for d in dims_sorted]
    elbo_stds = [g_elbo.get_group(d).std() for d in dims_sorted]

    g_r2 = raw.groupby("latent_dim")["r2_F4_rare_subtype"]
    r2_means = [g_r2.get_group(d).mean() for d in dims_sorted]
    r2_stds = [g_r2.get_group(d).std() for d in dims_sorted]

    l1 = ax1.plot(dims_sorted, elbo_means, "o-", color="#1a1a2e", linewidth=2,
                  markersize=7, label="ELBO (val loss)")
    ax1.fill_between(dims_sorted,
                     np.array(elbo_means) - np.array(elbo_stds),
                     np.array(elbo_means) + np.array(elbo_stds),
                     alpha=0.15, color="#1a1a2e")

    l2 = ax2.plot(dims_sorted, r2_means, "s--", color="#b2182b", linewidth=2,
                  markersize=7, label="R² (F4 Rare Subtype)")
    ax2.fill_between(dims_sorted,
                     np.clip(np.array(r2_means) - np.array(r2_stds), -0.1, None),
                     np.array(r2_means) + np.array(r2_stds),
                     alpha=0.15, color="#b2182b")

    ax1.set_xlabel("Latent Dimension", fontsize=12)
    ax1.set_ylabel("ELBO Validation Loss", fontsize=12, color="#1a1a2e")
    ax2.set_ylabel("Rare Subtype R² (factor recovery)", fontsize=12, color="#b2182b")
    ax1.invert_xaxis()

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc="center left")

    ax1.set_title(
        "Figure 5: The ELBO Blindspot\n"
        "ELBO remains stable while rare subtype knowledge is silently lost",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    path = os.path.join(figures_dir, "fig5_elbo_blindspot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    config = load_config()
    results_dir = config["experiment"]["results_dir"]
    figures_dir = config["experiment"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading results...")
    raw, summary, histories = load_results(results_dir)

    print(f"Loaded {len(raw)} rows across "
          f"{raw['latent_dim'].nunique()} latent dims, "
          f"{raw['seed'].nunique()} seeds\n")

    print("Generating figures...")
    fig1_overview(raw, figures_dir)
    fig2_factor_recovery(raw, figures_dir)
    fig3_mse_per_group(raw, figures_dir)
    fig4_collapse_heatmap(raw, figures_dir)
    fig5_elbo_blindspot(raw, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")
    print("figures: fig1_overview.png, fig2_factor_recovery.png, "
          "fig3_mse_per_group.png, fig4_posterior_collapse.png, "
          "fig5_elbo_blindspot.png")


if __name__ == "__main__":
    main()
