"""
utils.py — shared utilities for the VAE knowledge compression experiment.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Any, Dict


def load_config(path: str = "configs/experiment_config.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def flatten_probe_results(
    latent_dim: int,
    seed: int,
    probe_results: Dict,
) -> Dict:
    """
    Flatten nested probe results into a single flat dict row
    suitable for appending to a DataFrame.
    """
    row = {"latent_dim": latent_dim, "seed": seed}

    # Probe 1 — reconstruction
    r = probe_results.get("reconstruction", {})
    row["mse_overall"] = r.get("mse_overall", np.nan)
    for k, v in r.get("mse_per_group", {}).items():
        row[f"mse_{k}"] = v

    # Probe 2 — covariance
    c = probe_results.get("covariance", {})
    row["frobenius_distance"] = c.get("frobenius_distance", np.nan)
    row["normalized_frobenius"] = c.get("normalized_frobenius", np.nan)

    # Probe 3 — factor recovery
    fr = probe_results.get("factor_recovery", {})
    for k, v in fr.get("r2_per_factor", {}).items():
        row[f"r2_{k}"] = v

    # Probe 4 — cluster separability
    cl = probe_results.get("cluster", {})
    row["silhouette_rare"] = cl.get("silhouette_rare", np.nan)

    # Probe 5 — posterior collapse
    pc = probe_results.get("collapse", {})
    row["n_active_dims"] = pc.get("n_active", np.nan)
    row["n_collapsed_dims"] = pc.get("n_collapsed", np.nan)
    row["active_fraction"] = pc.get("active_fraction", np.nan)
    row["mean_kl"] = pc.get("mean_kl", np.nan)
    row["max_kl"] = pc.get("max_kl", np.nan)

    return row


def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with one row per (latent_dim, seed),
    return mean ± std across seeds, grouped by latent_dim.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["latent_dim", "seed"]]

    agg = df.groupby("latent_dim")[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["latent_dim"] + [
        f"{col}_{stat}" for col, stat in agg.columns[1:]
    ]
    return agg
