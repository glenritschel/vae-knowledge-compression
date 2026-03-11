"""
probes.py

Five knowledge probes applied to a trained VAE.

Probe 1 — Global reconstruction fidelity (MSE per feature group)
Probe 2 — Covariance structure recovery (Frobenius distance)
Probe 3 — Factor recovery via linear probe (R² per factor)
Probe 4 — Cluster separability (Silhouette score for rare subtype)
Probe 5 — Posterior collapse (active latent dimensions)
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

from src.vae_model import VAE
from src.data_generator import SyntheticDataset


# ─── helpers ──────────────────────────────────────────────────────────────────

def _encode_dataset(model: VAE, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return latent mu for all samples in X."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encoder(X_t)
    return mu.cpu().numpy()


def _decode_latent(model: VAE, Z: np.ndarray, device: torch.device) -> np.ndarray:
    """Decode a matrix of latent vectors."""
    model.eval()
    Z_t = torch.tensor(Z, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_recon = model.decoder(Z_t)
    return X_recon.cpu().numpy()


# ─── Probe 1: Global reconstruction fidelity ──────────────────────────────────

def probe_reconstruction(
    model: VAE,
    dataset: SyntheticDataset,
    device: torch.device,
) -> Dict:
    """
    Compute MSE overall and per factor group.
    Returns dict: {
        "mse_overall": float,
        "mse_per_group": {factor_name: float},
    }
    """
    Z = _encode_dataset(model, dataset.X, device)
    X_recon = _decode_latent(model, Z, device)

    mse_overall = float(np.mean((dataset.X - X_recon) ** 2))

    # Per-group MSE: average MSE only over features primarily driven by that factor
    n_factors = len(dataset.factor_names)
    mse_per_group = {}
    for fi, name in enumerate(dataset.factor_names):
        feat_indices = [
            i for i, a in enumerate(dataset.feature_assignments) if a == fi
        ]
        if feat_indices:
            mse_per_group[name] = float(
                np.mean((dataset.X[:, feat_indices] - X_recon[:, feat_indices]) ** 2)
            )
        else:
            mse_per_group[name] = np.nan

    return {"mse_overall": mse_overall, "mse_per_group": mse_per_group}


# ─── Probe 2: Covariance structure recovery ────────────────────────────────────

def probe_covariance(
    model: VAE,
    dataset: SyntheticDataset,
    device: torch.device,
) -> Dict:
    """
    Compare correlation matrix of real data vs. reconstructed data.
    Uses Frobenius norm of the difference as the score.
    Lower = better structure recovery.
    """
    Z = _encode_dataset(model, dataset.X, device)
    X_recon = _decode_latent(model, Z, device)

    corr_real = np.corrcoef(dataset.X, rowvar=False)
    corr_recon = np.corrcoef(X_recon, rowvar=False)

    # Replace NaNs (can occur if a feature has zero variance)
    corr_real = np.nan_to_num(corr_real)
    corr_recon = np.nan_to_num(corr_recon)

    frobenius_distance = float(np.linalg.norm(corr_real - corr_recon, "fro"))
    n = corr_real.shape[0]
    # Normalize by matrix size so it's comparable across feature counts
    normalized_distance = frobenius_distance / n

    return {
        "frobenius_distance": frobenius_distance,
        "normalized_frobenius": normalized_distance,
    }


# ─── Probe 3: Factor recovery via linear probe ─────────────────────────────────

def probe_factor_recovery(
    model: VAE,
    dataset: SyntheticDataset,
    device: torch.device,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Dict:
    """
    For each true generative factor, train a linear regression from
    the latent space (mu) to the factor score. Report R² per factor.

    R² near 1.0 = factor is explicitly encoded in the latent space.
    R² near 0.0 = factor has been lost.
    """
    Z = _encode_dataset(model, dataset.X, device)

    rng = np.random.RandomState(seed)
    n = Z.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    Z_train, Z_val = Z[train_idx], Z[val_idx]
    F_train = dataset.factor_scores[train_idx]
    F_val = dataset.factor_scores[val_idx]

    r2_per_factor = {}
    for fi, name in enumerate(dataset.factor_names):
        y_train = F_train[:, fi]
        y_val = F_val[:, fi]

        # Skip if factor is near-constant (rare subtype with very few actives)
        if y_train.std() < 1e-6:
            r2_per_factor[name] = np.nan
            continue

        reg = Ridge(alpha=1.0)
        reg.fit(Z_train, y_train)
        y_pred = reg.predict(Z_val)
        r2_per_factor[name] = float(r2_score(y_val, y_pred))

    return {"r2_per_factor": r2_per_factor}


# ─── Probe 4: Cluster separability ─────────────────────────────────────────────

def probe_cluster_separability(
    model: VAE,
    dataset: SyntheticDataset,
    device: torch.device,
) -> Dict:
    """
    Measure how separable the rare subtype cluster is in latent space.

    Uses Silhouette score comparing rare-subtype samples vs. all others.
    Score in [-1, 1]; higher = more separable = rare subtype still encoded.

    If too few rare samples exist, returns NaN.
    """
    Z = _encode_dataset(model, dataset.X, device)
    labels = dataset.rare_mask.astype(int)  # 1 = rare subtype, 0 = common

    n_rare = labels.sum()
    n_common = (labels == 0).sum()

    if n_rare < 2 or n_common < 2:
        return {"silhouette_rare": np.nan, "n_rare": int(n_rare)}

    try:
        sil = float(silhouette_score(Z, labels, sample_size=min(2000, len(Z))))
    except Exception:
        sil = np.nan

    return {"silhouette_rare": sil, "n_rare": int(n_rare)}


# ─── Probe 5: Posterior collapse ───────────────────────────────────────────────

def probe_posterior_collapse(
    model: VAE,
    dataset: SyntheticDataset,
    device: torch.device,
    collapse_threshold: float = 0.1,
) -> Dict:
    """
    Measure per-dimension KL divergence from the prior.
    Dimensions with KL < threshold are considered collapsed (carry no info).

    Returns:
        kl_per_dim: array of KL values per latent dimension
        n_active: number of non-collapsed dimensions
        n_collapsed: number of collapsed dimensions
        active_fraction: fraction of latent dims that are active
    """
    model.eval()
    X_t = torch.tensor(dataset.X, dtype=torch.float32).to(device)
    kl_per_dim = model.get_latent_kl_per_dim(X_t).cpu().numpy()

    n_active = int((kl_per_dim >= collapse_threshold).sum())
    n_collapsed = int((kl_per_dim < collapse_threshold).sum())
    total = len(kl_per_dim)

    return {
        "kl_per_dim": kl_per_dim.tolist(),
        "n_active": n_active,
        "n_collapsed": n_collapsed,
        "active_fraction": n_active / total if total > 0 else 0.0,
        "mean_kl": float(kl_per_dim.mean()),
        "max_kl": float(kl_per_dim.max()),
    }


# ─── Run all probes ────────────────────────────────────────────────────────────

def run_all_probes(
    model: VAE,
    dataset: SyntheticDataset,
    device: torch.device,
    seed: int = 0,
) -> Dict:
    """Run all five probes and return a unified results dict."""
    results = {}
    results["reconstruction"] = probe_reconstruction(model, dataset, device)
    results["covariance"] = probe_covariance(model, dataset, device)
    results["factor_recovery"] = probe_factor_recovery(model, dataset, device, seed=seed)
    results["cluster"] = probe_cluster_separability(model, dataset, device)
    results["collapse"] = probe_posterior_collapse(model, dataset, device)
    return results
