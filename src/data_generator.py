"""
data_generator.py

Generates synthetic structured data with known generative factors.
Ground truth factor scores are retained so probes can measure
exactly how well each factor is recovered at each compression level.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd


@dataclass
class FactorSpec:
    name: str
    n_features_affected: int
    variance: float
    prevalence: float          # fraction of samples where this factor is active


@dataclass
class SyntheticDataset:
    X: np.ndarray              # (n_samples, n_features) — the observed data
    factor_scores: np.ndarray  # (n_samples, n_factors) — ground truth factor values
    factor_names: List[str]
    loading_matrix: np.ndarray # (n_features, n_factors) — how factors map to features
    feature_assignments: List[int]  # which factor primarily drives each feature
    rare_mask: np.ndarray      # (n_samples,) bool — which samples have F4 rare subtype


def build_factor_specs(config: dict) -> List[FactorSpec]:
    specs = []
    for f in config["data"]["factors"]:
        specs.append(FactorSpec(
            name=f["name"],
            n_features_affected=f["n_features_affected"],
            variance=f["variance"],
            prevalence=f["prevalence"],
        ))
    return specs


def generate_dataset(config: dict, seed: int = 42) -> SyntheticDataset:
    """
    Generate a synthetic dataset with planted generative structure.

    Each factor has a set of features it primarily loads onto.
    Features not assigned to any factor receive only noise.
    The rare subtype factor (prevalence < 1) is active in only
    a fraction of samples, simulating a minority cluster.
    """
    rng = np.random.RandomState(seed)

    n_samples = config["data"]["n_samples"]
    n_features = config["data"]["n_features"]
    noise_level = config["data"]["noise_level"]
    factor_specs = build_factor_specs(config)
    n_factors = len(factor_specs)

    # --- Build loading matrix ---
    # Assign contiguous feature blocks to each factor
    loading_matrix = np.zeros((n_features, n_factors))
    feature_assignments = [-1] * n_features  # -1 = noise only

    cursor = 0
    for fi, spec in enumerate(factor_specs):
        n = spec.n_features_affected
        end = min(cursor + n, n_features)
        for feat_idx in range(cursor, end):
            # Random loading weight in [0.6, 1.0], positive
            loading_matrix[feat_idx, fi] = rng.uniform(0.6, 1.0)
            feature_assignments[feat_idx] = fi
        cursor = end

    # --- Generate factor scores ---
    factor_scores = np.zeros((n_samples, n_factors))
    rare_mask = np.zeros(n_samples, dtype=bool)

    for fi, spec in enumerate(factor_specs):
        if spec.prevalence >= 1.0:
            # Active in all samples — continuous factor
            factor_scores[:, fi] = rng.normal(
                loc=0.0,
                scale=np.sqrt(spec.variance),
                size=n_samples
            )
        else:
            # Rare subtype — active in only prevalence fraction of samples
            active = rng.rand(n_samples) < spec.prevalence
            rare_mask = active  # track which samples have this factor
            factor_scores[:, fi] = np.where(
                active,
                rng.normal(loc=2.0, scale=np.sqrt(spec.variance), size=n_samples),
                0.0
            )

    # --- Compose observed data ---
    # X = factor_scores @ loading_matrix.T + noise
    signal = factor_scores @ loading_matrix.T  # (n_samples, n_features)
    noise = rng.normal(0, noise_level, size=(n_samples, n_features))
    X = signal + noise

    # Standardize to zero mean, unit variance per feature
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    return SyntheticDataset(
        X=X.astype(np.float32),
        factor_scores=factor_scores.astype(np.float32),
        factor_names=[s.name for s in factor_specs],
        loading_matrix=loading_matrix.astype(np.float32),
        feature_assignments=feature_assignments,
        rare_mask=rare_mask,
    )


def train_val_split(
    dataset: SyntheticDataset,
    val_fraction: float = 0.15,
    seed: int = 42
) -> Tuple[SyntheticDataset, SyntheticDataset]:
    """Split a SyntheticDataset into train and validation sets."""
    rng = np.random.RandomState(seed)
    n = dataset.X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    def subset(ds, indices):
        return SyntheticDataset(
            X=ds.X[indices],
            factor_scores=ds.factor_scores[indices],
            factor_names=ds.factor_names,
            loading_matrix=ds.loading_matrix,
            feature_assignments=ds.feature_assignments,
            rare_mask=ds.rare_mask[indices],
        )

    return subset(dataset, train_idx), subset(dataset, val_idx)


def dataset_summary(dataset: SyntheticDataset) -> pd.DataFrame:
    """Return a DataFrame summarising the dataset's factor statistics."""
    rows = []
    for fi, name in enumerate(dataset.factor_names):
        scores = dataset.factor_scores[:, fi]
        active = scores != 0
        rows.append({
            "factor": name,
            "n_active_samples": int(active.sum()),
            "prevalence": float(active.mean()),
            "mean_score": float(scores[active].mean()) if active.any() else 0.0,
            "std_score": float(scores[active].std()) if active.any() else 0.0,
        })
    return pd.DataFrame(rows)
