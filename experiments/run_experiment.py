"""
run_experiment.py

Main entry point for the VAE knowledge compression experiment.

For each latent dimension in the sweep, and for each random seed:
  1. Generate a synthetic dataset with planted generative factors
  2. Train a VAE
  3. Run all five knowledge probes
  4. Save results to CSV

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --config configs/experiment_config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_generator import generate_dataset, train_val_split, dataset_summary
from src.vae_model import VAE
from src.trainer import train_vae
from src.probes import run_all_probes
from src.utils import (
    load_config, ensure_dirs, save_json,
    flatten_probe_results, summarise_results
)


def main(config_path: str = "configs/experiment_config.yaml"):
    config = load_config(config_path)

    results_dir = config["experiment"]["results_dir"]
    figures_dir = config["experiment"]["figures_dir"]
    ensure_dirs(results_dir, figures_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latent_dims = config["experiment"]["latent_dims"]
    n_seeds = config["data"]["n_seeds"]
    base_seed = config["experiment"]["random_seed"]

    all_rows = []
    all_histories = {}

    print(f"\nSweeping {len(latent_dims)} latent dims × {n_seeds} seeds "
          f"= {len(latent_dims) * n_seeds} total runs\n")

    for latent_dim in latent_dims:
        print(f"{'='*60}")
        print(f"Latent dim: {latent_dim}")
        print(f"{'='*60}")

        dim_histories = []

        for seed_idx in range(n_seeds):
            data_seed = base_seed + seed_idx * 100
            model_seed = base_seed + seed_idx

            # --- Generate data ---
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)

            dataset = generate_dataset(config, seed=data_seed)
            train_ds, val_ds = train_val_split(dataset, seed=data_seed)

            # Print dataset summary on first run only
            if latent_dim == latent_dims[0] and seed_idx == 0:
                print("\nDataset summary:")
                print(dataset_summary(dataset).to_string(index=False))
                print()

            # --- Build model ---
            model = VAE(
                input_dim=config["data"]["n_features"],
                hidden_dims=config["model"]["hidden_dims"],
                latent_dim=latent_dim,
                activation=config["model"]["activation"],
                beta=config["training"]["beta"],
            )

            # --- Train ---
            print(f"  Seed {seed_idx+1}/{n_seeds} | Training...", end="", flush=True)
            history = train_vae(
                model=model,
                X_train=train_ds.X,
                X_val=val_ds.X,
                config=config,
                device=device,
                verbose=False,
            )
            final_val_loss = history["val_loss"][-1]
            n_epochs = len(history["val_loss"])
            print(f" done ({n_epochs} epochs, val_loss={final_val_loss:.4f})")

            dim_histories.append(history)

            # --- Run probes ---
            probe_results = run_all_probes(
                model=model,
                dataset=val_ds,
                device=device,
                seed=model_seed,
            )

            # Flatten and store
            row = flatten_probe_results(latent_dim, seed_idx, probe_results)
            row["val_loss"] = final_val_loss
            row["val_recon"] = history["val_recon"][-1]
            row["val_kl"] = history["val_kl"][-1]
            row["n_epochs"] = n_epochs
            all_rows.append(row)

        all_histories[latent_dim] = dim_histories

    # --- Save raw results ---
    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(results_dir, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    # --- Save summary (mean ± std across seeds) ---
    summary_df = summarise_results(df)
    summary_path = os.path.join(results_dir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary results saved to {summary_path}")

    # --- Save training histories as JSON ---
    # Convert to serializable format
    histories_serializable = {}
    for ld, histories in all_histories.items():
        histories_serializable[str(ld)] = [
            {k: [float(v) for v in vals] for k, vals in h.items()}
            for h in histories
        ]
    history_path = os.path.join(results_dir, "training_histories.json")
    save_json(histories_serializable, history_path)
    print(f"Training histories saved to {history_path}")

    # --- Print summary table ---
    print("\n" + "="*60)
    print("SUMMARY: Key probes by latent dim (mean across seeds)")
    print("="*60)

    display_cols = [
        "latent_dim",
        "mse_overall_mean",
        "normalized_frobenius_mean",
        "silhouette_rare_mean",
        "n_active_dims_mean",
        "val_loss_mean",
    ]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))

    print("\nExperiment complete. Run analysis/plot_results.py to generate figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to experiment config YAML"
    )
    args = parser.parse_args()
    main(args.config)
