# VAE Knowledge Compression

**What knowledge does a Variational Autoencoder lose as its latent space shrinks?**

This repository contains the full experiment for the paper:

> *"The Knowledge Hierarchy in Compressed Representations: What Variational Autoencoders Forget First"*
> Glen Charles Ritschel, Ritschel Research, Tega Cay, SC
> Zenodo preprint, 2026

## Overview

We train a series of VAEs on synthetic structured data with known generative factors,
systematically reducing the latent dimension from D (full) down to 1. At each compression
level, we probe what types of knowledge survive using five targeted diagnostics.

**Core finding:** VAE compression loss follows a typed hierarchy:
1. Rare/minority knowledge dies first
2. Relational (covariance) knowledge dies second
3. Marginal (mean/variance) knowledge survives longest

The ELBO training metric is blind to the first two types of loss until compression is severe.

## Quickstart

```bash
pip install -r requirements.txt
python experiments/run_experiment.py
python analysis/plot_results.py
```

Results are written to `results/`. Figures are written to `figures/`.

## Repo Structure

```
src/
  data_generator.py   — Synthetic data with planted generative factors
  vae_model.py        — VAE implementation in PyTorch
  trainer.py          — Training loop with ELBO tracking
  probes.py           — Five knowledge probes
  utils.py            — Shared utilities
configs/
  experiment_config.yaml  — All hyperparameters
experiments/
  run_experiment.py   — Main experiment entry point
analysis/
  plot_results.py     — Generate all figures from saved CSVs
results/              — Generated at runtime (gitignored)
figures/              — Generated at runtime (gitignored)
```

## Citation

If you use this code, please cite the accompanying Zenodo deposit.
