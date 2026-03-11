"""
trainer.py

Training loop for a single VAE run.
Returns trained model and per-epoch loss history.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
import numpy as np

from src.vae_model import VAE


def train_vae(
    model: VAE,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Train a VAE and return loss history.

    Returns dict with keys: train_loss, val_loss, train_recon,
    train_kl, val_recon, val_kl — one value per epoch.
    """
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    patience = config["training"]["early_stopping_patience"]

    # Build DataLoaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "train_kl": [],
        "val_recon": [], "val_kl": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.to(device)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for (xb,) in train_loader:
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(xb)
            loss, recon, kl = model.elbo_loss(xb, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / n_batches)
        history["train_recon"].append(epoch_recon / n_batches)
        history["train_kl"].append(epoch_kl / n_batches)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            x_recon_val, mu_val, logvar_val, _ = model(X_val_t)
            val_loss, val_recon, val_kl = model.elbo_loss(
                X_val_t, x_recon_val, mu_val, logvar_val
            )

        history["val_loss"].append(val_loss.item())
        history["val_recon"].append(val_recon.item())
        history["val_kl"].append(val_kl.item())

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d} | "
                f"train {epoch_loss/n_batches:.4f} | "
                f"val {val_loss.item():.4f} | "
                f"recon {val_recon.item():.4f} | "
                f"kl {val_kl.item():.4f}"
            )

        # --- Early stopping ---
        if val_loss.item() < best_val_loss - 1e-5:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history
