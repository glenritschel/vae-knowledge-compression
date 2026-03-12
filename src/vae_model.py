"""
vae_model.py

Standard VAE implementation in PyTorch.
Architecture is symmetric: encoder and decoder mirror each other.
Latent dimension is the only variable changed across the experiment sweep.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn]
            in_dim = h

        self.net = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(in_dim, latent_dim)
        self.logvar_layer = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        # Decoder mirrors encoder — reverse hidden dims
        reversed_dims = list(reversed(hidden_dims))
        layers = []
        in_dim = latent_dim
        for h in reversed_dims:
            layers += [nn.Linear(in_dim, h), act_fn]
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = "relu",
        beta: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, activation)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, activation)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # deterministic at eval time

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def elbo_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO = reconstruction loss + beta * KL divergence.
        Returns total loss, recon loss, and KL loss separately for tracking.
        """
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="none")
        recon_loss = torch.mean(torch.sum(recon_loss, dim=1))
        # KL divergence per dimension, summed over dims, mean over batch
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mu and logvar for a batch."""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent vectors."""
        self.eval()
        with torch.no_grad():
            return self.decoder(z)

    def get_latent_kl_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-dimension KL divergence averaged over a batch.
        Used by Probe 6 (posterior collapse tracking).
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            # KL per dim: -0.5 * (1 + logvar - mu^2 - exp(logvar))
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            return kl_per_dim.mean(dim=0)  # (latent_dim,)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n points from the prior and decode."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            return self.decoder(z)
