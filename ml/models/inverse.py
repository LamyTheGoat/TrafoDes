"""
Inverse Design Model (Conditional VAE)

Given target specifications (NLL, LL, Ucc, etc.), predicts optimal design parameters.
Uses a Conditional Variational Autoencoder (CVAE) to handle the one-to-many mapping.

Usage:
    from ml.models.inverse import InverseDesignModel

    model = InverseDesignModel()
    specs = torch.tensor([[target_nll, target_ll, target_ucc, power, hv, lv]])
    params = model.predict(specs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import h5py


class InverseDesignDataset(Dataset):
    """Dataset for inverse design training."""

    def __init__(self, data_path: str, normalize: bool = True, valid_only: bool = True):
        """
        Load dataset for inverse design.

        Args:
            data_path: Path to HDF5 file
            normalize: Whether to normalize data
            valid_only: Only use valid (feasible) designs
        """
        with h5py.File(data_path, 'r') as f:
            # Filter valid designs if requested
            if valid_only:
                mask = f['is_valid'][:]
            else:
                mask = np.ones(len(f['nll'][:]), dtype=bool)

            # Design parameters (targets for inverse model)
            self.params = np.stack([
                f['core_diameter'][:][mask],
                f['core_length'][:][mask],
                f['lv_turns'][:][mask].astype(np.float32),
                f['foil_height'][:][mask],
                f['foil_thickness'][:][mask],
                f['hv_thickness'][:][mask],
                f['hv_length'][:][mask],
            ], axis=1)

            # Specifications (inputs for inverse model)
            self.specs = np.stack([
                f['nll'][:][mask],
                f['ll'][:][mask],
                f['ucc'][:][mask],
                f['price'][:][mask],
            ], axis=1)

            self.metadata = dict(f.attrs)

        self.normalize = normalize
        self.n_samples = len(self.params)

        # Normalization statistics
        if normalize:
            self.params_mean = self.params.mean(axis=0)
            self.params_std = self.params.std(axis=0) + 1e-8
            self.specs_mean = self.specs.mean(axis=0)
            self.specs_std = self.specs.std(axis=0) + 1e-8
        else:
            self.params_mean = np.zeros(7)
            self.params_std = np.ones(7)
            self.specs_mean = np.zeros(4)
            self.specs_std = np.ones(4)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        params = self.params[idx]
        specs = self.specs[idx]

        if self.normalize:
            params = (params - self.params_mean) / self.params_std
            specs = (specs - self.specs_mean) / self.specs_std

        return (
            torch.tensor(specs, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32)
        )

    def get_normalization_params(self) -> Dict:
        """Return normalization parameters."""
        return {
            'params_mean': self.params_mean,
            'params_std': self.params_std,
            'specs_mean': self.specs_mean,
            'specs_std': self.specs_std,
        }


class InverseDesignModel(nn.Module):
    """
    Conditional Variational Autoencoder for inverse transformer design.

    Given target specifications (NLL, LL, Ucc), generates design parameters.

    Architecture:
        Encoder: specs -> latent distribution (mu, log_var)
        Decoder: latent + specs -> design parameters
    """

    def __init__(self,
                 spec_dim: int = 4,
                 param_dim: int = 7,
                 latent_dim: int = 16,
                 hidden_dims: Tuple[int, ...] = (128, 64),
                 param_bounds: Optional[np.ndarray] = None):
        """
        Initialize CVAE.

        Args:
            spec_dim: Dimension of specification input
            param_dim: Dimension of parameter output
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            param_bounds: (7, 2) array of [min, max] for each parameter
        """
        super().__init__()

        self.spec_dim = spec_dim
        self.param_dim = param_dim
        self.latent_dim = latent_dim

        # Default parameter bounds
        if param_bounds is None:
            from mainRect import (
                CORE_MINIMUM, CORE_MAXIMUM,
                FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM,
                FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM,
                FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM,
                HVTHICK_MINIMUM, HVTHICK_MAXIMUM,
                HV_LEN_MAXIMUM,
            )
            param_bounds = np.array([
                [CORE_MINIMUM, CORE_MAXIMUM],
                [0, CORE_MAXIMUM],
                [FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM],
                [FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM],
                [FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM],
                [HVTHICK_MINIMUM, HVTHICK_MAXIMUM],
                [HVTHICK_MINIMUM, HV_LEN_MAXIMUM],
            ])

        self.register_buffer('param_min', torch.tensor(param_bounds[:, 0], dtype=torch.float32))
        self.register_buffer('param_max', torch.tensor(param_bounds[:, 1], dtype=torch.float32))

        # Encoder: (specs + params) -> latent
        encoder_layers = []
        prev_dim = spec_dim + param_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder: (latent + specs) -> params
        decoder_layers = []
        prev_dim = latent_dim + spec_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim

        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(hidden_dims[0], param_dim)

        # Store normalization params
        self.register_buffer('params_mean', torch.zeros(param_dim))
        self.register_buffer('params_std', torch.ones(param_dim))
        self.register_buffer('specs_mean', torch.zeros(spec_dim))
        self.register_buffer('specs_std', torch.ones(spec_dim))

    def encode(self, specs: torch.Tensor, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to latent distribution.

        Args:
            specs: Specification input (batch_size, spec_dim)
            params: Parameter input (batch_size, param_dim)

        Returns:
            mu, log_var of latent distribution
        """
        x = torch.cat([specs, params], dim=-1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, specs: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to parameters.

        Args:
            z: Latent vector (batch_size, latent_dim)
            specs: Specification conditioning (batch_size, spec_dim)

        Returns:
            Predicted parameters (batch_size, param_dim)
        """
        x = torch.cat([z, specs], dim=-1)
        h = self.decoder(x)
        out = self.fc_out(h)
        # Apply sigmoid to bound outputs to [0, 1], then scale
        return torch.sigmoid(out)

    def forward(self, specs: torch.Tensor, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            specs: Specifications (normalized)
            params: Ground truth parameters (normalized)

        Returns:
            reconstructed params, mu, log_var
        """
        mu, log_var = self.encode(specs, params)
        z = self.reparameterize(mu, log_var)
        recon_params = self.decode(z, specs)
        return recon_params, mu, log_var

    def predict(self, specs: torch.Tensor, n_samples: int = 1, normalize_input: bool = True) -> torch.Tensor:
        """
        Generate design parameters from specifications.

        Args:
            specs: Target specifications (batch_size, spec_dim)
            n_samples: Number of samples to generate per specification
            normalize_input: Whether to normalize input specs

        Returns:
            Generated parameters (batch_size * n_samples, param_dim) in original scale
        """
        if normalize_input:
            specs = (specs - self.specs_mean) / self.specs_std

        batch_size = specs.shape[0]

        # Sample from prior
        z = torch.randn(batch_size * n_samples, self.latent_dim, device=specs.device)

        # Repeat specs for multiple samples
        if n_samples > 1:
            specs = specs.repeat_interleave(n_samples, dim=0)

        # Decode
        params_norm = self.decode(z, specs)

        # Denormalize parameters
        params = params_norm * self.params_std + self.params_mean

        # Apply bounds and constraints
        params = self._apply_constraints(params)

        return params

    def _apply_constraints(self, params: torch.Tensor) -> torch.Tensor:
        """Apply physical constraints to parameters."""
        # Clamp to bounds
        params = torch.clamp(params, self.param_min, self.param_max)

        # core_length <= core_diameter
        params[:, 1] = torch.clamp(params[:, 1], max=params[:, 0])

        # hv_length >= hv_thickness
        params[:, 6] = torch.clamp(params[:, 6], min=params[:, 5])

        # Round lv_turns
        params[:, 2] = torch.round(params[:, 2])

        return params

    def set_normalization_params(self, norm_params: Dict):
        """Set normalization parameters."""
        self.params_mean.copy_(torch.tensor(norm_params['params_mean'], dtype=torch.float32))
        self.params_std.copy_(torch.tensor(norm_params['params_std'], dtype=torch.float32))
        self.specs_mean.copy_(torch.tensor(norm_params['specs_mean'], dtype=torch.float32))
        self.specs_std.copy_(torch.tensor(norm_params['specs_std'], dtype=torch.float32))


class CVAELoss(nn.Module):
    """
    CVAE loss function combining reconstruction and KL divergence.
    """

    def __init__(self, beta: float = 1.0, physics_weight: float = 0.0):
        """
        Args:
            beta: Weight for KL divergence (beta-VAE style)
            physics_weight: Weight for physics consistency loss (if using surrogate)
        """
        super().__init__()
        self.beta = beta
        self.physics_weight = physics_weight

    def forward(self,
                recon_params: torch.Tensor,
                params: torch.Tensor,
                mu: torch.Tensor,
                log_var: torch.Tensor,
                physics_loss: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute CVAE loss.

        Args:
            recon_params: Reconstructed parameters
            params: Ground truth parameters
            mu: Latent mean
            log_var: Latent log variance
            physics_loss: Optional physics consistency loss

        Returns:
            total_loss, loss_components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_params, params)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        if physics_loss is not None and self.physics_weight > 0:
            total_loss = total_loss + self.physics_weight * physics_loss

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'physics_loss': physics_loss.item() if physics_loss is not None else 0,
        }


def train_inverse_model(model: InverseDesignModel,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       epochs: int = 100,
                       lr: float = 1e-3,
                       device: str = 'cpu',
                       surrogate_model = None) -> Dict:
    """
    Train inverse design model.

    Args:
        model: InverseDesignModel to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        surrogate_model: Optional surrogate for physics consistency

    Returns:
        Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = CVAELoss(beta=1.0, physics_weight=0.1 if surrogate_model else 0.0)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0

        for specs, params in train_loader:
            specs = specs.to(device)
            params = params.to(device)

            optimizer.zero_grad()
            recon_params, mu, log_var = model(specs, params)

            # Physics consistency loss (optional)
            physics_loss = None
            if surrogate_model is not None:
                with torch.no_grad():
                    pred_specs = surrogate_model(recon_params)
                physics_loss = F.mse_loss(
                    torch.stack([pred_specs['nll'], pred_specs['ll'], pred_specs['ucc']], dim=-1),
                    specs[:, :3]
                )

            loss, _ = criterion(recon_params, params, mu, log_var, physics_loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for specs, params in val_loader:
                specs = specs.to(device)
                params = params.to(device)

                recon_params, mu, log_var = model(specs, params)
                loss, _ = criterion(recon_params, params, mu, log_var)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    return history


def load_inverse_model(checkpoint_path: str, device: str = 'cpu') -> InverseDesignModel:
    """Load trained inverse design model."""
    model = InverseDesignModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'normalization_params' in checkpoint:
            model.set_normalization_params(checkpoint['normalization_params'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
