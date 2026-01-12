"""
Surrogate Model for Transformer Design

Multi-task neural network that predicts transformer outputs (NLL, LL, Ucc, Cost)
from design parameters. Provides ~1000x speedup over physics-based calculations.

Usage:
    from ml.models.surrogate import SurrogateModel, SurrogateDataset

    model = SurrogateModel()
    predictions = model(input_tensor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import h5py


class SurrogateDataset(Dataset):
    """PyTorch Dataset for transformer design data."""

    def __init__(self, data_path: str, normalize: bool = True):
        """
        Load dataset from HDF5 file.

        Args:
            data_path: Path to HDF5 file
            normalize: Whether to normalize inputs and outputs
        """
        self.data = {}
        with h5py.File(data_path, 'r') as f:
            # Input features
            self.inputs = np.stack([
                f['core_diameter'][:],
                f['core_length'][:],
                f['lv_turns'][:].astype(np.float32),
                f['foil_height'][:],
                f['foil_thickness'][:],
                f['hv_thickness'][:],
                f['hv_length'][:],
            ], axis=1)

            # Output targets
            self.nll = f['nll'][:]
            self.ll = f['ll'][:]
            self.ucc = f['ucc'][:]
            self.price = f['price'][:]
            self.is_valid = f['is_valid'][:].astype(np.float32)

            # Metadata
            self.metadata = dict(f.attrs)

        self.normalize = normalize
        self.n_samples = len(self.nll)

        # Calculate normalization statistics
        if normalize:
            self.input_mean = self.inputs.mean(axis=0)
            self.input_std = self.inputs.std(axis=0) + 1e-8

            # Use log transform for outputs (they span wide ranges)
            self.nll_mean = np.log1p(self.nll).mean()
            self.nll_std = np.log1p(self.nll).std() + 1e-8
            self.ll_mean = np.log1p(self.ll).mean()
            self.ll_std = np.log1p(self.ll).std() + 1e-8
            self.ucc_mean = self.ucc.mean()
            self.ucc_std = self.ucc.std() + 1e-8
            self.price_mean = np.log1p(self.price).mean()
            self.price_std = np.log1p(self.price).std() + 1e-8
        else:
            self.input_mean = np.zeros(7)
            self.input_std = np.ones(7)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.inputs[idx]

        if self.normalize:
            x = (x - self.input_mean) / self.input_std

        targets = {
            'nll': np.log1p(self.nll[idx]) if self.normalize else self.nll[idx],
            'll': np.log1p(self.ll[idx]) if self.normalize else self.ll[idx],
            'ucc': self.ucc[idx],
            'price': np.log1p(self.price[idx]) if self.normalize else self.price[idx],
            'valid': self.is_valid[idx],
        }

        return torch.tensor(x, dtype=torch.float32), {
            k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()
        }

    def get_normalization_params(self) -> Dict:
        """Return normalization parameters for inference."""
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'nll_mean': getattr(self, 'nll_mean', 0),
            'nll_std': getattr(self, 'nll_std', 1),
            'll_mean': getattr(self, 'll_mean', 0),
            'll_std': getattr(self, 'll_std', 1),
            'ucc_mean': getattr(self, 'ucc_mean', 0),
            'ucc_std': getattr(self, 'ucc_std', 1),
            'price_mean': getattr(self, 'price_mean', 0),
            'price_std': getattr(self, 'price_std', 1),
        }


class SurrogateModel(nn.Module):
    """
    Multi-task neural network for transformer design prediction.

    Architecture:
        - Shared feature extraction layers
        - Separate output heads for each target (NLL, LL, Ucc, Price, Valid)

    Input: 7 design parameters
    Output: Dictionary with predictions for each target
    """

    def __init__(self,
                 input_dim: int = 7,
                 hidden_dims: Tuple[int, ...] = (256, 256, 128, 64),
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize surrogate model.

        Args:
            input_dim: Number of input features (design parameters)
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build shared layers
        layers = []
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # No dropout on last shared layer
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.shared = nn.Sequential(*layers)

        # Output heads
        last_hidden = hidden_dims[-1]

        # NLL head (No-Load Loss)
        self.nll_head = nn.Sequential(
            nn.Linear(last_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # LL head (Load Loss)
        self.ll_head = nn.Sequential(
            nn.Linear(last_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Ucc head (Impedance)
        self.ucc_head = nn.Sequential(
            nn.Linear(last_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Price head
        self.price_head = nn.Sequential(
            nn.Linear(last_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Validity head (binary classification)
        self.valid_head = nn.Sequential(
            nn.Linear(last_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Store normalization params (will be set during training/loading)
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('nll_mean', torch.tensor(0.0))
        self.register_buffer('nll_std', torch.tensor(1.0))
        self.register_buffer('ll_mean', torch.tensor(0.0))
        self.register_buffer('ll_std', torch.tensor(1.0))
        self.register_buffer('ucc_mean', torch.tensor(0.0))
        self.register_buffer('ucc_std', torch.tensor(1.0))
        self.register_buffer('price_mean', torch.tensor(0.0))
        self.register_buffer('price_std', torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary with predictions:
                - nll: No-Load Loss prediction
                - ll: Load Loss prediction
                - ucc: Impedance prediction
                - price: Price prediction
                - valid: Validity probability
        """
        features = self.shared(x)

        return {
            'nll': self.nll_head(features).squeeze(-1),
            'll': self.ll_head(features).squeeze(-1),
            'ucc': self.ucc_head(features).squeeze(-1),
            'price': self.price_head(features).squeeze(-1),
            'valid': self.valid_head(features).squeeze(-1),
        }

    def predict_denormalized(self, x: torch.Tensor, normalize_input: bool = True) -> Dict[str, torch.Tensor]:
        """
        Predict with denormalized outputs.

        Args:
            x: Input tensor (raw or normalized based on normalize_input)
            normalize_input: If True, normalize input before prediction

        Returns:
            Dictionary with denormalized predictions
        """
        if normalize_input:
            x = (x - self.input_mean) / self.input_std

        preds = self.forward(x)

        # Denormalize outputs (inverse of log1p is expm1)
        return {
            'nll': torch.expm1(preds['nll'] * self.nll_std + self.nll_mean),
            'll': torch.expm1(preds['ll'] * self.ll_std + self.ll_mean),
            'ucc': preds['ucc'] * self.ucc_std + self.ucc_mean,
            'price': torch.expm1(preds['price'] * self.price_std + self.price_mean),
            'valid': preds['valid'],
        }

    def set_normalization_params(self, params: Dict):
        """Set normalization parameters from dataset."""
        self.input_mean.copy_(torch.tensor(params['input_mean'], dtype=torch.float32))
        self.input_std.copy_(torch.tensor(params['input_std'], dtype=torch.float32))
        self.nll_mean.copy_(torch.tensor(params['nll_mean'], dtype=torch.float32))
        self.nll_std.copy_(torch.tensor(params['nll_std'], dtype=torch.float32))
        self.ll_mean.copy_(torch.tensor(params['ll_mean'], dtype=torch.float32))
        self.ll_std.copy_(torch.tensor(params['ll_std'], dtype=torch.float32))
        self.ucc_mean.copy_(torch.tensor(params['ucc_mean'], dtype=torch.float32))
        self.ucc_std.copy_(torch.tensor(params['ucc_std'], dtype=torch.float32))
        self.price_mean.copy_(torch.tensor(params['price_mean'], dtype=torch.float32))
        self.price_std.copy_(torch.tensor(params['price_std'], dtype=torch.float32))


class SurrogateLoss(nn.Module):
    """
    Multi-task loss for surrogate model training.

    Combines MSE losses for regression targets and BCE for validity.
    """

    def __init__(self,
                 w_nll: float = 1.0,
                 w_ll: float = 1.0,
                 w_ucc: float = 1.0,
                 w_price: float = 1.0,
                 w_valid: float = 0.5):
        """
        Initialize loss function.

        Args:
            w_*: Weight for each loss component
        """
        super().__init__()
        self.w_nll = w_nll
        self.w_ll = w_ll
        self.w_ucc = w_ucc
        self.w_price = w_price
        self.w_valid = w_valid

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        loss_nll = F.mse_loss(predictions['nll'], targets['nll'])
        loss_ll = F.mse_loss(predictions['ll'], targets['ll'])
        loss_ucc = F.mse_loss(predictions['ucc'], targets['ucc'])
        loss_price = F.mse_loss(predictions['price'], targets['price'])
        loss_valid = F.binary_cross_entropy(predictions['valid'], targets['valid'])

        total_loss = (
            self.w_nll * loss_nll +
            self.w_ll * loss_ll +
            self.w_ucc * loss_ucc +
            self.w_price * loss_price +
            self.w_valid * loss_valid
        )

        return total_loss, {
            'nll': loss_nll.item(),
            'll': loss_ll.item(),
            'ucc': loss_ucc.item(),
            'price': loss_price.item(),
            'valid': loss_valid.item(),
        }


def calculate_metrics(predictions: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor],
                     dataset: Optional[SurrogateDataset] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: Model predictions (normalized)
        targets: Ground truth (normalized)
        dataset: Dataset with normalization params for MAPE calculation

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # MSE for each target
    for key in ['nll', 'll', 'ucc', 'price']:
        mse = F.mse_loss(predictions[key], targets[key]).item()
        metrics[f'{key}_mse'] = mse
        metrics[f'{key}_rmse'] = np.sqrt(mse)

    # Validity accuracy
    pred_valid = (predictions['valid'] > 0.5).float()
    metrics['valid_accuracy'] = (pred_valid == targets['valid']).float().mean().item()

    # F1 score for validity
    tp = ((pred_valid == 1) & (targets['valid'] == 1)).sum().float()
    fp = ((pred_valid == 1) & (targets['valid'] == 0)).sum().float()
    fn = ((pred_valid == 0) & (targets['valid'] == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    metrics['valid_f1'] = (2 * precision * recall / (precision + recall + 1e-8)).item()

    return metrics


# Convenience function to create model and load weights
def load_surrogate_model(checkpoint_path: str, device: str = 'cpu') -> SurrogateModel:
    """Load a trained surrogate model from checkpoint."""
    model = SurrogateModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
