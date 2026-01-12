"""
Training Script for Surrogate Model

Trains the multi-task neural network to predict transformer outputs.

Usage:
    python train_surrogate.py --data ml/data/raw/transformer_data.h5 --epochs 100
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ml.models.surrogate import (
    SurrogateModel,
    SurrogateDataset,
    SurrogateLoss,
    calculate_metrics,
)


def get_device() -> torch.device:
    """Get best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_epoch(model: SurrogateModel,
                dataloader: DataLoader,
                criterion: SurrogateLoss,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {'nll': 0, 'll': 0, 'ucc': 0, 'price': 0, 'valid': 0}
    n_batches = 0

    for x, targets in dataloader:
        x = x.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        predictions = model(x)
        loss, components = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        **{f'loss_{k}': v / n_batches for k, v in loss_components.items()}
    }


@torch.no_grad()
def evaluate(model: SurrogateModel,
             dataloader: DataLoader,
             criterion: SurrogateLoss,
             device: torch.device) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    loss_components = {'nll': 0, 'll': 0, 'ucc': 0, 'price': 0, 'valid': 0}
    n_batches = 0

    all_predictions = {'nll': [], 'll': [], 'ucc': [], 'price': [], 'valid': []}
    all_targets = {'nll': [], 'll': [], 'ucc': [], 'price': [], 'valid': []}

    for x, targets in dataloader:
        x = x.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        predictions = model(x)
        loss, components = criterion(predictions, targets)

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
        n_batches += 1

        for k in all_predictions:
            all_predictions[k].append(predictions[k].cpu())
            all_targets[k].append(targets[k].cpu())

    # Concatenate all predictions and targets
    all_predictions = {k: torch.cat(v) for k, v in all_predictions.items()}
    all_targets = {k: torch.cat(v) for k, v in all_targets.items()}

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)

    return {
        'loss': total_loss / n_batches,
        **{f'loss_{k}': v / n_batches for k, v in loss_components.items()},
        **metrics
    }


def train(args):
    """Main training function."""
    print(f"Training surrogate model")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    device = get_device()
    print(f"  Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = SurrogateDataset(args.data, normalize=True)
    print(f"  Total samples: {len(full_dataset)}")

    # Split into train/val/test
    n_total = len(full_dataset)
    n_test = int(n_total * 0.1)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val - n_test

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    model = SurrogateModel(
        input_dim=7,
        hidden_dims=(256, 256, 128, 64),
        dropout=args.dropout,
        use_batch_norm=True
    )
    model.set_normalization_params(full_dataset.get_normalization_params())
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # Loss and optimizer
    criterion = SurrogateLoss(
        w_nll=1.0, w_ll=1.0, w_ucc=1.0, w_price=1.0, w_valid=0.5
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        history['val'].append(val_metrics)

        scheduler.step()

        epoch_time = time.time() - start_time

        # Print progress
        if epoch % args.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics.get('valid_accuracy', 0):.3f} | "
                  f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'normalization_params': full_dataset.get_normalization_params(),
            }, output_dir / 'best_model.pt')

        # Save checkpoint periodically
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'normalization_params': full_dataset.get_normalization_params(),
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  NLL RMSE: {test_metrics.get('nll_rmse', 0):.4f}")
    print(f"  LL RMSE: {test_metrics.get('ll_rmse', 0):.4f}")
    print(f"  Ucc RMSE: {test_metrics.get('ucc_rmse', 0):.4f}")
    print(f"  Price RMSE: {test_metrics.get('price_rmse', 0):.4f}")
    print(f"  Validity Accuracy: {test_metrics.get('valid_accuracy', 0):.3f}")
    print(f"  Validity F1: {test_metrics.get('valid_f1', 0):.3f}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'normalization_params': full_dataset.get_normalization_params(),
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Models saved to {output_dir}")

    return model, history, test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train surrogate model')
    parser.add_argument('--data', type=str, default='ml/data/raw/transformer_data.h5',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='ml/checkpoints/surrogate',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--print_every', type=int, default=10,
                       help='Print progress every N epochs')
    parser.add_argument('--save_every', type=int, default=50,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Make paths absolute
    base_dir = Path(__file__).parent.parent.parent
    args.data = str(base_dir / args.data)
    args.output_dir = str(base_dir / args.output_dir)

    train(args)
