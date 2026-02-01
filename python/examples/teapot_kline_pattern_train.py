# -*- coding: utf-8 -*-
"""
Teapot K-line Pattern Training (多尺度 CNN 二分类).

Trains MultiScaleCNN on positive (.pt from seed builder) vs negative (.pt) samples.
Supports data augmentation (noise, shift) for few-shot.

Usage:
    python teapot_kline_pattern_train.py --pos-dir outputs/teapot/processed_samples --neg-dir outputs/teapot/neg_samples
    python teapot_kline_pattern_train.py --pos-dir data/pos --neg-dir data/neg --epochs 80 --noise-std 0.01 --output models/kline_v_pattern.pth

Arguments:
    --pos-dir       Directory of positive .pt files (5, 60)
    --neg-dir       Directory of negative .pt files (5, 60)
    --output        Path to save best model (default: outputs/teapot/models/kline_v_pattern.pth)
    --epochs        Epochs (default: 50)
    --batch-size    Batch size (default: 16)
    --lr            Learning rate (default: 0.001)
    --val-ratio     Validation ratio (default: 0.2)
    --noise-std     Gaussian noise std for augmentation (default: 0.01)
    --shift-max     Max time shift for augmentation (default: 2)
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from nq.ai.teapot_pattern.dataset import KLineDataset
from nq.ai.teapot_pattern.model import MultiScaleCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(
    pos_dir: str,
    neg_dir: str,
    output_path: str = "outputs/teapot/models/kline_v_pattern.pth",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    val_ratio: float = 0.2,
    noise_std: float = 0.01,
    shift_max: int = 2,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    dataset = KLineDataset(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        noise_std=noise_std,
        shift_max=shift_max,
    )
    if len(dataset) == 0:
        logger.error("No samples: pos_dir=%s neg_dir=%s", pos_dir, neg_dir)
        return

    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiScaleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = (model(x) > 0.5).float()
                correct += (pred == y).sum().item()

        val_acc = correct / n_val
        avg_loss = total_loss / len(train_loader)
        logger.info(
            "Epoch %d/%d | Loss: %.4f | Val Acc: %.2f%%",
            epoch + 1,
            epochs,
            avg_loss,
            val_acc * 100,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info("Saved best model to %s (Val Acc: %.2f%%)", output_path, val_acc * 100)

    logger.info("Training done. Best Val Acc: %.2f%%", best_val_acc * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Teapot K-line pattern (MultiScaleCNN)")
    parser.add_argument("--pos-dir", type=str, required=True, help="Directory of positive .pt files")
    parser.add_argument("--neg-dir", type=str, required=True, help="Directory of negative .pt files")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/models/kline_v_pattern.pth",
        help="Path to save best model",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std for augmentation")
    parser.add_argument("--shift-max", type=int, default=2, help="Max time shift for augmentation")

    args = parser.parse_args()
    train_model(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        noise_std=args.noise_std,
        shift_max=args.shift_max,
    )


if __name__ == "__main__":
    main()
