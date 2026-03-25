"""
Train the U-Net segmentation model on the ultrasound dataset.

Usage:
    python train.py                        # defaults: data/ folder, 25 epochs
    python train.py --data data/ --epochs 50 --batch-size 4 --lr 1e-4
    python train.py --val-split 0.2        # hold out 20 % for validation
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset import (
    UltrasoundDataset,
    discover_pairs,
    get_train_transform,
    get_val_transform,
)
from model import build_unet, train_model, DEVICE


def collate_fn(batch):
    """Custom collate that separates the metadata dicts from tensors."""
    images, masks, metas = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(metas)


def stratified_split(pairs, val_fraction: float, seed: int = 42):
    """Split pairs into train/val while preserving class proportions."""
    by_class = defaultdict(list)
    for p in pairs:
        by_class[p["class_name"]].append(p)

    train_pairs, val_pairs = [], []
    rng = random.Random(seed)

    for cls, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val_pairs.extend(items[:n_val])
        train_pairs.extend(items[n_val:])

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net on ultrasound data")
    parser.add_argument("--data", type=str, default="data", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data reserved for validation (0 = no val)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--save-path", type=str, default="unet_resnet34.pth")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    all_pairs = discover_pairs(args.data)
    print(f"Found {len(all_pairs)} image/mask pairs.")

    if args.val_split > 0:
        train_pairs, val_pairs = stratified_split(all_pairs, args.val_split)
        print(f"Split: {len(train_pairs)} train / {len(val_pairs)} val  (stratified by class)")
    else:
        train_pairs = all_pairs
        val_pairs = None

    train_ds = UltrasoundDataset(pairs=train_pairs, transform=get_train_transform())
    val_ds = (
        UltrasoundDataset(pairs=val_pairs, transform=get_val_transform())
        if val_pairs else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=(DEVICE.type == "cuda"),
        )
        if val_ds is not None
        else None
    )

    model = build_unet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_path=args.save_path,
    )

    print("\nTraining complete.")
    print(f"  Final train dice: {history['train_dice'][-1]:.4f}")
    if history["val_dice"]:
        print(f"  Best val dice:    {max(history['val_dice']):.4f}")
    print(f"  Weights saved to: {args.save_path}")


if __name__ == "__main__":
    main()
