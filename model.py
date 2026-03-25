"""
U-Net model initialisation (segmentation_models_pytorch + ResNet-34 backbone),
training loop, and inference helpers.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> smp.Unet:
    """Return a U-Net with a pre-trained encoder."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Weighted sum of Dice loss and Binary Cross-Entropy for stable training."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, targets) + self.bw * self.bce(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection / (pred_bin.sum() + target.sum() + 1e-8)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-8)).item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_epoch(
    model: smp.Unet,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    is_train: bool,
) -> Dict[str, float]:
    """Run a single train or validation epoch. Returns avg loss, dice, iou."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for images, masks, _meta in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_dice += dice_score(logits, masks) * bs
            total_iou += iou_score(logits, masks) * bs
            n += bs

    return {
        "loss": total_loss / max(n, 1),
        "dice": total_dice / max(n, 1),
        "iou": total_iou / max(n, 1),
    }


def train_model(
    model: smp.Unet,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 25,
    lr: float = 1e-4,
    patience: int = 7,
    save_path: str = "unet_resnet34.pth",
) -> Dict[str, List[float]]:
    """Full training loop with validation, early stopping, and checkpointing.

    Returns a history dict with keys: train_loss, train_dice, val_loss, val_dice.
    """
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, is_train=True)
        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])

        log = (
            f"Epoch {epoch:>3d}/{epochs} | "
            f"Train  loss={train_metrics['loss']:.4f}  dice={train_metrics['dice']:.4f}  iou={train_metrics['iou']:.4f}"
        )

        if val_loader is not None:
            val_metrics = _run_epoch(model, val_loader, criterion, None, is_train=False)
            history["val_loss"].append(val_metrics["loss"])
            history["val_dice"].append(val_metrics["dice"])
            history["val_iou"].append(val_metrics["iou"])
            scheduler.step(val_metrics["loss"])
            log += (
                f" | Val  loss={val_metrics['loss']:.4f}  "
                f"dice={val_metrics['dice']:.4f}  iou={val_metrics['iou']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), save_path)
                log += "  [saved]"
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                    break
        else:
            torch.save(model.state_dict(), save_path)

        print(log)

    if val_loader is not None:
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        print(f"Restored best weights from {save_path} (val_loss={best_val_loss:.4f}).")

    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: smp.Unet, image_tensor: torch.Tensor) -> torch.Tensor:
    """Run inference on a single image tensor.

    Parameters
    ----------
    model : smp.Unet
        Trained (or randomly-initialised) U-Net.
    image_tensor : torch.Tensor
        Shape ``(1, 3, H, W)`` in [0, 1].

    Returns
    -------
    prob_map : torch.Tensor
        Shape ``(1, 1, H, W)`` with values in [0, 1] (sigmoid applied).
    """
    model.eval()
    logits = model(image_tensor.to(DEVICE))
    return torch.sigmoid(logits).cpu()


def predict_binary(
    model: smp.Unet,
    image_tensor: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return a hard binary mask ``{0, 1}`` after thresholding."""
    prob = predict(model, image_tensor)
    return (prob >= threshold).float()
