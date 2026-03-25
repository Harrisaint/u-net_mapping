"""
Custom PyTorch Dataset for paired ultrasound images and segmentation masks.

Naming convention (inside data/{normal,benign,malignant}):
    Image : <class> (<number>).png        e.g. benign (23).png
    Mask  : <class> (<number>)_mask.png    e.g. benign (23)_mask.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ("normal", "benign", "malignant")
IMG_SIZE = 256

_IMAGE_RE = re.compile(
    r"^(?P<cls>normal|benign|malignant)\s*\((?P<num>\d+)\)\.png$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Albumentations transform factories
# ---------------------------------------------------------------------------

def get_train_transform(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5,
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def discover_pairs(root: str | Path) -> List[Dict]:
    """Walk *root*/normal, *root*/benign, *root*/malignant and return a list of
    dicts with keys ``image_path``, ``mask_path``, ``class_name``, ``index``."""

    root = Path(root)
    pairs: List[Dict] = []

    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue

        for fname in sorted(cls_dir.iterdir()):
            m = _IMAGE_RE.match(fname.name)
            if m is None:
                continue

            num = m.group("num")
            mask_name = f"{cls} ({num})_mask.png"
            mask_path = cls_dir / mask_name

            if not mask_path.exists():
                continue

            pairs.append(
                {
                    "image_path": str(fname),
                    "mask_path": str(mask_path),
                    "class_name": cls,
                    "index": int(num),
                }
            )

    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UltrasoundDataset(Dataset):
    """PyTorch Dataset that yields ``(image_tensor, mask_tensor, metadata)``.

    Parameters
    ----------
    root : str or Path, optional
        Dataset root directory.  Ignored when *pairs* is supplied.
    pairs : list[dict], optional
        Pre-computed list from :func:`discover_pairs`.  Useful when the
        train/val split has already been performed on the pairs list so
        each split can receive its own *transform*.
    transform : albumentations.Compose, optional
        Albumentations pipeline applied to **both** the image and mask
        simultaneously.  If ``None``, images are resized to *img_size*
        and normalised to [0, 1] with no augmentation.
    img_size : int
        Target spatial resolution (only used when *transform* is ``None``).
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        pairs: Optional[List[Dict]] = None,
        transform: Optional[Callable] = None,
        img_size: int = IMG_SIZE,
    ) -> None:
        if pairs is not None:
            self.pairs = pairs
        elif root is not None:
            self.pairs = discover_pairs(root)
        else:
            raise ValueError("Provide either 'root' or 'pairs'.")

        self.transform = transform
        self.img_size = img_size

        if not self.pairs:
            raise FileNotFoundError(
                "No image/mask pairs found. "
                "Expected sub-folders: normal/, benign/, malignant/"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        rec = self.pairs[idx]

        img = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {rec['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {rec['mask_path']}")
        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"].float()            # (3, H, W)
            mask_tensor = augmented["mask"].unsqueeze(0).float()  # (1, H, W)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        meta = {
            "class_name": rec["class_name"],
            "index": rec["index"],
            "image_path": rec["image_path"],
            "mask_path": rec["mask_path"],
        }

        return img_tensor, mask_tensor, meta


# ---------------------------------------------------------------------------
# Single-image helpers (inference / Streamlit — no augmentation needed)
# ---------------------------------------------------------------------------

def load_single_image(path: str | Path, img_size: int = IMG_SIZE) -> torch.Tensor:
    """Load and preprocess a single image into a ``(1, 3, H, W)`` tensor."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def load_single_image_from_bytes(
    buf: bytes, img_size: int = IMG_SIZE
) -> Tuple[torch.Tensor, np.ndarray]:
    """Decode an in-memory image buffer (e.g. from ``st.file_uploader``).

    Returns ``(tensor, rgb_array)`` where tensor is ``(1, 3, H, W)``
    and rgb_array is ``(H, W, 3)`` uint8.
    """
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file could not be decoded as an image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), img
