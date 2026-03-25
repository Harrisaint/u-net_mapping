"""
Metadata extraction from a binary segmentation mask.

Computes area ratio, centroid, and bounding-box dimensions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


def extract_metadata(mask: torch.Tensor | np.ndarray) -> Dict:
    """Derive quantitative features from a binary mask.

    Parameters
    ----------
    mask : torch.Tensor or np.ndarray
        Binary mask (values in {0, 1}).  Accepted shapes:
        ``(1, 1, H, W)``, ``(1, H, W)``, ``(H, W)``.

    Returns
    -------
    dict with keys:
        area_ratio       – float, lesion pixels / total pixels
        centroid          – (x, y) tuple or None if no lesion
        bounding_box      – dict {x, y, width, height} or None
        lesion_pixel_count – int
        total_pixel_count  – int
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    mask = mask.squeeze()

    mask_bin = (mask > 0).astype(np.uint8)
    total_pixels = mask_bin.size
    lesion_pixels = int(mask_bin.sum())
    area_ratio = lesion_pixels / total_pixels if total_pixels else 0.0

    centroid: Optional[Tuple[int, int]] = None
    bbox: Optional[Dict] = None

    if lesion_pixels > 0:
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

        x, y, w, h = cv2.boundingRect(largest)
        bbox = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}

    return {
        "area_ratio": round(area_ratio, 6),
        "centroid": centroid,
        "bounding_box": bbox,
        "lesion_pixel_count": lesion_pixels,
        "total_pixel_count": total_pixels,
    }


def format_metadata_for_prompt(meta: Dict) -> str:
    """Convert the metadata dict into a concise, LLM-friendly string."""
    lines = [
        f"Lesion area ratio: {meta['area_ratio']:.4%} of the image",
    ]
    if meta["centroid"]:
        cx, cy = meta["centroid"]
        lines.append(f"Lesion centroid: ({cx}, {cy})")
    if meta["bounding_box"]:
        bb = meta["bounding_box"]
        lines.append(
            f"Bounding box: top-left ({bb['x']}, {bb['y']}), "
            f"width {bb['width']}px, height {bb['height']}px"
        )
    if meta["centroid"] is None:
        lines.append("No lesion detected in the segmentation mask.")
    return "\n".join(lines)
