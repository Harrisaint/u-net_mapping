"""
Heatmap generation: convert a U-Net probability map into a colour overlay
on the original ultrasound image.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def probability_map_to_heatmap(
    prob_map: torch.Tensor,
    image_tensor: torch.Tensor,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a JET heatmap of the model's probability output onto the
    original image.

    Parameters
    ----------
    prob_map : torch.Tensor
        Shape ``(1, 1, H, W)`` with values in [0, 1].
    image_tensor : torch.Tensor
        Shape ``(1, 3, H, W)`` with values in [0, 1].
    alpha : float
        Opacity of the heatmap layer (0 = fully transparent, 1 = fully opaque).
    colormap : int
        OpenCV colour-map constant.

    Returns
    -------
    overlay : np.ndarray
        ``(H, W, 3)`` uint8 RGB image with the heatmap blended on top.
    """
    prob_np = prob_map.squeeze().cpu().numpy()
    heat_uint8 = (prob_np * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    img_np = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)

    overlay = cv2.addWeighted(heat_color, alpha, img_uint8, 1 - alpha, 0)
    return overlay
