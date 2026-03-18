from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

def separate_layers(
    quantized: np.ndarray,
    palette: List[Tuple[int,int,int]],
    min_area: int = 100
) -> List[Tuple[Tuple[int,int,int], np.ndarray]]:
    """
    For each color in palette, create a binary mask where that color is white
    and everything else is black.

    Args:
        quantized: H x W x 3 quantized BGR image from quantize.py
        palette:   list of BGR color tuples (most frequent first)
        min_area:  skip layers where total white pixel area < min_area

    Returns:
        list of (color_bgr, binary_mask) tuples
        - color_bgr: the BGR tuple for this layer
        - binary_mask: uint8 numpy array (H x W), 255 = this color, 0 = other
        - ordered same as palette (most frequent first = bottom SVG layer)
    """
    results = []
    for color in palette:
        mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        if np.count_nonzero(mask) >= min_area:
            results.append((color, mask))
    return results

def bgr_to_hex(bgr: Tuple[int,int,int]) -> str:
    """Convert BGR tuple to #rrggbb hex string for SVG fill."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"
