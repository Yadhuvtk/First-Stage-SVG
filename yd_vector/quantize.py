from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

def quantize_colors(image: np.ndarray, n_colors: int = 8) -> Tuple[np.ndarray, List[Tuple[int,int,int]]]:
    """
    Reduce image to n_colors dominant colors using K-means.

    Args:
        image: BGR numpy array (H x W x 3)
        n_colors: number of dominant colors to extract (default 8)

    Returns:
        quantized: H x W x 3 image where every pixel is replaced
                   by its nearest cluster center color (BGR)
        palette:   list of n_colors BGR tuples, ordered by frequency
                   (most frequent color first)
    """
    h, w = image.shape[:2]
    data = image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, flags)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    quantized = res.reshape((h, w, 3))

    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    palette = [tuple(int(c) for c in centers[i]) for i in sorted_idx]

    return quantized, palette

def auto_color_count(image: np.ndarray) -> int:
    """
    Suggest a good number of colors based on image complexity.
    - Small or simple images: return 4
    - Medium images: return 8
    - Large complex images: return 16
    Use image size and unique color count as heuristics.
    """
    h, w = image.shape[:2]
    pixels = h * w
    if pixels < 100000:
        return 4
    elif pixels < 500000:
        return 8
    else:
        return 16
