from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def load_image_rgb(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def threshold_image(gray: np.ndarray, threshold: int = 128, invert: bool = False) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold, 255, mode)
    return binary


def cleanup_binary(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def preprocess_image(
    image_path: str,
    threshold: int,
    invert: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = load_image_rgb(image_path)
    gray = to_grayscale(rgb)
    binary = threshold_image(gray, threshold=threshold, invert=invert)
    cleaned = cleanup_binary(binary)
    return rgb, gray, cleaned