"""ML-powered pre-processing pipeline for the YD-Vector tracer.

Steps
-----
1. upscale(image_np, scale=4)
   Real-ESRGAN x4 (RealESRGAN_x4plus, CUDA:0, fp16, cached singleton).

2. segment_with_sam(image_np, n_colors=8)
   SAM 2.1 hiera_large automatic mask generation -> dominant LAB colour per
   mask -> greedy colour-merge -> sorted (color_hex, binary_mask) pairs.

3. preprocess_pipeline(image_path, do_upscale=True, n_colors=8)
   Chains upscale -> segment_with_sam.  Returns list ready for
   PurePythonTracer.trace_layers().

Both model weights are downloaded automatically on first use to models/.
"""
from __future__ import annotations

import os
import urllib.request
import warnings
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR           = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR         = os.path.join(os.path.dirname(_THIS_DIR), "models")
_REALESRGAN_WEIGHTS = os.path.join(_MODELS_DIR, "RealESRGAN_x4plus.pth")
_SAM2_CHECKPOINT    = os.path.join(_MODELS_DIR, "sam2.1_hiera_large.pt")

_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
_SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)

# Module-level singletons -- loaded once and reused across calls
_esrgan_model = None
_sam2_gen     = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _download(url: str, dest: str) -> None:
    _ensure_dir(dest)
    size_mb = ""
    print(f"[ml_preprocess] Downloading {os.path.basename(dest)} from {url} ...")
    urllib.request.urlretrieve(url, dest)
    size_mb = f" ({os.path.getsize(dest) / 1e6:.1f} MB)"
    print(f"[ml_preprocess] Saved -> {dest}{size_mb}")


def _lab_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in OpenCV uint8-scale LAB space."""
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def _dominant_lab(image_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Mean LAB colour (float64, uint8-scale) of the pixels where mask is True."""
    lab    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)   # uint8 HxWx3
    pixels = lab[mask_bool]                                # (N, 3) uint8
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float64)
    return pixels.astype(np.float64).mean(axis=0)


def _lab_to_hex(mean_lab: np.ndarray) -> str:
    """Convert mean LAB (uint8-scale float64) -> RGB hex string."""
    u8   = np.clip(np.round(mean_lab), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bgr  = cv2.cvtColor(u8, cv2.COLOR_LAB2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


# ---------------------------------------------------------------------------
# 1. Real-ESRGAN x4 upscaling
# ---------------------------------------------------------------------------

def _upscale_cv2(image_np: np.ndarray, scale: int) -> np.ndarray:
    """Lanczos fallback upscaler -- used when Real-ESRGAN is not installed."""
    h, w = image_np.shape[:2]
    return cv2.resize(image_np, (w * scale, h * scale),
                      interpolation=cv2.INTER_LANCZOS4)


def _load_esrgan() -> object:
    """Load (or return cached) RealESRGANer on CUDA:0, fp16.

    Returns None if realesrgan / basicsr / torch are not installed.
    Callers should check for None and fall back to _upscale_cv2().

    Install order (Python 3.13 + CUDA 12.x):
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        pip install git+https://github.com/XPixelGroup/BasicSR.git
        pip install realesrgan
    """
    global _esrgan_model
    if _esrgan_model is not None:
        return _esrgan_model

    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        return None

    if not os.path.isfile(_REALESRGAN_WEIGHTS):
        _download(_REALESRGAN_URL, _REALESRGAN_WEIGHTS)

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4,
    )

    if torch.cuda.is_available():
        device   = "cuda:0"
        use_half = True
    else:
        device   = "cpu"
        use_half = False
        warnings.warn(
            "[ml_preprocess] No CUDA GPU -- Real-ESRGAN running on CPU (slow).",
            RuntimeWarning, stacklevel=4,
        )

    _esrgan_model = RealESRGANer(
        scale=4,
        model_path=_REALESRGAN_WEIGHTS,
        model=model,
        device=device,
        half=use_half,
    )
    print(f"[ml_preprocess] Real-ESRGAN loaded on {device}"
          + (" (fp16)" if use_half else ""))
    return _esrgan_model


def upscale(image_np: np.ndarray, scale: int = 4) -> np.ndarray:
    """Upscale *image_np* (HxWx3 BGR uint8) by *scale*.

    Uses Real-ESRGAN x4plus on CUDA:0 (fp16) when available.
    Falls back to OpenCV Lanczos4 with a warning if the ML
    dependencies (torch / basicsr / realesrgan) are not installed.

    Args:
        image_np: BGR uint8 numpy array (H, W, 3).
        scale:    Output scale factor (default 4).

    Returns:
        BGR uint8 numpy array (H*scale, W*scale, 3).
    """
    upsampler = _load_esrgan()
    if upsampler is None:
        warnings.warn(
            "[ml_preprocess] Real-ESRGAN not available -- using OpenCV Lanczos4 "
            "upscaling instead.\n"
            "To enable Real-ESRGAN install:\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install git+https://github.com/XPixelGroup/BasicSR.git\n"
            "  pip install realesrgan",
            RuntimeWarning, stacklevel=2,
        )
        return _upscale_cv2(image_np, scale)
    out, _ = upsampler.enhance(image_np, outscale=scale)
    return out   # BGR uint8


# ---------------------------------------------------------------------------
# 2. SAM 2.1 segmentation
# ---------------------------------------------------------------------------

def _segment_kmeans(
    image_np: np.ndarray,
    n_colors: int,
    color_merge_threshold: float,
) -> List[Tuple[str, np.ndarray]]:
    """k-means colour segmentation fallback (no SAM / torch required)."""
    h, w = image_np.shape[:2]
    pixels  = image_np.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)   # (n_colors, 3) BGR
    labels  = labels.flatten()

    # Compute lab centres for merging
    lab_centers = []
    for c in centers:
        lab = cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)
        lab_centers.append(lab)

    # Build (lab, mask, area) entries
    entries = []
    for k in range(n_colors):
        mask = (labels == k).reshape(h, w).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        entries.append([lab_centers[k], mask, area, centers[k]])

    entries.sort(key=lambda x: x[2], reverse=True)

    # Greedy merge
    used   = [False] * len(entries)
    merged = []
    for i in range(len(entries)):
        if used[i]:
            continue
        lab_i, mask_i, area_i, bgr_i = entries[i]
        comb_mask = mask_i.copy()
        comb_lab  = lab_i.copy()
        comb_bgr  = bgr_i.astype(np.float64)
        count     = 1
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            if _lab_dist(comb_lab, entries[j][0]) < color_merge_threshold:
                np.maximum(comb_mask, entries[j][1], out=comb_mask)
                comb_lab = (comb_lab * count + entries[j][0]) / (count + 1)
                comb_bgr = (comb_bgr * count + entries[j][3].astype(np.float64)) / (count + 1)
                count    += 1
                used[j]   = True
        b, g, r = int(comb_bgr[0]), int(comb_bgr[1]), int(comb_bgr[2])
        hex_col = "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
        )
        merged.append((hex_col, (comb_mask * 255).astype(np.uint8)))
        used[i] = True

    return merged[:n_colors]


def _load_sam2() -> object:
    """Load (or return cached) SAM2AutomaticMaskGenerator on CUDA:0.

    Returns None if sam2 / torch are not installed.
    Callers should check for None and fall back to _segment_kmeans().

    Install:
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        pip install git+https://github.com/facebookresearch/sam2.git
    """
    global _sam2_gen
    if _sam2_gen is not None:
        return _sam2_gen

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        return None

    if not os.path.isfile(_SAM2_CHECKPOINT):
        _download(_SAM2_URL, _SAM2_CHECKPOINT)

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sam2 = build_sam2(
        "configs/sam2.1/sam2.1_hiera_large.yaml",
        _SAM2_CHECKPOINT,
        device=device,
    )
    _sam2_gen = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )
    print(f"[ml_preprocess] SAM 2.1 hiera_large loaded on {device}")
    return _sam2_gen


def segment_with_sam(
    image_np: np.ndarray,
    n_colors: int = 8,
    color_merge_threshold: float = 12.0,
) -> List[Tuple[str, np.ndarray]]:
    """Segment *image_np* with SAM 2.1 and group masks by colour.

    Falls back to k-means colour segmentation when SAM 2 / torch are
    not installed, printing a warning with install instructions.

    Args:
        image_np:              BGR uint8 (H, W, 3).
        n_colors:              Maximum number of colour groups to return.
        color_merge_threshold: LAB distance below which two masks are merged.

    Returns:
        List of ``(color_hex, binary_mask)`` sorted by area descending
        (largest = bottom SVG layer).  *binary_mask* is uint8 HxW (255 = fg).
    """
    generator = _load_sam2()

    if generator is None:
        warnings.warn(
            "[ml_preprocess] SAM 2.1 not available -- using k-means colour "
            "segmentation instead.\n"
            "To enable SAM 2.1 install:\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git",
            RuntimeWarning, stacklevel=2,
        )
        return _segment_kmeans(image_np, n_colors, color_merge_threshold)

    # SAM 2 expects RGB
    image_rgb  = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    masks_data = generator.generate(image_rgb)   # list of dicts

    if not masks_data:
        return []

    # Sort by area descending so greedy merge favours large regions
    masks_data.sort(key=lambda m: m["area"], reverse=True)

    # Compute dominant LAB colour for every raw mask
    entries = []   # [lab_float64, combined_mask_u8, area]
    for m in masks_data:
        seg  = m["segmentation"]                        # bool HxW
        lab  = _dominant_lab(image_np, seg)             # float64 [0,255]
        area = int(np.count_nonzero(seg))
        entries.append([lab, seg.astype(np.uint8), area])

    # Greedy colour-merge: fold entry j into i when LAB distance < threshold
    used   = [False] * len(entries)
    merged = []
    for i in range(len(entries)):
        if used[i]:
            continue
        lab_i, mask_i, area_i = entries[i]
        comb_mask = mask_i.copy().astype(np.uint8)
        comb_area = area_i
        comb_lab  = lab_i.copy()
        count     = 1
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            if _lab_dist(comb_lab, entries[j][0]) < color_merge_threshold:
                # Union the masks
                np.maximum(comb_mask, entries[j][1], out=comb_mask)
                comb_area += entries[j][2]
                # Running mean of LAB
                comb_lab  = (comb_lab * count + entries[j][0]) / (count + 1)
                count     += 1
                used[j]    = True
        merged.append([comb_lab, comb_mask, comb_area])
        used[i] = True

    # Keep at most n_colors largest groups
    merged.sort(key=lambda x: x[2], reverse=True)
    merged = merged[:n_colors]

    # Build output list
    result: List[Tuple[str, np.ndarray]] = []
    for lab, mask, _ in merged:
        hex_col     = _lab_to_hex(lab)
        binary_mask = (mask * 255).astype(np.uint8)
        result.append((hex_col, binary_mask))

    return result


# ---------------------------------------------------------------------------
# 3. Compound pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    image_path: str,
    do_upscale: bool = True,
    n_colors: int = 8,
    color_merge_threshold: float = 12.0,
) -> List[Tuple[str, np.ndarray]]:
    """Full ML pre-processing: load -> (optional upscale) -> SAM segment.

    Args:
        image_path:            Path to the input raster image.
        do_upscale:            Run Real-ESRGAN x4 before segmentation.
        n_colors:              Max colour groups returned by SAM.
        color_merge_threshold: LAB merge threshold (default 12).

    Returns:
        List of ``(color_hex, binary_mask_uint8)`` sorted largest area first,
        ready for ``PurePythonTracer.trace_layers()``.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"[ml_preprocess] Cannot read image: {image_path}")

    if do_upscale:
        h0, w0 = image_bgr.shape[:2]
        print(f"[ml_preprocess] Upscaling {os.path.basename(image_path)} "
              f"{w0}x{h0} -> x4 ...")
        image_bgr = upscale(image_bgr, scale=4)
        h1, w1 = image_bgr.shape[:2]
        print(f"[ml_preprocess] Upscaled  {w1}x{h1} px")

    print(f"[ml_preprocess] Segmenting with SAM 2.1 "
          f"(n_colors={n_colors}, merge_thr={color_merge_threshold}) ...")
    layers = segment_with_sam(
        image_bgr,
        n_colors=n_colors,
        color_merge_threshold=color_merge_threshold,
    )
    print(f"[ml_preprocess] {len(layers)} colour layers extracted")
    return layers
