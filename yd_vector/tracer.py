"""
YD-Vector: potrace-style bitmap-to-SVG tracer.
Pipeline: load → grayscale → threshold → decompose into paths → 
          fit Bezier curves → export SVG
"""

import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Params:
    threshold: int = 128          # 0-255 binarization cutoff
    turd_size: int = 2            # suppress speckles smaller than this area
    alpha_max: float = 1.0        # corner smoothness (0=sharp, 1.333=smooth)
    opti_curve: bool = True       # optimize Bezier curves
    opti_tolerance: float = 0.2   # curve optimization tolerance
    scale: float = 1.0            # output scale multiplier
    invert: bool = False          # invert bitmap before tracing


@dataclass
class Path:
    points: List[Tuple[float, float]] = field(default_factory=list)
    is_hole: bool = False         # True = inner hole (white region inside dark)


def load_image(path: str) -> np.ndarray:
    """Load any raster image and return as uint8 numpy array (H,W,C or H,W)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """RGB → grayscale using luminance weights (same as Pillow/potrace)."""
    if img.ndim == 2:
        return img
    return (0.299 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2]).astype(np.uint8)


def binarize(gray: np.ndarray, threshold: int = 128, invert: bool = False) -> np.ndarray:
    """
    Convert grayscale to binary bitmap.
    Returns uint8 array: 0 = background (white), 255 = foreground (dark).
    """
    binary = (gray < threshold).astype(np.uint8) * 255
    if invert:
        binary = 255 - binary
    return binary


def suppress_specks(binary: np.ndarray, min_area: int = 2) -> np.ndarray:
    """Remove connected components smaller than min_area pixels (like potrace turdsize)."""
    if min_area <= 0:
        return binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def extract_contours(binary: np.ndarray) -> Tuple[List, List]:
    """
    Extract contours using OpenCV RETR_CCOMP (outer + holes).
    Returns (contours, hierarchy).
    """
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


def rdp_simplify(points: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification."""
    if len(points) < 3:
        return points
    result = cv2.approxPolyDP(points, epsilon, closed=True)
    return result.reshape(-1, 2)


def fit_bezier_path(pts: np.ndarray, alpha_max: float = 1.0) -> List:
    """
    Fit cubic Bezier segments to a sequence of points.
    Returns list of SVG path commands.
    Returns a list of ('M'|'C'|'L'|'Z', ...) tuples.
    """
    if len(pts) < 2:
        return []

    commands = []
    n = len(pts)

    commands.append(('M', float(pts[0][0]), float(pts[0][1])))

    if n < 4:
        # Too few points for Bezier — use straight lines
        for i in range(1, n):
            commands.append(('L', float(pts[i][0]), float(pts[i][1])))
        commands.append(('Z',))
        return commands

    # Catmull-Rom → cubic Bezier conversion
    # alpha_max controls how "round" corners become
    alpha = min(alpha_max / 3.0, 1.0 / 3.0)

    for i in range(n):
        p0 = pts[(i - 1) % n].astype(float)
        p1 = pts[i].astype(float)
        p2 = pts[(i + 1) % n].astype(float)
        p3 = pts[(i + 2) % n].astype(float)

        # Control points
        cp1 = p1 + alpha * (p2 - p0)
        cp2 = p2 - alpha * (p3 - p1)

        commands.append((
            'C',
            cp1[0], cp1[1],
            cp2[0], cp2[1],
            float(p2[0]), float(p2[1])
        ))

    commands.append(('Z',))
    return commands


def optimize_path(commands: List, tolerance: float = 0.2) -> List:
    """
    Merge consecutive nearly-collinear Bezier segments into lines.
    Reduces SVG file size like potrace's curve optimization.
    """
    if not commands:
        return commands

    optimized = [commands[0]]
    for cmd in commands[1:]:
        if cmd[0] == 'C' and optimized[-1][0] == 'C':
            # Check if control points are nearly on the straight line
            prev = optimized[-1]
            # End point of previous, start of current curve
            ex, ey = prev[5], prev[6]
            cx1, cy1 = cmd[1], cmd[2]
            cx2, cy2 = cmd[3], cmd[4]
            ex2, ey2 = cmd[5], cmd[6]

            # Direction vector
            dx, dy = ex2 - ex, ey2 - ey
            length = (dx**2 + dy**2) ** 0.5
            if length < 1e-6:
                optimized.append(cmd)
                continue

            # Distance of control points from the line
            def dist_to_line(px, py):
                return abs(dy * px - dx * py + ex2 * ey - ey2 * ex) / length

            d1 = dist_to_line(cx1, cy1)
            d2 = dist_to_line(cx2, cy2)

            if d1 < tolerance and d2 < tolerance:
                optimized.append(('L', ex2, ey2))
            else:
                optimized.append(cmd)
        else:
            optimized.append(cmd)

    return optimized


def commands_to_d(commands: List) -> str:
    """Serialize path commands list to SVG 'd' attribute string."""
    parts = []
    for cmd in commands:
        if cmd[0] == 'M':
            parts.append(f"M {cmd[1]:.3f} {cmd[2]:.3f}")
        elif cmd[0] == 'L':
            parts.append(f"L {cmd[1]:.3f} {cmd[2]:.3f}")
        elif cmd[0] == 'C':
            parts.append(
                f"C {cmd[1]:.3f} {cmd[2]:.3f} "
                f"{cmd[3]:.3f} {cmd[4]:.3f} "
                f"{cmd[5]:.3f} {cmd[6]:.3f}"
            )
        elif cmd[0] == 'Z':
            parts.append("Z")
    return " ".join(parts)


def trace(
    input_path: str,
    params: Optional[Params] = None
) -> Tuple[List[str], int, int]:
    """
    Full trace pipeline. Returns (list_of_svg_path_d_strings, width, height).
    """
    if params is None:
        params = Params()

    # 1. Load & binarize
    img = load_image(input_path)
    gray = to_grayscale(img)
    binary = binarize(gray, params.threshold, params.invert)

    # 2. Suppress noise
    binary = suppress_specks(binary, params.turd_size)

    h, w = binary.shape

    # 3. Extract contours with hierarchy (outer paths + holes)
    contours, hierarchy = extract_contours(binary)

    paths_d = []

    if hierarchy is None:
        return paths_d, w, h

    hierarchy = hierarchy[0]  # shape: (N, 4) — next, prev, child, parent

    rdp_eps = max(0.5, params.opti_tolerance * 2)

    for i, contour in enumerate(contours):
        pts = contour.reshape(-1, 2)

        # Skip tiny contours
        area = cv2.contourArea(contour)
        if area < params.turd_size:
            continue

        # Simplify with RDP
        pts = rdp_simplify(pts.astype(np.float32), epsilon=rdp_eps)

        # Scale
        if params.scale != 1.0:
            pts = pts * params.scale

        # Fit Bezier
        commands = fit_bezier_path(pts, alpha_max=params.alpha_max)

        # Optimize curves
        if params.opti_curve:
            commands = optimize_path(commands, tolerance=params.opti_tolerance)

        d = commands_to_d(commands)
        if d:
            paths_d.append(d)

    return paths_d, int(w * params.scale), int(h * params.scale)


def export_svg(
    paths_d: List[str],
    width: int,
    height: int,
    output_path: str,
    foreground: str = "#000000",
    background: str = "white",
) -> None:
    """Write SVG file using evenodd fill rule (like potrace)."""
    path_elements = "\n  ".join(
        f'<path d="{d}" fill-rule="evenodd"/>' for d in paths_d
    )
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}"
     viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="{background}"/>
  <g fill="{foreground}" stroke="none">
  {path_elements}
  </g>
</svg>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"[yd-vector] Saved → {output_path}  ({len(paths_d)} paths)")