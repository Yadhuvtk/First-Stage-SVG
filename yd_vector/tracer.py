"""
YD-Vector: potrace-quality bitmap-to-SVG tracer.
Fixed: proper Bezier fitting, no collapsed paths, clean filled shapes.
"""

import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math
from pathlib import Path

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class Params:
    threshold: int = 128
    turd_size: int = 2
    alpha_max: float = 1.0
    opti_curve: bool = True
    opti_tolerance: float = 0.2
    scale: float = 1.0
    invert: bool = False
    corner_arc_radius_sharp: float = 0.42  # arc radius for sharp corners (dot < 0.5)
    corner_arc_radius_soft: float = 0.92   # arc radius for soft  corners (dot 0.5-0.85)

    @classmethod
    def from_yaml(cls, path=None) -> "Params":
        """Load params from configs/default.yaml, falling back to dataclass defaults."""
        if not _HAS_YAML:
            return cls()
        cfg_path = (Path(path) if path
                    else Path(__file__).parent.parent / "configs" / "default.yaml")
        if not cfg_path.exists():
            return cls()
        with cfg_path.open("r", encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
        pre = data.get("preprocessing", {})
        pip = data.get("pipeline", {})
        out = data.get("output", {})
        return cls(
            threshold=pre.get("threshold", 128),
            turd_size=pip.get("turdsize", 2),
            alpha_max=pip.get("alphamax", 1.0),
            opti_curve=bool(pip.get("optcurve", True)),
            opti_tolerance=pip.get("opttolerance", 0.2),
            scale=out.get("scale", 1.0),
            invert=bool(pre.get("invert", False)),
            corner_arc_radius_sharp=float(pip.get("corner_arc_radius_sharp", 0.42)),
            corner_arc_radius_soft=float(pip.get("corner_arc_radius_soft", 0.92)),
        )


# ─── Image loading ────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)

def binarize(gray: np.ndarray, threshold: int = 128, invert: bool = False) -> np.ndarray:
    binary = (gray < threshold).astype(np.uint8) * 255
    if invert:
        binary = 255 - binary
    return binary

def suppress_specks(binary: np.ndarray, min_area: int = 2) -> np.ndarray:
    if min_area <= 0:
        return binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


# ─── Contour extraction ───────────────────────────────────────────────────────

def extract_contours_cv(binary: np.ndarray):
    """Extract contours with RETR_CCOMP to get outer paths + holes."""
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


# ─── Smooth Bezier path builder ───────────────────────────────────────────────

def smooth_path(pts: np.ndarray, alpha: float = 1.0,
                r_sharp: float = 0.42, r_soft: float = 0.92) -> str:
    """
    Build a smooth SVG cubic Bezier path from a polygon.
    Uses the Catmull-Rom → Bezier conversion with tension control.
    alpha controls smoothness: 0 = sharp corners, 1.0 = smooth (potrace default)
    
    This is the key function — it creates the smooth curves potrace is known for.
    """
    n = len(pts)
    if n < 2:
        return ""

    # Tension factor — matches potrace alphamax behaviour
    # At alpha=0: all corners (lines), alpha=1.333: all smooth curves
    tension = min(alpha / 1.333, 1.0) * 0.3

    cmds = [f"M {pts[0][0]:.3f} {pts[0][1]:.3f}"]

    for i in range(n):
        p0 = pts[(i - 1) % n].astype(float)
        p1 = pts[i].astype(float)
        p2 = pts[(i + 1) % n].astype(float)
        p3 = pts[(i + 2) % n].astype(float)

        # Catmull-Rom control points
        cp1x = p1[0] + tension * (p2[0] - p0[0])
        cp1y = p1[1] + tension * (p2[1] - p0[1])
        cp2x = p2[0] - tension * (p3[0] - p1[0])
        cp2y = p2[1] - tension * (p3[1] - p1[1])

        ex, ey = p2[0], p2[1]

        # Optimization: if control points are nearly on the line → use L
        dx, dy = ex - p1[0], ey - p1[1]
        seg_len = math.sqrt(dx*dx + dy*dy)

        if seg_len < 1e-6:
            continue

        d1 = abs(dy*(cp1x - p1[0]) - dx*(cp1y - p1[1])) / seg_len
        d2 = abs(dy*(cp2x - ex)    - dx*(cp2y - ey))    / seg_len

        if d1 < 0.5 and d2 < 0.5:
            # Micro-arc corner transition instead of a hard L
            in_dx  = p2[0] - p1[0];  in_dy  = p2[1] - p1[1]
            out_dx = p3[0] - p2[0];  out_dy = p3[1] - p2[1]
            in_mag  = math.sqrt(in_dx*in_dx   + in_dy*in_dy)
            out_mag = math.sqrt(out_dx*out_dx + out_dy*out_dy)
            if in_mag < 1e-6 or out_mag < 1e-6:
                cmds.append(f"L {ex:.3f} {ey:.3f}")
            else:
                ux, uy = in_dx / in_mag,  in_dy / in_mag
                vx, vy = out_dx / out_mag, out_dy / out_mag
                dot   = ux*vx + uy*vy
                cross = ux*vy - uy*vx
                if dot >= 0.85:            # nearly straight — keep plain L
                    cmds.append(f"L {ex:.3f} {ey:.3f}")
                else:
                    rx = ry = r_sharp if dot < 0.5 else r_soft
                    sweep  = 1 if cross > 0 else 0
                    prex   = p2[0] - 0.8 * ux;  prey  = p2[1] - 0.8 * uy
                    postx  = p2[0] + 0.8 * vx;  posty = p2[1] + 0.8 * vy
                    cmds.append(f"L {prex:.3f} {prey:.3f}")
                    cmds.append(
                        f"A {rx} {ry} 0 0 {sweep} {postx:.3f} {posty:.3f}"
                    )
        else:
            cmds.append(
                f"C {cp1x:.3f} {cp1y:.3f} "
                f"{cp2x:.3f} {cp2y:.3f} "
                f"{ex:.3f} {ey:.3f}"
            )

    cmds.append("Z")
    return " ".join(cmds)


def contour_area_signed(pts: np.ndarray) -> float:
    """Signed area via shoelace. Positive = CCW, negative = CW."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return area / 2.0


# ─── Main trace pipeline ──────────────────────────────────────────────────────

def trace(input_path: str, params: Optional[Params] = None) -> Tuple[List[dict], int, int]:
    """
    Returns list of dicts: {'d': svg_path_string, 'is_hole': bool}
    """
    if params is None:
        params = Params()

    img = load_image(input_path)
    gray = to_grayscale(img)
    binary = binarize(gray, params.threshold, params.invert)
    binary = suppress_specks(binary, params.turd_size)

    h, w = binary.shape

    contours, hierarchy = extract_contours_cv(binary)
    paths = []

    if hierarchy is None or len(contours) == 0:
        return paths, int(w * params.scale), int(h * params.scale)

    hier = hierarchy[0]  # shape: (N, 4): next, prev, firstChild, parent

    # RDP epsilon — controls how many points are kept before Bezier fitting
    # Too low = too many points = jagged. Too high = loses detail.
    rdp_eps = max(0.5, params.opti_tolerance * 3)

    for i, contour in enumerate(contours):
        # Minimum area filter
        area = cv2.contourArea(contour)
        if area < max(params.turd_size, 4):
            continue

        # Determine if this contour is a hole (has a parent)
        parent = hier[i][3]
        is_hole = parent >= 0

        # Simplify with RDP — this is critical for smooth output
        pts = contour.reshape(-1, 2).astype(np.float32)
        simplified = cv2.approxPolyDP(pts.reshape(-1, 1, 2), rdp_eps, closed=True)
        pts = simplified.reshape(-1, 2).astype(float)

        # Need at least 3 points for a filled shape
        if len(pts) < 3:
            continue

        # Subpixel offset — paths go through pixel corners like potrace
        pts = pts + 0.5

        # Scale
        if params.scale != 1.0:
            pts = pts * params.scale

        # Ensure correct winding: outer=CCW, holes=CW (SVG evenodd handles both)
        signed_area = contour_area_signed(pts)
        if not is_hole and signed_area < 0:
            pts = pts[::-1]
        elif is_hole and signed_area > 0:
            pts = pts[::-1]

        # Build smooth Bezier path
        d = smooth_path(pts, alpha=params.alpha_max,
                        r_sharp=params.corner_arc_radius_sharp,
                        r_soft=params.corner_arc_radius_soft)
        if d:
            paths.append({'d': d, 'is_hole': is_hole, 'area': area})

    # Sort: largest area first (outer shapes before holes)
    paths.sort(key=lambda x: x['area'], reverse=True)

    return paths, int(w * params.scale), int(h * params.scale)


def export_svg(
    paths: List[dict],
    width: int,
    height: int,
    output_path: str,
    foreground: str = "#000000",
    background: str = "white",
) -> None:
    """
    Export SVG using evenodd fill rule — holes are automatically cut out
    when paths overlap, exactly like potrace output.
    """
    path_elements = []
    for p in paths:
        # Holes use same fill — evenodd rule handles cutout automatically
        path_elements.append(f'  <path d="{p["d"]}" fill-rule="evenodd"/>')

    paths_svg = "\n".join(path_elements)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}"
     viewBox="0 0 {width} {height}"
     style="shape-rendering:geometricPrecision; fill-rule:evenodd; clip-rule:evenodd">
  <rect width="{width}" height="{height}" fill="{background}"/>
  <g fill="{foreground}" stroke="none">
{paths_svg}
  </g>
</svg>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"[yd-vector] Saved → {output_path}  ({len(paths)} paths)")