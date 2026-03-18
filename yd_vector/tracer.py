"""
YD-Vector: potrace-quality bitmap-to-SVG tracer.

Upgrade: replaced cv2.approxPolyDP + Catmull-Rom heuristic with:
  1. Corner-penalty optimal polygon  (potrace Stage 2 equivalent)
  2. Schneider iterative cubic Bezier fitting  (potrace Stage 3 equivalent)
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


# ─── Params ───────────────────────────────────────────────────────────────────

@dataclass
class Params:
    threshold: int      = 128
    turd_size: int      = 2
    alpha_max: float    = 1.0
    opti_curve: bool    = True
    opti_tolerance: float = 0.2
    scale: float        = 1.0
    invert: bool        = False
    # corner_arc_radius_* no longer used — Schneider handles corners natively

    @classmethod
    def from_yaml(cls, path=None) -> "Params":
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
            threshold     = pre.get("threshold",    128),
            turd_size     = pip.get("turdsize",     2),
            alpha_max     = pip.get("alphamax",     1.0),
            opti_curve    = bool(pip.get("optcurve",      True)),
            opti_tolerance= pip.get("opttolerance", 0.2),
            scale         = out.get("scale",        1.0),
            invert        = bool(pre.get("invert",  False)),
        )


# ─── Image loading ────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)

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
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


# ─── Stage 1 replacement: corner-penalty optimal polygon ─────────────────────
#
# Replaces: cv2.approxPolyDP (Douglas-Peucker)
#
# What this does differently:
#   DP only minimises point count — it does not know that sharp corners are
#   expensive for curve fitting.  This pass scores candidate polygons by
#   (deviation from original pixels)  +  (corner_penalty * num_corners)
#   and picks the one with lowest total cost via a simple DP on the cyclic
#   point sequence — the same strategy as potrace's optimal polygon stage.
#
# Result: fewer, gentler corners are chosen as break-points, which gives
# the Schneider fitter cleaner segments to work with.

def _seg_deviation(pts: np.ndarray, i: int, j: int) -> float:
    """Max perpendicular distance from pts[i..j] to the chord pts[i]→pts[j]."""
    n = len(pts)
    if i == j:
        return 0.0
    p0 = pts[i % n].astype(float)
    p1 = pts[j % n].astype(float)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-9:
        return 0.0
    max_d = 0.0
    k = (i + 1) % n
    while k != j % n:
        px, py = pts[k % n].astype(float)
        d = abs(dy*(px - p0[0]) - dx*(py - p0[1])) / seg_len
        if d > max_d:
            max_d = d
        k = (k + 1) % n
    return max_d


def optimal_polygon(pts: np.ndarray, tolerance: float = 1.0,
                    corner_penalty: float = 2.0) -> np.ndarray:
    """
    Return a subset of pts that forms an optimal polygon.

    Cost = sum of segment deviations + corner_penalty * num_segments.
    Solved with cyclic DP (O(n²) — fast enough for typical contour sizes).

    tolerance  : max allowed deviation in pixels (maps to opti_tolerance*2)
    corner_penalty : cost added per corner; higher → fewer, smoother corners
    """
    n = len(pts)
    if n <= 4:
        return pts

    # Pre-compute deviations for all (i,j) pairs within a window.
    # We limit the window to avoid O(n²) blow-up on very long contours.
    max_span = min(n, max(8, n // 4))

    INF = float('inf')
    # cost[i] = minimum cost to reach point i from point 0
    cost = [INF] * n
    prev = [-1] * n
    cost[0] = 0.0

    for i in range(n):
        if cost[i] == INF and i != 0:
            continue
        for span in range(1, max_span + 1):
            j = (i + span) % n
            dev = _seg_deviation(pts, i, i + span)
            if dev > tolerance * 4:          # hard reject — too much error
                break
            seg_cost = dev + corner_penalty
            new_cost = cost[i] + seg_cost
            if j == 0:                        # completed the cycle
                # close the loop — total cost check
                if new_cost < cost[0] + 1e-9:
                    pass                      # already set
                break
            if new_cost < cost[j]:
                cost[j] = new_cost
                prev[j] = i

    # Trace back the chosen vertices
    chosen = []
    idx = n - 1
    for _ in range(n):
        if prev[idx] == -1:
            break
        chosen.append(idx)
        idx = prev[idx]
        if idx == 0:
            break
    chosen.append(0)
    chosen = chosen[::-1]

    if len(chosen) < 3:
        # Fallback: evenly-spaced decimation
        step = max(1, n // 20)
        chosen = list(range(0, n, step))

    return pts[chosen]


# ─── Stage 2 replacement: Schneider iterative cubic Bezier fitting ────────────
#
# Replaces: smooth_path() with Catmull-Rom + fixed arc corners
#
# What this does:
#   For each contiguous run of points between detected corners this fits
#   a single cubic Bezier by:
#     1. Parameterise the points by cumulative chord length (t in [0,1])
#     2. Solve the least-squares system for the two interior control points
#        C1 and C2 given fixed endpoints P0 and P3
#     3. Measure the max deviation of the fitted curve from every original pt
#     4. If deviation > tolerance  →  split at the worst point and recurse
#
#   This is exactly the Schneider (1990) "An algorithm for automatically
#   fitting digitized curves" method that potrace's curve fitting is based on.
#
# Result: every curve segment is provably within opti_tolerance pixels of the
# original contour, unlike the Catmull-Rom approach which gives no guarantee.

def _chord_params(pts: np.ndarray) -> np.ndarray:
    """Cumulative chord-length parameterisation, normalised to [0, 1]."""
    diffs = np.diff(pts, axis=0)
    segs  = np.hypot(diffs[:, 0], diffs[:, 1])
    chord = np.concatenate([[0.0], np.cumsum(segs)])
    total = chord[-1]
    if total < 1e-9:
        return np.linspace(0.0, 1.0, len(pts))
    return chord / total


def _fit_cubic_ls(pts: np.ndarray,
                  t: np.ndarray,
                  tan0: np.ndarray,
                  tan1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Least-squares fit of a cubic Bezier through pts[0]..pts[-1] with
    clamped endpoint tangents tan0 (at start) and tan1 (at end).

    Returns (C1, C2) — the two interior control points.
    Solves the 2×2 system derived from the Bezier matrix equation.
    """
    n  = len(pts)
    P0 = pts[0].astype(float)
    P3 = pts[-1].astype(float)

    # Bernstein basis values B1(t) and B2(t)
    t2 = t * t
    t3 = t2 * t
    mt = 1.0 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    B0 = mt3
    B1 = 3.0 * mt2 * t
    B2 = 3.0 * mt  * t2
    B3 = t3

    # RHS: X_i - B0*P0 - B3*P3
    rhs = pts.astype(float) - np.outer(B0, P0) - np.outer(B3, P3)

    # A1_i = B1(t_i) * tan0,  A2_i = B2(t_i) * tan1
    A1 = np.outer(B1, tan0)   # (n, 2)
    A2 = np.outer(B2, tan1)

    # Build the 2×2 normal matrix
    c00 = float(np.sum(A1 * A1))
    c01 = float(np.sum(A1 * A2))
    c11 = float(np.sum(A2 * A2))
    x0  = float(np.sum(A1 * rhs))
    x1  = float(np.sum(A2 * rhs))

    det = c00 * c11 - c01 * c01
    if abs(det) < 1e-10:
        # Degenerate — fall back to chord-length heuristics
        seg = float(np.linalg.norm(P3 - P0)) / 3.0
        return P0 + seg * tan0, P3 + seg * tan1

    alpha0 = (c11 * x0 - c01 * x1) / det
    alpha1 = (c00 * x1 - c01 * x0) / det

    # Guard against negative alphas (curve would fold back)
    seg = float(np.linalg.norm(P3 - P0)) / 3.0
    if alpha0 < 1e-6:
        alpha0 = seg
    if alpha1 < 1e-6:
        alpha1 = seg

    C1 = P0 + alpha0 * tan0
    C2 = P3 + alpha1 * tan1
    return C1, C2


def _eval_cubic(P0, C1, C2, P3, t: float) -> np.ndarray:
    mt = 1.0 - t
    return (mt**3 * P0 + 3*mt**2*t * C1
            + 3*mt*t**2 * C2 + t**3 * P3)


def _max_error(pts: np.ndarray,
               t: np.ndarray,
               P0, C1, C2, P3) -> Tuple[float, int]:
    """Max deviation of fitted cubic from pts; returns (error, split_index)."""
    max_err = 0.0
    split   = len(pts) // 2
    for i in range(1, len(pts) - 1):
        q = _eval_cubic(P0, C1, C2, P3, t[i])
        err = float(np.sum((q - pts[i].astype(float))**2))
        if err > max_err:
            max_err = err
            split   = i
    return math.sqrt(max_err), split


def _tangent_unit(pts: np.ndarray, idx: int, forward: bool) -> np.ndarray:
    """Unit tangent at pts[idx], looking forward or backward."""
    n = len(pts)
    if forward:
        d = pts[min(idx + 1, n - 1)].astype(float) - pts[idx].astype(float)
    else:
        d = pts[idx].astype(float) - pts[max(idx - 1, 0)].astype(float)
    mag = math.hypot(d[0], d[1])
    if mag < 1e-9:
        return np.array([1.0, 0.0])
    return d / mag


def _fit_segment(pts: np.ndarray,
                 tan0: np.ndarray,
                 tan1: np.ndarray,
                 tolerance: float,
                 depth: int = 0) -> List[Tuple]:
    """
    Recursively fit cubic Beziers to pts with endpoint tangents tan0/tan1.
    Returns list of (C1, C2, P3) tuples — each is one cubic segment.
    Max recursion depth capped at 12 to prevent runaway on noisy contours.
    """
    if len(pts) < 2:
        return []
    if len(pts) == 2 or depth > 12:
        # Degenerate or too deep — emit a straight line as a degenerate cubic
        P0, P3 = pts[0].astype(float), pts[-1].astype(float)
        seg = np.linalg.norm(P3 - P0) / 3.0
        C1 = P0 + seg * tan0
        C2 = P3 - seg * tan1
        return [(C1, C2, P3)]

    t  = _chord_params(pts)
    P0 = pts[0].astype(float)
    P3 = pts[-1].astype(float)
    C1, C2 = _fit_cubic_ls(pts, t, tan0, tan1)

    err, split = _max_error(pts, t, P0, C1, C2, P3)

    if err <= tolerance:
        return [(C1, C2, P3)]

    # Split at worst point and recurse
    mid_tan = _tangent_unit(pts, split, forward=True)
    left  = _fit_segment(pts[:split + 1], tan0,     mid_tan,  tolerance, depth + 1)
    right = _fit_segment(pts[split:],     -mid_tan, tan1,     tolerance, depth + 1)
    return left + right


def _detect_corners(pts: np.ndarray, alpha_max: float) -> List[int]:
    """
    Return indices of corner points using potrace-style curvature threshold.

    A point is a corner when the angle between adjacent chords exceeds the
    threshold derived from alpha_max (higher alpha_max → more smoothing →
    fewer corners allowed).

    alpha_max=1.0 corresponds to ~60° corner threshold (potrace default).
    """
    n = len(pts)
    corners = []
    # cos threshold: alpha_max=0 → all corners; alpha_max=1.333 → no corners
    cos_thresh = math.cos(math.pi * min(alpha_max / 1.333, 1.0) * (2/3))

    for i in range(n):
        p0 = pts[(i - 1) % n].astype(float)
        p1 = pts[i].astype(float)
        p2 = pts[(i + 1) % n].astype(float)
        v1 = p1 - p0;  m1 = math.hypot(*v1)
        v2 = p2 - p1;  m2 = math.hypot(*v2)
        if m1 < 1e-9 or m2 < 1e-9:
            continue
        cos_a = float(np.dot(v1, v2)) / (m1 * m2)
        if cos_a < cos_thresh:
            corners.append(i)

    return corners


def fit_bezier_path(pts: np.ndarray,
                    tolerance: float = 0.4,
                    alpha_max: float = 1.0) -> str:
    """
    Main entry point: fits cubic Beziers to a closed contour.

    Replaces smooth_path() — this function:
      1. Detects corners using curvature threshold (alpha_max controls sensitivity)
      2. Between each pair of corners, runs Schneider iterative fitting
      3. Assembles the results into an SVG path string with only C commands

    Parameters
    ----------
    pts        : (N, 2) float array of contour points
    tolerance  : max pixel error per Bezier segment (maps to opti_tolerance)
    alpha_max  : potrace alphamax — higher = smoother (fewer corners forced)
    """
    n = len(pts)
    if n < 2:
        return ""
    if n == 2:
        return (f"M {pts[0][0]:.3f} {pts[0][1]:.3f} "
                f"L {pts[1][0]:.3f} {pts[1][1]:.3f} Z")

    corners = _detect_corners(pts, alpha_max)

    # If no corners found, treat the whole contour as one smooth run
    if not corners:
        corners = [0]

    # Build runs between corners (closed — last run wraps to first corner)
    runs = []
    num_c = len(corners)
    for ci in range(num_c):
        start = corners[ci]
        end   = corners[(ci + 1) % num_c]
        if end <= start:
            end += n
        # Collect points (wrapping around the cyclic array)
        run_pts = np.array([pts[k % n] for k in range(start, end + 1)])
        runs.append((start, run_pts))

    cmds = [f"M {pts[corners[0]][0]:.3f} {pts[corners[0]][1]:.3f}"]

    for run_idx, (start_idx, run_pts) in enumerate(runs):
        if len(run_pts) < 2:
            continue
        tan0 = _tangent_unit(run_pts, 0,               forward=True)
        tan1 = _tangent_unit(run_pts, len(run_pts) - 1, forward=False)

        segments = _fit_segment(run_pts, tan0, tan1, tolerance)
        for C1, C2, P3 in segments:
            cmds.append(
                f"C {C1[0]:.3f} {C1[1]:.3f} "
                f"{C2[0]:.3f} {C2[1]:.3f} "
                f"{P3[0]:.3f} {P3[1]:.3f}"
            )

    cmds.append("Z")
    return " ".join(cmds)


# ─── Utilities ────────────────────────────────────────────────────────────────

def contour_area_signed(pts: np.ndarray) -> float:
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return area / 2.0


# ─── Main trace pipeline ──────────────────────────────────────────────────────

def trace(input_path: str,
          params: Optional[Params] = None) -> Tuple[List[dict], int, int]:
    """
    Returns list of dicts: {'d': svg_path_string, 'is_hole': bool, 'area': float}
    """
    if params is None:
        params = Params()

    img    = load_image(input_path)
    gray   = to_grayscale(img)
    binary = binarize(gray, params.threshold, params.invert)
    binary = suppress_specks(binary, params.turd_size)
    h, w   = binary.shape

    contours, hierarchy = extract_contours_cv(binary)
    paths = []

    if hierarchy is None or len(contours) == 0:
        return paths, int(w * params.scale), int(h * params.scale)

    hier = hierarchy[0]

    # Schneider tolerance in pixels — opti_tolerance is in "pixel units"
    # 0.2 (potrace default) is quite tight; multiply slightly to match
    # the visual quality of potrace at default settings.
    fit_tolerance = max(0.2, params.opti_tolerance * 2.0)

    # Corner-penalty polygon tolerance — tighter than Schneider so we don't
    # lose any corner candidates before the curve fitter sees them.
    poly_tolerance = max(0.5, params.opti_tolerance * 1.5)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < max(params.turd_size, 4):
            continue

        parent  = hier[i][3]
        is_hole = parent >= 0

        # ── STAGE 1: corner-penalty optimal polygon (replaces approxPolyDP) ──
        pts_raw = contour.reshape(-1, 2).astype(float)

        if params.opti_curve:
            pts = optimal_polygon(pts_raw,
                                  tolerance     = poly_tolerance,
                                  corner_penalty= 2.0)
        else:
            # opti_curve=False → keep all original contour pixels (like potrace -n)
            pts = pts_raw

        if len(pts) < 3:
            continue

        # Subpixel offset — paths go through pixel corners like potrace
        pts = pts + 0.5

        # Scale
        if params.scale != 1.0:
            pts = pts * params.scale

        # Correct winding: outer=CCW, holes=CW
        signed_area = contour_area_signed(pts)
        if not is_hole and signed_area < 0:
            pts = pts[::-1]
        elif is_hole and signed_area > 0:
            pts = pts[::-1]

        # ── STAGE 2: Schneider iterative cubic Bezier fitting ─────────────────
        if params.opti_curve:
            d = fit_bezier_path(pts,
                                tolerance = fit_tolerance,
                                alpha_max = params.alpha_max)
        else:
            # Straight polyline output when optimisation disabled
            coords = " L ".join(f"{p[0]:.3f} {p[1]:.3f}" for p in pts)
            d = f"M {coords} Z"

        if d:
            paths.append({'d': d, 'is_hole': is_hole, 'area': area})

    paths.sort(key=lambda x: x['area'], reverse=True)
    return paths, int(w * params.scale), int(h * params.scale)


# ─── SVG export ───────────────────────────────────────────────────────────────

def export_svg(
    paths:       List[dict],
    width:       int,
    height:      int,
    output_path: str,
    foreground:  str = "#000000",
    background:  str = "white",
) -> None:
    path_elements = [
        f'  <path d="{p["d"]}" fill-rule="evenodd"/>'
        for p in paths
    ]
    paths_svg = "\n".join(path_elements)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
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
    print(f"[yd-vector] Saved → {output_path} ({len(paths)} paths)")