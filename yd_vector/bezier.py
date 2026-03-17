"""Cubic Bézier fitting for bitmap-traced contours.

Public API
----------
chord_length_parameterize(points)
    Returns t values from 0 to 1 based on cumulative chord distance.

fit_cubic_bezier(points, t_params=None)
    Fits a cubic Bézier to a point list using chord-length parameterization
    (least-squares, fixed endpoints). Returns (P0, P1, P2, P3).

detect_corners(points, threshold=0.85)
    Computes dot product of consecutive direction vectors; returns indices
    where corners occur (lower threshold → more corners detected).

contour_to_svg_path(points, closed=True, ...)
    Splits contour at corners into smooth segments; fits cubic Bézier to each
    segment with ≥ bezier_min_points points, uses L for shorter segments.
    At each real corner, inserts a micro-arc transition (L entry + A exit)
    instead of a sharp join.  Near-perfect circles are emitted as A arcs.
    Returns an SVG path data string.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from yd_vector.models import Point


# ---------------------------------------------------------------------------
# Chord-length parameterization
# ---------------------------------------------------------------------------

def chord_length_parameterize(points: List[Point]) -> List[float]:
    """Return t values in [0, 1] proportional to cumulative arc length."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    dists = [0.0]
    for i in range(1, n):
        d = math.hypot(points[i][0] - points[i - 1][0],
                       points[i][1] - points[i - 1][1])
        dists.append(dists[-1] + d)

    total = dists[-1]
    if total == 0.0:
        return [float(i) / (n - 1) for i in range(n)]

    return [d / total for d in dists]


# Keep old name as alias
chord_parameterize = chord_length_parameterize


# ---------------------------------------------------------------------------
# Cubic Bézier fitting  (least-squares, fixed endpoints)
# ---------------------------------------------------------------------------

def _bernstein(t: float) -> Tuple[float, float, float, float]:
    mt = 1.0 - t
    return (mt ** 3, 3.0 * mt ** 2 * t, 3.0 * mt * t ** 2, t ** 3)


def _fit_cubic_bezier_raw(
    points: List[Point],
    t_params: List[float],
) -> Tuple[Point, Point, Point, Point]:
    """Fit a cubic Bézier given points and pre-computed t parameters."""
    n   = len(points)
    p0  = points[0]
    p3  = points[-1]

    if n < 2:
        return p0, p0, p3, p3

    if n == 2:
        p1 = (p0[0] + (p3[0] - p0[0]) / 3.0,
              p0[1] + (p3[1] - p0[1]) / 3.0)
        p2 = (p0[0] + 2.0 * (p3[0] - p0[0]) / 3.0,
              p0[1] + 2.0 * (p3[1] - p0[1]) / 3.0)
        return p0, p1, p2, p3

    A = np.zeros((n, 4), dtype=float)
    for i, t in enumerate(t_params):
        A[i] = _bernstein(t)

    px = np.array([p[0] for p in points], dtype=float)
    py = np.array([p[1] for p in points], dtype=float)

    qx = px - A[:, 0] * p0[0] - A[:, 3] * p3[0]
    qy = py - A[:, 0] * p0[1] - A[:, 3] * p3[1]

    A_inner = A[:, 1:3]
    AtA     = A_inner.T @ A_inner
    Atqx    = A_inner.T @ qx
    Atqy    = A_inner.T @ qy

    try:
        sol_x = np.linalg.solve(AtA, Atqx)
        sol_y = np.linalg.solve(AtA, Atqy)
    except np.linalg.LinAlgError:
        p1 = (p0[0] + (p3[0] - p0[0]) / 3.0,
              p0[1] + (p3[1] - p0[1]) / 3.0)
        p2 = (p0[0] + 2.0 * (p3[0] - p0[0]) / 3.0,
              p0[1] + 2.0 * (p3[1] - p0[1]) / 3.0)
        return p0, p1, p2, p3

    return p0, (float(sol_x[0]), float(sol_y[0])), (float(sol_x[1]), float(sol_y[1])), p3


def fit_cubic_bezier(
    points: List[Point],
    t_params: Optional[List[float]] = None,
) -> Tuple[Point, Point, Point, Point]:
    """Fit a cubic Bézier to *points* (fixed endpoints, least-squares).

    Uses chord-length parameterization when *t_params* is not supplied.
    Returns ``(P0, P1, P2, P3)``.
    """
    if t_params is None:
        t_params = chord_length_parameterize(points)
    return _fit_cubic_bezier_raw(points, t_params)


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------

def detect_corners(points: List[Point], threshold: float = 0.85) -> List[int]:
    """Return indices of corner vertices using dot-product cosine threshold.

    - ``threshold = 1.0``  → no corners detected
    - ``threshold = 0.85`` → ~32° bend triggers a corner  (default)
    - ``threshold = 0.0``  → only 90° bends or sharper
    - ``threshold = -1.0`` → every vertex is a corner
    """
    n = len(points)
    if n < 3:
        return list(range(n))

    corners: List[int] = []
    for i in range(n):
        prev = points[(i - 1) % n]
        curr = points[i]
        nxt  = points[(i + 1) % n]

        dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
        dx2, dy2 = nxt[0]  - curr[0], nxt[1]  - curr[1]

        mag1 = math.hypot(dx1, dy1)
        mag2 = math.hypot(dx2, dy2)

        if mag1 == 0.0 or mag2 == 0.0:
            corners.append(i)
            continue

        cos_theta = (dx1 * dx2 + dy1 * dy2) / (mag1 * mag2)
        if max(-1.0, min(1.0, cos_theta)) <= threshold:
            corners.append(i)

    return corners


# ---------------------------------------------------------------------------
# Winding helpers
# ---------------------------------------------------------------------------

def _signed_area(points: List[Point]) -> float:
    n    = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return area / 2.0


def _ensure_ccw(points: List[Point]) -> List[Point]:
    return list(reversed(points)) if _signed_area(points) < 0 else list(points)


def _ensure_cw(points: List[Point]) -> List[Point]:
    return list(reversed(points)) if _signed_area(points) > 0 else list(points)


# ---------------------------------------------------------------------------
# Corner micro-arc helpers
# ---------------------------------------------------------------------------

_CORNER_OFFSET = 0.8   # px — distance from corner point to arc endpoints


def _corner_arc(
    pts: List[Point],
    ci: int,
    n: int,
    offset: float,
    r_sharp: float,
    r_soft: float,
) -> Dict:
    """Compute arc entry/exit and sweep for a single corner vertex.

    Returns a dict with keys:
      entry   — point offset px before the corner (on incoming segment)
      exit    — point offset px after  the corner (on outgoing segment)
      r       — arc radius (r_sharp if dot < 0.5, else r_soft)
      sweep   — SVG sweep-flag (1 if left-turn in path direction, else 0)
    """
    prev = pts[(ci - 1) % n]
    curr = pts[ci]
    nxt  = pts[(ci + 1) % n]

    dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
    dx2, dy2 = nxt[0]  - curr[0], nxt[1]  - curr[1]

    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)

    if mag1 == 0.0 or mag2 == 0.0:
        # Degenerate — zero-length edge; no arc
        return {'entry': curr, 'exit': curr, 'r': r_sharp, 'sweep': 0}

    in_dx, in_dy   = dx1 / mag1, dy1 / mag1
    out_dx, out_dy = dx2 / mag2, dy2 / mag2

    dot   = max(-1.0, min(1.0, in_dx * out_dx + in_dy * out_dy))
    cross = in_dx * out_dy - in_dy * out_dx   # >0 = left-turn in path direction

    r = r_sharp if dot < 0.5 else r_soft

    # Clamp so entry/exit never overshoot the neighbouring edge
    actual_offset = min(offset, mag1 * 0.45, mag2 * 0.45)

    entry = (curr[0] - actual_offset * in_dx,  curr[1] - actual_offset * in_dy)
    exit_ = (curr[0] + actual_offset * out_dx, curr[1] + actual_offset * out_dy)

    return {
        'entry': entry,
        'exit':  exit_,
        'r':     r,
        'sweep': 1 if cross > 0 else 0,
    }


def _compute_corner_arcs(
    pts: List[Point],
    corners: List[int],
    n: int,
    offset: float,
    r_sharp: float,
    r_soft: float,
) -> List[Dict]:
    return [
        _corner_arc(pts, ci, n, offset, r_sharp, r_soft)
        for ci in corners
    ]


# ---------------------------------------------------------------------------
# Circle detection & emission
# ---------------------------------------------------------------------------

def _detect_circle(
    points: List[Point],
    aspect_tol: float = 0.10,
    cv_tol: float     = 0.06,
    circ_tol: float   = 0.85,
) -> Optional[Tuple[float, float, float]]:
    """Return ``(cx, cy, r)`` if the contour is a near-perfect circle, else ``None``.

    Tests:
    1. Bounding-box aspect ratio ≥ (1 - aspect_tol)  (close to square)
    2. Radius coefficient of variation (std/mean) ≤ cv_tol
    3. Isoperimetric circularity = 4π·area / perimeter² ≥ circ_tol
    """
    n = len(points)
    if n < 8:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)

    if w == 0.0 or h == 0.0:
        return None

    if min(w, h) / max(w, h) < 1.0 - aspect_tol:
        return None

    cx = sum(xs) / n
    cy = sum(ys) / n

    dists   = [math.hypot(p[0] - cx, p[1] - cy) for p in points]
    mean_r  = sum(dists) / n
    if mean_r == 0.0:
        return None

    std_r = math.sqrt(sum((d - mean_r) ** 2 for d in dists) / n)
    if std_r / mean_r > cv_tol:
        return None

    area = abs(_signed_area(points))
    perim = sum(
        math.hypot(points[(i + 1) % n][0] - points[i][0],
                   points[(i + 1) % n][1] - points[i][1])
        for i in range(n)
    )
    if perim == 0.0:
        return None

    circularity = 4.0 * math.pi * area / (perim ** 2)
    if circularity < circ_tol:
        return None

    return (cx, cy, mean_r)


def _circle_to_svg_path(cx: float, cy: float, r: float, is_hole: bool) -> str:
    """Emit a full circle as two SVG arc commands.

    Outer (CCW): sweep=0 — each arc goes counter-clockwise.
    Hole  (CW):  sweep=1 — each arc goes clockwise.
    """
    sweep = 1 if is_hole else 0
    x1 = cx + r
    x2 = cx - r
    y  = cy
    return (
        f"M {x1:.3f} {y:.3f} "
        f"A {r:.3f} {r:.3f} 0 1 {sweep} {x2:.3f} {y:.3f} "
        f"A {r:.3f} {r:.3f} 0 1 {sweep} {x1:.3f} {y:.3f} "
        f"Z"
    )


# ---------------------------------------------------------------------------
# Contour → SVG path string
# ---------------------------------------------------------------------------

def contour_to_svg_path(
    points: List[Point],
    closed: bool = True,
    corner_threshold: float = 0.85,
    is_hole: bool = False,
    bezier_min_points: int = 3,
    corner_arc_radius_sharp: float = 0.42,
    corner_arc_radius_soft: float  = 0.92,
) -> str:
    """Convert a contour point list to an SVG path data string.

    At each detected corner, instead of a hard join, a micro-arc transition
    is inserted:
      L  (corner − 0.8 px along incoming direction)
      A  rx ry 0 0 [sweep] (corner + 0.8 px along outgoing direction)

    Arc radii:
      dot product < 0.5  → corner_arc_radius_sharp (0.42 by default)
      dot product 0.5–1  → corner_arc_radius_soft  (0.92 by default)

    Near-perfect circles (aspect ≈ 1, circularity ≈ 1) are replaced with
    a pair of SVG ``A`` arcs using the exact computed radius.

    Outer contours are wound CCW; holes are wound CW (fill-rule=evenodd).
    """
    if not points:
        return ""

    n = len(points)
    if n < 3:
        cmds = [f"M {points[0][0]:.3f} {points[0][1]:.3f}"]
        for i in range(1, n):
            cmds.append(f"L {points[i][0]:.3f} {points[i][1]:.3f}")
        if closed:
            cmds.append("Z")
        return " ".join(cmds)

    # ── Circle fast-path ────────────────────────────────────────────────────
    circle = _detect_circle(points)
    if circle is not None:
        cx, cy, r = circle
        return _circle_to_svg_path(cx, cy, r, is_hole)

    # ── Winding ─────────────────────────────────────────────────────────────
    pts: List[Point] = _ensure_cw(points) if is_hole else _ensure_ccw(points)

    # ── Corner detection ────────────────────────────────────────────────────
    real_corners = sorted(set(detect_corners(pts, threshold=corner_threshold)))
    use_arcs     = len(real_corners) > 0

    if real_corners:
        corners = real_corners
    else:
        # No real corners — distribute anchor points for smooth Bézier loop
        num_anchors = max(4, n // 20)
        corners = [i * n // num_anchors for i in range(num_anchors)]

    num_corners = len(corners)

    # ── Precompute arc data (only for real corners) ─────────────────────────
    if use_arcs:
        arc_info = _compute_corner_arcs(
            pts, corners, n, _CORNER_OFFSET,
            corner_arc_radius_sharp, corner_arc_radius_soft,
        )
        # Start at the exit point of the first corner
        start = arc_info[0]['exit']
    else:
        arc_info = None
        start    = pts[corners[0]]

    cmds = [f"M {start[0]:.3f} {start[1]:.3f}"]

    for seg_i in range(num_corners):
        from_idx  = corners[seg_i]
        to_idx    = corners[(seg_i + 1) % num_corners]

        # Steps forward (modular) from from_idx to to_idx
        steps = (to_idx - from_idx) % n
        if steps == 0:
            steps = n  # single-corner contour: wrap full loop

        seg_pts: List[Point] = [pts[(from_idx + s) % n] for s in range(steps + 1)]

        if use_arcs:
            # Replace segment endpoints with arc exit/entry so that the
            # Bézier spans the trimmed segment (not into the corner radius).
            next_arc = arc_info[(seg_i + 1) % num_corners]
            seg_pts[0]  = arc_info[seg_i]['exit']
            seg_pts[-1] = next_arc['entry']
        else:
            next_arc = None

        # ── Emit segment ─────────────────────────────────────────────────
        if len(seg_pts) < bezier_min_points:
            ep = seg_pts[-1]
            cmds.append(f"L {ep[0]:.3f} {ep[1]:.3f}")
        else:
            t_params = chord_length_parameterize(seg_pts)
            _, p1, p2, _ = _fit_cubic_bezier_raw(seg_pts, t_params)
            ep = seg_pts[-1]
            cmds.append(
                f"C {p1[0]:.3f} {p1[1]:.3f} "
                f"{p2[0]:.3f} {p2[1]:.3f} "
                f"{ep[0]:.3f} {ep[1]:.3f}"
            )

        # ── Emit micro-arc at the NEXT corner ────────────────────────────
        # (also handles the wrap-around arc back through corner 0)
        if use_arcs and next_arc is not None:
            r     = next_arc['r']
            sweep = next_arc['sweep']
            ex    = next_arc['exit']
            entry = next_arc['entry']
            # L to entry (spec-mandated explicit move to arc start)
            cmds.append(f"L {entry[0]:.3f} {entry[1]:.3f}")
            # A arc through the corner to its exit
            cmds.append(
                f"A {r:.3f} {r:.3f} 0 0 {sweep} {ex[0]:.3f} {ex[1]:.3f}"
            )

    if closed:
        cmds.append("Z")
    return " ".join(cmds)
