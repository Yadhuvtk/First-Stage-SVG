"""
tracer.py — Pure Python Potrace-style Vectorizer Engine
========================================================
Implements the 5-step vectorization pipeline from:
    Selinger, P. (2003). "Potrace: a polygon-based tracing algorithm."
    https://potrace.sourceforge.net/potrace.pdf

STRICT CONSTRAINTS:
  - No pypotrace, potrace executable, or C/Rust bindings.
  - No cv2.approxPolyDP (Douglas-Peucker is forbidden for fit stage).
  - Only: numpy, math, cv2 (image I/O, threshold, findContours only).

USAGE:
    python tracer.py input.png output.svg [options]

CLI OPTIONS:
    --threshold   int    Grayscale threshold 0-255    (default: 128)
    --opttol      float  Polygon deviation tolerance  (default: 0.2)
    --alphamax    float  Corner smoothness threshold  (default: 1.0)
    --invert             Trace dark shapes (black ink on white bg)
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np


# =============================================================================
# SECTION 1 — GEOMETRIC MATH PRIMITIVES
# =============================================================================

def vec2(x, y):
    """Construct a 2D numpy vector."""
    return np.array([float(x), float(y)], dtype=np.float64)


def dot(a, b):
    """Standard 2D dot product: a · b."""
    return float(a[0] * b[0] + a[1] * b[1])


def cross(a, b):
    """
    2D 'cross product' (scalar z-component of 3D cross):
        cross(a, b) = a.x * b.y - a.y * b.x
    This gives the signed area of the parallelogram spanned by a, b.
    Positive = counter-clockwise turn, Negative = clockwise turn.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def norm(v):
    """Euclidean magnitude of a 2D vector."""
    return math.hypot(v[0], v[1])


def normalize(v):
    """Return a unit vector in direction of v. Returns zero-vec if degenerate."""
    n = norm(v)
    return v / n if n > 1e-12 else vec2(0.0, 0.0)


def point_to_segment_dist_sq(p, a, b):
    """
    Squared perpendicular distance from point p to segment [a, b].

    Math:
        Project p onto the infinite line through a, b:
            t = ((p - a) · (b - a)) / |b - a|²
        Clamp t to [0, 1] so that we stay on the segment.
        Return |p - (a + t*(b-a))|²
    """
    ab = b - a
    ab_sq = dot(ab, ab)
    if ab_sq < 1e-12:
        # Degenerate segment — just use distance to endpoint a
        return dot(p - a, p - a)
    t = dot(p - a, ab) / ab_sq
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    diff = p - proj
    return dot(diff, diff)


# =============================================================================
# SECTION 2 — STEP 1: PATH DECOMPOSITION
# =============================================================================

class PathDecomposer:
    """
    Extracts hierarchical contours from a binary image and converts them
    into closed lists of 2D float points (one list per contour).

    We use cv2.RETR_TREE to preserve parent-child (hole) relationships so
    that the SVG fill-rule="evenodd" can correctly punch holes (e.g., the
    inside of the letter 'O').
    """

    def __init__(self, binary_image: np.ndarray):
        """
        Args:
            binary_image: uint8 numpy array, values in {0, 255}.
        """
        self.image = binary_image
        self.height, self.width = binary_image.shape[:2]

    def extract(self):
        """
        Returns:
            List of paths, where each path is a (N, 2) float64 numpy array
            of closed contour points in pixel space.
        """
        # cv2.CHAIN_APPROX_NONE keeps every boundary pixel — mandatory for
        # Potrace-style tracing which needs the full topological boundary.
        contours, _ = cv2.findContours(
            self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        paths = []
        for c in contours:
            pts = c.reshape(-1, 2).astype(np.float64)
            if len(pts) >= 4:   # Ignore degenerate tiny contours
                paths.append(pts)
        return paths


# =============================================================================
# SECTION 3 — STEP 2: OPTIMAL POLYGON APPROXIMATION
# =============================================================================

class OptimalPolygonApproximator:
    """
    Implements Selinger's shortest-enclosing-polygon algorithm.

    Key idea (Section 2.3 of the paper):
        Walk the boundary path. From each start vertex, greedily extend
        the current segment to the farthest possible subsequent vertex such
        that ALL intermediate path points lie within `opttolerance` of the
        straight line connecting start and tentative end.

    This is fundamentally different from Douglas-Peucker:
        • D-P recursively splits — always starts with the full polyline.
        • Potrace greedily extends — starts with nothing, only grows forward.
        The result is a minimal vertex polygon that honours the boundary.

    Complexity: O(N * k) where k is max look-ahead (bounded by segment length).
    """

    def __init__(self, opttolerance: float = 0.2):
        """
        Args:
            opttolerance: Maximum allowed deviation (in pixels) from the
                          original boundary path. Selinger's default is ~0.2.
                          Lower → more vertices / more faithful.
                          Higher → fewer vertices / smoother simplification.
        """
        self.tol_sq = opttolerance * opttolerance   # Work in squared distance

    def approximate(self, path: np.ndarray) -> np.ndarray:
        """
        Args:
            path: (N, 2) float64 array of closed boundary pixels.

        Returns:
            (M, 2) float64 array of polygon vertices (M << N).
        """
        N = len(path)
        if N < 4:
            return path

        vertex_indices = []
        i = 0

        while i < N:
            vertex_indices.append(i)
            a = path[i]
            # best_end tracks the farthest index we can legally extend to
            best_end = i + 1

            for j in range(i + 2, i + N):   # never wrap more than once
                j_mod = j % N
                b = path[j_mod]

                # Check ALL intermediate points stay within tolerance
                within_tol = True
                for k in range(i + 1, j):
                    k_mod = k % N
                    d_sq = point_to_segment_dist_sq(path[k_mod], a, b)
                    if d_sq > self.tol_sq:
                        within_tol = False
                        break

                if within_tol:
                    best_end = j
                else:
                    break   # Greedy: first failure stops extension

            i = best_end

        poly = path[np.array(vertex_indices, dtype=int) % N]
        return poly


# =============================================================================
# SECTION 4 — STEP 3: CORNER DETECTION
# =============================================================================

class CornerDetector:
    """
    Classifies each polygon vertex as either a "Sharp Corner" or "Smooth".

    Mathematical basis (Section 2.4 of the paper):
        At each vertex V[i], we look at:
            - Incoming direction:  d_in  = V[i] - V[i-1]  (normalized)
            - Outgoing direction:  d_out = V[i+1] - V[i]  (normalized)

        The 'alpha' penalty is derived from the cross product (sine of angle)
        and dot product (cosine of angle) between d_in and d_out:
            cos_a = dot(d_in, d_out)
            sin_a = |cross(d_in, d_out)|   (absolute value for unsigned angle)

        We compute:
            alpha = 2 - cos_a - sin_a

        When alpha < alphamax  → Smooth vertex (curve through it)
        When alpha >= alphamax → Sharp corner (hard angle, use LineTo)

    Note on alphamax:
        alpha ∈ [0, 3]:
            alpha = 0   → perfectly straight (0° bend)
            alpha = 1   → 90° bend
            alpha ≈ 1.41 → 135° bend
            alpha = 2   → 180° reversal
        Selinger's default is alphamax = 1.0  (preserve corners > 90°).
    """

    def __init__(self, alphamax: float = 1.0):
        """
        Args:
            alphamax: Penalty threshold. Vertices with alpha >= this are
                      classified as sharp corners.
                      Lower → more corners preserved.
                      Higher → more vertices treated as smooth curves.
        """
        self.alphamax = alphamax

    def classify(self, poly: np.ndarray):
        """
        Args:
            poly: (M, 2) float64 polygon vertices.

        Returns:
            List[bool] of length M: True = sharp corner, False = smooth.
        """
        M = len(poly)
        is_corner = []

        for i in range(M):
            prev_v = poly[(i - 1) % M]
            curr_v = poly[i]
            next_v = poly[(i + 1) % M]

            d_in  = normalize(curr_v - prev_v)
            d_out = normalize(next_v - curr_v)

            # cos and sin of the turning angle at this vertex
            cos_a = dot(d_in, d_out)
            sin_a = abs(cross(d_in, d_out))

            # Alpha penalty: 0 for straight, increases with sharpness of bend
            alpha = 2.0 - cos_a - sin_a

            is_corner.append(alpha >= self.alphamax)

        return is_corner


# =============================================================================
# SECTION 5 — STEP 4: LEAST-SQUARES BÉZIER CURVE FITTING
# =============================================================================

class BezierFitter:
    """
    Fits a cubic Bézier curve to a sequence of points using Philip J.
    Schneider's Least-Squares method ("An Algorithm for Automatically
    Fitting Digitized Curves", Graphics Gems I, 1990).

    A cubic Bézier is defined by 4 control points:
        B(t) = (1-t)³ P0 + 3(1-t)²t P1 + 3(1-t)t² P2 + t³ P3,  t ∈ [0,1]

    We know P0 (start), P3 (end), tangent T0 at P0, and tangent T3 at P3.
    So we can write:
        P1 = P0 + α * T0
        P2 = P3 - β * T3

    The unknowns are the scalars α ≥ 0 and β ≥ 0.

    LEAST SQUARES FORMULATION:
    ---------------------------
    Given n data points D[i] with corresponding parameter values t[i],
    we want to minimise the sum of squared residuals:
        S = Σ |B(t[i]) - D[i]|²

    Expanding B(t) and substituting P1 = P0 + α T0, P2 = P3 - β T3:
        B(t) = b0(t) P0 + b1(t)(P0 + α T0) + b2(t)(P3 - β T3) + b3(t) P3

    where the Bernstein basis functions are:
        b0(t) = (1-t)³
        b1(t) = 3(1-t)²t
        b2(t) = 3(1-t)t²
        b3(t) = t³

    Rearranging, the residual for each point is:
        R[i] = D[i] - [(b0+b1)*P0 + (b2+b3)*P3]
             = α (b1*T0) + β (-b2*T3) + ... (higher order stuff absorbed)

    Taking ∂S/∂α = 0 and ∂S/∂β = 0 yields a 2×2 linear system:
        [ C11  C12 ] [ α ]   [ X1 ]
        [ C21  C22 ] [ β ] = [ X2 ]

    where:
        A1[i] = b1(t[i]) * T0        (contribution of α to B(t[i]))
        A2[i] = -b2(t[i]) * T3       (contribution of β to B(t[i]))

        C11 = Σ A1[i] · A1[i]
        C12 = C21 = Σ A1[i] · A2[i]
        C22 = Σ A2[i] · A2[i]

        X1  = Σ R[i] · A1[i]
        X2  = Σ R[i] · A2[i]

    Solved by Cramer's rule:
        det = C11*C22 - C12²
        α   = (X1*C22 - X2*C12) / det
        β   = (C11*X2 - C12*X1) / det
    """

    def fit(
        self,
        points: np.ndarray,
        P0: np.ndarray,
        P3: np.ndarray,
        T0: np.ndarray,
        T3: np.ndarray,
    ):
        """
        Compute optimal control points P1, P2 for the cubic Bézier through
        data points, anchored at P0 (in tangent T0) and P3 (in tangent T3).

        Args:
            points : (N, 2) float64 — the intermediate data points to fit
            P0, P3 : start and end anchors (2D float vectors)
            T0, T3 : unit tangent vectors at P0 and P3

        Returns:
            (P1, P2) — the two inner Bézier control points.
        """
        n = len(points)
        dist = norm(P3 - P0)
        # Heuristic fallback if system is degenerate or very short segment
        fallback_alpha = dist / 3.0
        fallback_beta  = dist / 3.0

        if n < 2:
            return P0 + fallback_alpha * T0, P3 - fallback_beta * T3

        # --- Chord-length parameterisation ---
        # t[0]=0 at P0, t[n-1]=1 at P3, intermediate values proportional
        # to cumulative arc length along the data point polyline.
        chord_lens = [0.0]
        for i in range(1, n):
            chord_lens.append(chord_lens[-1] + norm(points[i] - points[i - 1]))
        total_len = chord_lens[-1]

        if total_len < 1e-12:
            return P0 + fallback_alpha * T0, P3 - fallback_beta * T3

        t_params = [cl / total_len for cl in chord_lens]

        # --- Build 2×2 least-squares system C * [alpha, beta] = X ---
        C11 = C12 = C22 = 0.0
        X1  = X2  = 0.0

        for i in range(n):
            t  = t_params[i]
            mt = 1.0 - t

            # Bernstein basis
            b0 = mt ** 3
            b1 = 3.0 * mt**2 * t
            b2 = 3.0 * mt * t**2
            b3 = t ** 3

            # Basis vectors for the two free parameters (α, β)
            A1 =  b1 * T0   # how much α shifts B(t)
            A2 = -b2 * T3   # how much β shifts B(t) (negative — pulling P2 toward P3)

            # Residual: actual data point minus the 'fixed' Bézier at t
            # (i.e., the curve with P1=P0, P2=P3 — zero offset as baseline)
            R = points[i] - (b0 + b1) * P0 - (b2 + b3) * P3

            C11 += dot(A1, A1)
            C12 += dot(A1, A2)
            C22 += dot(A2, A2)
            X1  += dot(R,  A1)
            X2  += dot(R,  A2)

        det = C11 * C22 - C12 * C12

        if abs(det) < 1e-10:
            # Singular system — use heuristic 1/3 split
            alpha = fallback_alpha
            beta  = fallback_beta
        else:
            # Cramer's rule solution
            alpha = (X1 * C22 - X2 * C12) / det
            beta  = (C11 * X2 - C12 * X1) / det

        # Guard: control points must not overshoot — clamp to plausible range
        max_reach = dist * 2.0
        if alpha <= 0 or alpha > max_reach:
            alpha = fallback_alpha
        if beta <= 0 or beta > max_reach:
            beta = fallback_beta

        P1 = P0 + alpha * T0
        P2 = P3 - beta  * T3
        return P1, P2


# =============================================================================
# SECTION 6 — STEP 4 (CONT): SEGMENT BUILDER
# =============================================================================

class SegmentBuilder:
    """
    Converts an optimal polygon + corner classification into a list of typed
    SVG segments (LineTo or CubicBezier) suitable for SVG path serialization.

    For each polygon edge [V[k] → V[k+1]]:
        • If BOTH endpoints are sharp corners → LineTo.
        • Otherwise → cubic Bézier via least-squares fitting.

    Tangent computation at smooth vertices uses the 'skip-one' rule:
        At smooth vertex i, the outgoing tangent points toward V[i+1],
        but we use the vector across the vertex (from V[i-1] to V[i+1])
        to achieve C1 continuity — the tangent is shared by both edges
        meeting at i, so the curves join smoothly with no kink.
    """

    def __init__(self, fitter: BezierFitter = None):
        self.fitter = fitter or BezierFitter()

    def build(self, poly, is_corner, path):
        """
        Args:
            poly      : (M, 2) polygon vertices
            is_corner : (M,) bool — True = sharp corner
            path      : (N, 2) original full boundary path (for intermediate pts)

        Returns:
            List of segment dicts:
                {'type': 'L', 'end': array}
                {'type': 'C', 'cp1': array, 'cp2': array, 'end': array}
        """
        M = len(poly)
        segments = []

        for k in range(M):
            nk = (k + 1) % M
            V_k    = poly[k]
            V_next = poly[nk]

            c_start = is_corner[k]
            c_end   = is_corner[nk]

            # --- LineTo: both endpoints are hard corners ---
            if c_start and c_end:
                segments.append({'type': 'L', 'end': V_next})
                continue

            # --- Compute outgoing tangent at V_k ---
            if c_start:
                # Hard corner: tangent simply points toward the next vertex
                T0 = normalize(V_next - V_k)
            else:
                # Smooth: tangent spans previous to next vertex (skip-one rule)
                V_prev = poly[(k - 1) % M]
                T0 = normalize(V_next - V_prev)

            # --- Compute incoming tangent at V_next (reversed = outgoing of neighbour) ---
            if c_end:
                T3 = normalize(V_next - V_k)
            else:
                V_nnext = poly[(nk + 1) % M]
                T3 = normalize(V_nnext - V_k)

            # --- Extract intermediate path pixels for this edge ---
            mid_points = self._extract_edge_points(poly, k, path)

            # --- Least-squares Bézier fitting ---
            P1, P2 = self.fitter.fit(mid_points, V_k, V_next, T0, T3)

            segments.append({'type': 'C', 'cp1': P1, 'cp2': P2, 'end': V_next})

        return segments

    def _extract_edge_points(self, poly, k, path):
        """
        Attempt to find the original boundary pixels that belong to edge k.
        If path-to-poly mapping is too complex, returns just start+end.
        """
        M = len(poly)
        nk = (k + 1) % M
        V_k    = poly[k]
        V_next = poly[nk]

        # Look for the closest path index to V_k and V_next
        def nearest_idx(target, arr):
            diffs = arr - target
            return int(np.argmin(np.einsum('ij,ij->i', diffs, diffs)))

        if len(path) < 4:
            return np.array([V_k, V_next])

        i0 = nearest_idx(V_k, path)
        i1 = nearest_idx(V_next, path)

        if i0 == i1:
            return np.array([V_k, V_next])

        N = len(path)
        if i0 < i1:
            pts = path[i0 : i1 + 1]
        else:
            pts = np.concatenate([path[i0:], path[: i1 + 1]])

        if len(pts) > 200:
            # Thin out for speed while keeping shape
            idx = np.linspace(0, len(pts) - 1, 50, dtype=int)
            pts = pts[idx]

        return pts


# =============================================================================
# SECTION 7 — STEP 5: SVG FORMATTING
# =============================================================================

class SVGFormatter:
    """
    Serializes a list of typed segments into a valid SVG <path> string.

    SVG path command reference:
        M x y      — MoveTo: lift pen, start new subpath at (x, y)
        L x y      — LineTo: draw straight line to (x, y)
        C x1 y1, x2 y2, x y  — Cubic Bézier: control pts (x1,y1),(x2,y2), end (x,y)
        Z          — ClosePath: draw line back to last M point

    The fill-rule="evenodd" attribute is mandatory:
        With evenodd, a point is inside the shape if a ray from that point
        crosses the boundary path an ODD number of times. This correctly
        makes nested contours (e.g. the hole in 'O') transparent, regardless
        of winding direction.
    """

    def __init__(self, width: int, height: int, fill: str = "#000000"):
        self.width  = width
        self.height = height
        self.fill   = fill

    def format_num(self, v):
        """Render a float to 3 decimal places, stripping trailing zeros."""
        return f"{v:.3f}".rstrip('0').rstrip('.')

    def fmt(self, pt):
        return f"{self.format_num(pt[0])} {self.format_num(pt[1])}"

    def segments_to_path_data(self, start_pt, segments):
        """Convert start point + segment list to an SVG 'd' subpath string."""
        cmds = [f"M {self.fmt(start_pt)}"]
        for seg in segments:
            if seg['type'] == 'L':
                cmds.append(f"L {self.fmt(seg['end'])}")
            elif seg['type'] == 'C':
                cp1 = self.fmt(seg['cp1'])
                cp2 = self.fmt(seg['cp2'])
                end = self.fmt(seg['end'])
                cmds.append(f"C {cp1}, {cp2}, {end}")
        cmds.append("Z")
        return " ".join(cmds)

    def build_svg(self, all_subpath_data: list[str]) -> str:
        """
        Assemble the complete SVG document.

        All contours (outer boundaries + holes) are concatenated into a
        single <path> d attribute. The evenodd fill-rule then correctly
        handles all nesting.
        """
        combined_d = " ".join(all_subpath_data)
        lines = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg"',
            f'     version="1.1"',
            f'     width="{self.width}"',
            f'     height="{self.height}"',
            f'     viewBox="0 0 {self.width} {self.height}">',
            f'  <!--',
            f'    Generated by tracer.py — Pure Python Potrace-style Engine',
            f'    fill-rule="evenodd": nested contours (holes) render transparent.',
            f'  -->',
            f'  <path',
            f'    fill="{self.fill}"',
            f'    fill-rule="evenodd"',
            f'    d="{combined_d}"',
            f'  />',
            f'</svg>',
        ]
        return "\n".join(lines)


# =============================================================================
# SECTION 8 — ORCHESTRATION: THE MAIN PIPELINE
# =============================================================================

def vectorize(
    binary: np.ndarray,
    width: int,
    height: int,
    opttolerance: float = 0.2,
    alphamax: float = 1.0,
    fill: str = "#000000",
) -> str:
    """
    Full vectorization pipeline:
        1. Path Decomposition      — extract contours
        2. Polygon Approximation   — greedy optimal polygon
        3. Corner Detection        — classify vertices
        4. Bézier Fitting          — compute control points
        5. SVG Formatting          — serialize to SVG string

    Args:
        binary       : uint8 (H × W) binarized image, values in {0, 255}
        width/height : image dimensions in pixels
        opttolerance : max deviation for polygon approx (pixels)
        alphamax     : corner penalty threshold
        fill         : SVG fill color hex string

    Returns:
        Complete SVG document as a string.
    """
    # Instantiate pipeline components
    decomposer   = PathDecomposer(binary)
    approx       = OptimalPolygonApproximator(opttolerance)
    detector     = CornerDetector(alphamax)
    fitter       = BezierFitter()
    builder      = SegmentBuilder(fitter)
    formatter    = SVGFormatter(width, height, fill)

    # Step 1: Extract all closed boundary paths
    paths = decomposer.extract()
    if not paths:
        print("[tracer] WARNING: No contours found in image.", file=sys.stderr)

    subpath_data_list = []

    for path in paths:
        # Step 2: Optimal polygon approximation
        poly = approx.approximate(path)
        if len(poly) < 3:
            continue

        # Step 3: Corner classification
        is_corner = detector.classify(poly)

        # Step 4: Build typed segments with Bézier fitting
        segments = builder.build(poly, is_corner, path)
        if not segments:
            continue

        # Step 5: Serialize this contour to SVG path data
        subpath_str = formatter.segments_to_path_data(poly[0], segments)
        subpath_data_list.append(subpath_str)

    # Assemble full SVG
    svg = formatter.build_svg(subpath_data_list)
    return svg


# =============================================================================
# SECTION 9 — BINARIZATION HELPER
# =============================================================================

def load_and_binarize(image_path: str, threshold: int = 128, invert: bool = True):
    """
    Load an image file and produce a binary uint8 mask (0 or 255).

    Args:
        image_path : path to input image (PNG, JPEG, BMP, etc.)
        threshold  : grayscale cutoff — pixels above this → white
        invert     : if True, invert before tracing (trace dark shapes on
                     light backgrounds, i.e. standard black-ink artwork)

    Returns:
        (binary_image, width, height)
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = img_bgr.shape[:2]
    _, binary = cv2.threshold(img_bgr, threshold, 255, cv2.THRESH_BINARY)

    if invert:
        binary = cv2.bitwise_not(binary)

    return binary, width, height


# =============================================================================
# SECTION 10 — CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="tracer.py",
        description=(
            "Pure Python Potrace-style Vectorizer — "
            "implements Selinger 2003 pipeline from scratch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tracer.py logo.png logo.svg
  python tracer.py photo.jpg out.svg --threshold 100 --opttol 0.5 --alphamax 1.2
  python tracer.py white_on_black.png out.svg --no-invert --fill "#ff0000"
        """,
    )
    parser.add_argument("input",          help="Input raster image (PNG/JPEG/BMP)")
    parser.add_argument("output",         help="Output SVG file path")
    parser.add_argument(
        "--threshold", type=int, default=128,
        help="Grayscale binarization threshold 0–255 (default: 128)"
    )
    parser.add_argument(
        "--opttol", type=float, default=0.2,
        help="Optimal polygon tolerance in pixels (default: 0.2)"
    )
    parser.add_argument(
        "--alphamax", type=float, default=1.0,
        help="Corner detection threshold — 0..2+ (default: 1.0)"
    )
    parser.add_argument(
        "--no-invert", action="store_true",
        help="Do NOT invert image (trace white shapes instead of black)"
    )
    parser.add_argument(
        "--fill", default="#000000",
        help='SVG fill color hex string (default: "#000000")'
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[tracer] ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[tracer] Loading:    {args.input}")
    binary, width, height = load_and_binarize(
        args.input,
        threshold=args.threshold,
        invert=not args.no_invert,
    )
    print(f"[tracer] Image size: {width}×{height} px  |  threshold={args.threshold}")

    print(f"[tracer] Vectorizing (opttol={args.opttol}, alphamax={args.alphamax}) …")
    svg = vectorize(
        binary, width, height,
        opttolerance=args.opttol,
        alphamax=args.alphamax,
        fill=args.fill,
    )

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"[tracer] Written:    {args.output}  ✓")


if __name__ == "__main__":
    main()
