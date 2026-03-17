import argparse
import sys
import os
import cv2
import numpy as np
from PIL import Image

def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    # If the segment is just a point, return the distance to that point
    if ab_len_sq == 0:
        return np.linalg.norm(p - a)
    # Project point on line and clamp to segment endpoints
    t = np.dot(ap, ab) / ab_len_sq
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def find_optimal_polygon(path, opttolerance=1.0):
    """
    Greedy Polygon Approximation algorithm (The Potrace Way):
    Finds the longest possible segment that traces the pixel boundaries 
    without deviating more than `opttolerance`.
    """
    N = len(path)
    if N <= 2:
        return list(range(N))
        
    poly_indices = []
    i = 0
    while i < N:
        poly_indices.append(i)
        a = path[i]
        best_j = i + 1
        
        # Look ahead for longest valid segment
        for j in range(i + 2, N + 1): 
            # j=N maps to index 0, allowing natural connection of the closed shape
            idx_j = j % N
            b = path[idx_j]
            valid = True
            
            # Distance check all intermediate target points between i and j
            for k in range(i + 1, j):
                dist = point_to_segment_dist(path[k % N], a, b)
                if dist > opttolerance:
                    valid = False
                    break
                    
            if valid:
                best_j = j
            else:
                break
                
        i = best_j
        
    return poly_indices

def is_corner(V_prev, V_curr, V_next, alphamax_deg):
    """
    Determine if a vertex is a sharp corner by measuring the direction angle change.
    """
    vec1 = V_curr - V_prev
    vec2 = V_next - V_curr
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return True # Degenerate lines default to sharp corners
        
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2
    
    # Inner dot product to calculate angle change
    dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_change_deg = np.arccos(dot) * 180.0 / np.pi
    
    return angle_change_deg > alphamax_deg

def fit_bezier(points, P0, P3, T0, T3):
    """
    Calculates the best fitting cubic Bezier curve that interpolates through points.
    Implementation of Philip J. Schneider's Least-Squares Curve Fitting algorithm.
    """
    m = len(points)
    dist = np.linalg.norm(P3 - P0)
    
    if m <= 2:
        return dist / 3.0, dist / 3.0
        
    # Calculate chord lengths to use as our `t` param estimators
    L = 0.0
    chord_lengths = [0.0]
    for i in range(1, m):
        d = np.linalg.norm(points[i] - points[i-1])
        L += d
        chord_lengths.append(L)
        
    if L == 0:
        return 0.0, 0.0
        
    t = [cl / L for cl in chord_lengths]
    
    # Initialize Least Squares matrices and vectors
    C11 = C12 = C22 = X1 = X2 = 0.0
    
    for i in range(m):
        ti = t[i]
        inv_t = 1.0 - ti
        
        # Base scalar interpolation C(t) = P0*(...t) + P3*(...t)
        b0 = inv_t**3 + 3 * inv_t**2 * ti
        b1 = 3 * inv_t * ti**2 + ti**3
        C_ti = P0 * b0 + P3 * b1
        
        # Error delta
        Dj = points[i] - C_ti
        
        # Basis vector scaling
        a1_coeff = 3 * inv_t**2 * ti
        a2_coeff = -3 * inv_t * ti**2
        
        A1 = a1_coeff * T0
        A2 = a2_coeff * T3
        
        C11 += np.dot(A1, A1)
        C12 += np.dot(A1, A2)
        C22 += np.dot(A2, A2)
        
        X1 += np.dot(Dj, A1)
        X2 += np.dot(Dj, A2)
        
    det = C11 * C22 - C12**2
    
    # Heuristic fallback
    alpha = beta = dist / 3.0
    
    if abs(det) > 1e-8:
        a = (X1 * C22 - X2 * C12) / det
        b = (C11 * X2 - C12 * X1) / det
        
        # Prevent control points acting wildly looping beyond boundary boundaries
        max_dist = dist * 1.5
        if 0 < a < max_dist and 0 < b < max_dist:
            alpha, beta = a, b
            
    return alpha, beta

def process_contour(contour_points, opttolerance, alphamax):
    path = contour_points.astype(float)
    
    # 1. OPTIMAL POLYGON APPROXIMATION
    poly_indices = find_optimal_polygon(path, opttolerance)
    if not poly_indices:
        return None
        
    m = len(poly_indices)
    if m < 3:
        return None
        
    V = path[poly_indices]
    
    # 2. CORNER DETECTION
    corners = []
    for k in range(m):
        prev_k = (k - 1) % m
        next_k = (k + 1) % m
        corners.append(is_corner(V[prev_k], V[k], V[next_k], alphamax))
        
    # 3. BEZIER CURVE CONVERSION
    svg_segments = []
    for k in range(m):
        next_k = (k + 1) % m
        
        idx_start = poly_indices[k]
        idx_end = poly_indices[next_k]
        
        # Isolate the true pixel points belonging to this segment interpolation
        if idx_start < idx_end:
            segment_points = path[idx_start : idx_end + 1]
        else:
            segment_points = np.concatenate((path[idx_start:], path[:idx_end+1]))
            
        V_k = V[k]
        V_next = V[next_k]
        
        c_start = corners[k]
        c_end = corners[next_k]
        
        # Tangent mapping: ensuring C1 visual continuity out of smooth vertices
        if c_start:
            T_out = V_next - V_k
        else:
            prev_k = (k - 1) % m
            T_out = V_next - V[prev_k]
            
        if c_end:
             # Backward tangent for P3 aligns exactly with segment line
            T_in = V_next - V_k
        else:
            nnext_k = (next_k + 1) % m
            T_in = V[nnext_k] - V_k
            
        norm_out = np.linalg.norm(T_out)
        T_out = T_out / norm_out if norm_out > 0 else np.array([0., 0.])
        
        norm_in = np.linalg.norm(T_in)
        T_in = T_in / norm_in if norm_in > 0 else np.array([0., 0.])
        
        # If boundary is between two straight lines, L command is completely optimal
        if c_start and c_end:
            svg_segments.append(('L', V_next))
            continue
            
        # Fit cubic bezier geometry via Schneider's Least Squares Math
        alpha, beta = fit_bezier(segment_points, V_k, V_next, T_out, T_in)
        
        P1 = V_k + alpha * T_out
        P2 = V_next - beta * T_in
        
        svg_segments.append(('C', P1, P2, V_next))
        
    return V[0], svg_segments

def vectorize_image(image_path, out_svg_path, threshold=128, opttolerance=1.0, alphamax=60.0, invert=True):
    # 1. IMAGE I/O & BINARIZATION 
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img)
    _, binary = cv2.threshold(img_arr, threshold, 255, cv2.THRESH_BINARY)
    
    # Default traces white elements. 
    # Shapes rendered in black outline images must be inverted.
    if invert:
        binary = cv2.bitwise_not(binary)
        
    width, height = img.size
        
    # BOUNDARY HOLE EXTRACTION WITH HIERARCHY TRACE
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("No paths found.")
        return
        
    # 4. SVG GENERATION (EvenOdd Rule eliminates Winding Subtlety Needs)
    svg_paths = []
    
    for c in contours:
        contour_points = c.reshape(-1, 2)
        if len(contour_points) < 3:
            continue
            
        result = process_contour(contour_points, opttolerance, alphamax)
        if result is None:
            continue
            
        start_pt, segments = result
        
        # Build SVG formatting payload for this specific vector iteration geometry
        d = []
        d.append(f"M {start_pt[0]:.2f} {start_pt[1]:.2f}")
        for seg in segments:
            if seg[0] == 'L':
                d.append(f"L {seg[1][0]:.2f} {seg[1][1]:.2f}")
            elif seg[0] == 'C':
                P1, P2, P3 = seg[1], seg[2], seg[3]
                d.append(f"C {P1[0]:.2f} {P1[1]:.2f}, {P2[0]:.2f} {P2[1]:.2f}, {P3[0]:.2f} {P3[1]:.2f}")
        d.append("Z")
        svg_paths.append(" ".join(d))
        
    svg_content = [
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <!-- fill-rule="evenodd" is absolutely crucial here to properly subtract internal holes parsed out earlier -->',
        f'  <path fill-rule="evenodd" d="{" ".join(svg_paths)}" fill="#000000" />',
        f'</svg>'
    ]
    
    with open(out_svg_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg_content))
    print(f"Optimized SVG written to: {out_svg_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Python Potrace-style Vectorizer Engine")
    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Output SVG file")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold (0-255)")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Polygon approximation tolerance (opttolerance)")
    parser.add_argument("--alphamax", type=float, default=60.0, help="Corner penalty threshold angle in degrees")
    parser.add_argument("--no-invert", action="store_true", help="Do not invert image (trace white shapes instead of black)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
        
    vectorize_image(
        args.input, 
        args.output, 
        threshold=args.threshold,
        opttolerance=args.tolerance,
        alphamax=args.alphamax,
        invert=not args.no_invert
    )
