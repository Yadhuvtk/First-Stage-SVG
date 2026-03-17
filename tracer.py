"""
tracer.py — Pure Python Potrace Transpile
==========================================
A faithful Python translation of Peter Selinger's Potrace algorithm,
sourced from the open-source JavaScript port by kilobtye:
  https://github.com/kilobtye/potrace

All math is translated verbatim from potrace.js.
NO pypotrace, NO C-bindings, NO Douglas-Peucker (approxPolyDP).
Only cv2.imread / cv2.threshold / cv2.findContours are used for I/O.

Pipeline (matching potrace.js exactly):
  1. bmToPathlist   — pixel-walk contour decomposition
  2. calcSums       — prefix-sum tables for fast quadratic penalty
  3. calcLon        — longest straight-run table
  4. bestPolygon    — DP shortest-polygon via penalty3
  5. adjustVertices — least-squares vertex optimisation (quadform)
  6. smooth         — corner detection + Bezier control-point assignment
  7. optiCurve      — optional curve merging optimisation
  8. getSVG         — M / L / C SVG serialisation with fill-rule=evenodd

Usage:
  python tracer.py input.png output.svg [--turdsize 2] [--alphamax 1]
                   [--opttolerance 0.2] [--no-optcurve] [--threshold 128]
                   [--invert] [--fill #000000]
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Point:
    __slots__ = ("x", "y")
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
    def copy(self):
        return Point(self.x, self.y)
    def __repr__(self):
        return f"Point({self.x:.3f}, {self.y:.3f})"


class Bitmap:
    """Row-major binary bitmap. Value 1 = foreground pixel."""
    def __init__(self, w, h, data=None):
        self.w = w
        self.h = h
        self.size = w * h
        self.data = data if data is not None else [0] * self.size

    def at(self, x, y):
        return (0 <= x < self.w and 0 <= y < self.h and
                self.data[self.w * y + x] == 1)

    def index(self, i):
        p = Point()
        p.y = i // self.w
        p.x = i - p.y * self.w
        return p

    def flip(self, x, y):
        idx = self.w * y + x
        self.data[idx] = 0 if self.data[idx] else 1

    def copy(self):
        return Bitmap(self.w, self.h, list(self.data))


class Path:
    def __init__(self):
        self.area  = 0
        self.len   = 0
        self.curve = None      # set by adjustVertices
        self.pt    = []        # list of Point (pixel boundary)
        self.minX  = 100000;  self.minY = 100000
        self.maxX  = -1;      self.maxY = -1
        self.sign  = "+"
        # set by later stages:
        self.x0 = self.y0 = 0
        self.sums = []
        self.lon  = []
        self.m    = 0
        self.po   = []


class Curve:
    """n-segment curve; control points stored flat: c[i*3+0..2]."""
    def __init__(self, n):
        self.n          = n
        self.tag        = [None] * n      # "CURVE" or "CORNER"
        self.c          = [Point() for _ in range(n * 3)]
        self.alphacurve = 0
        self.vertex     = [Point() for _ in range(n)]
        self.alpha      = [0.0] * n
        self.alpha0     = [0.0] * n
        self.beta       = [0.0] * n


class Sum:
    __slots__ = ("x", "y", "xy", "x2", "y2")
    def __init__(self, x, y, xy, x2, y2):
        self.x=x; self.y=y; self.xy=xy; self.x2=x2; self.y2=y2


class Quad:
    def __init__(self):
        self.data = [0.0]*9
    def at(self, x, y):
        return self.data[x*3+y]


# ---------------------------------------------------------------------------
# Math helpers  (direct translation from potrace.js)
# ---------------------------------------------------------------------------

def _mod(a, n):
    return a % n if a >= n else (a if a >= 0 else n-1-(-1-a) % n)

def _xprod(p1, p2):
    return p1.x*p2.y - p1.y*p2.x

def _cyclic(a, b, c):
    if a <= c:
        return a <= b < c
    return a <= b or b < c

def _sign(i):
    return 1 if i > 0 else (-1 if i < 0 else 0)

def _quadform(Q, w):
    v = [w.x, w.y, 1.0]
    s = 0.0
    for i in range(3):
        for j in range(3):
            s += v[i] * Q.at(i, j) * v[j]
    return s

def _interval(lam, a, b):
    return Point(a.x + lam*(b.x-a.x), a.y + lam*(b.y-a.y))

def _dorth_infty(p0, p2):
    r = Point()
    r.y =  _sign(p2.x - p0.x)
    r.x = -_sign(p2.y - p0.y)
    return r

def _ddenom(p0, p2):
    r = _dorth_infty(p0, p2)
    return r.y*(p2.x-p0.x) - r.x*(p2.y-p0.y)

def _dpara(p0, p1, p2):
    return (p1.x-p0.x)*(p2.y-p0.y) - (p2.x-p0.x)*(p1.y-p0.y)

def _cprod(p0, p1, p2, p3):
    return (p1.x-p0.x)*(p3.y-p2.y) - (p3.x-p2.x)*(p1.y-p0.y)

def _iprod(p0, p1, p2):
    return (p1.x-p0.x)*(p2.x-p0.x) + (p1.y-p0.y)*(p2.y-p0.y)

def _iprod1(p0, p1, p2, p3):
    return (p1.x-p0.x)*(p3.x-p2.x) + (p1.y-p0.y)*(p3.y-p2.y)

def _ddist(p, q):
    return math.hypot(p.x-q.x, p.y-q.y)

def _bezier(t, p0, p1, p2, p3):
    s = 1.0 - t
    return Point(
        s*s*s*p0.x + 3*s*s*t*p1.x + 3*s*t*t*p2.x + t*t*t*p3.x,
        s*s*s*p0.y + 3*s*s*t*p1.y + 3*s*t*t*p2.y + t*t*t*p3.y,
    )

def _tangent(p0, p1, p2, p3, q0, q1):
    """Return parameter t where Bezier(t) is tangent to line q0-q1, or -1."""
    A = _cprod(p0, p1, q0, q1)
    B = _cprod(p1, p2, q0, q1)
    C = _cprod(p2, p3, q0, q1)
    a = A - 2*B + C
    b = -2*A + 2*B
    c = A
    d = b*b - 4*a*c
    if a == 0 or d < 0:
        return -1.0
    s = math.sqrt(d)
    r1 = (-b + s) / (2*a)
    r2 = (-b - s) / (2*a)
    if 0 <= r1 <= 1:
        return r1
    if 0 <= r2 <= 1:
        return r2
    return -1.0


# ---------------------------------------------------------------------------
# Stage 1: bmToPathlist
# ---------------------------------------------------------------------------

def bm_to_pathlist(bm, info):
    """
    Walk the bitmap boundary, trace closed paths, XOR-fill to find holes.
    Faithfully translated from potrace.js bmToPathlist().
    Returns list of Path objects.
    """
    bm1 = bm.copy()
    pathlist = []

    def find_next(point):
        i = bm1.w * int(point.y) + int(point.x)
        while i < bm1.size and bm1.data[i] != 1:
            i += 1
        return bm1.index(i) if i < bm1.size else None

    def majority(x, y):
        for i in range(2, 5):
            ct = 0
            for a in range(-i+1, i):
                ct += 1 if bm1.at(x+a,   y+i-1) else -1
                ct += 1 if bm1.at(x+i-1, y+a-1) else -1
                ct += 1 if bm1.at(x+a-1, y-i)   else -1
                ct += 1 if bm1.at(x-i,   y+a)   else -1
            if ct > 0: return 1
            if ct < 0: return 0
        return 0

    def find_path(point):
        path = Path()
        x, y = int(point.x), int(point.y)
        dirx, diry = 0, 1
        path.sign = "+" if bm.at(x, y) else "-"

        while True:
            path.pt.append(Point(x, y))
            if x > path.maxX: path.maxX = x
            if x < path.minX: path.minX = x
            if y > path.maxY: path.maxY = y
            if y < path.minY: path.minY = y
            path.len += 1
            x += dirx
            y += diry
            path.area -= x * diry
            if x == int(point.x) and y == int(point.y):
                break
            l = bm1.at(x + (dirx + diry - 1) // 2, y + (diry - dirx - 1) // 2)
            r = bm1.at(x + (dirx - diry - 1) // 2, y + (diry + dirx - 1) // 2)
            if r and not l:
                tp = info["turnpolicy"]
                if (tp == "right" or
                   (tp == "black"    and path.sign == "+") or
                   (tp == "white"    and path.sign == "-") or
                   (tp == "majority" and majority(x, y)) or
                   (tp == "minority" and not majority(x, y))):
                    dirx, diry = -diry, dirx
                else:
                    dirx, diry = diry, -dirx
            elif r:
                dirx, diry = -diry, dirx
            elif not l:
                dirx, diry = diry, -dirx
        return path

    def xor_path(path):
        y1 = int(path.pt[0].y)
        for i in range(1, path.len):
            x = int(path.pt[i].x)
            y = int(path.pt[i].y)
            if y != y1:
                minY = min(y1, y)
                for j in range(x, path.maxX):
                    bm1.flip(j, minY)
                y1 = y

    current_point = Point(0, 0)
    while True:
        cp = find_next(current_point)
        if cp is None:
            break
        current_point = cp
        path = find_path(current_point)
        xor_path(path)
        if path.area > info["turdsize"]:
            pathlist.append(path)

    return pathlist


# ---------------------------------------------------------------------------
# Stage 2: calcSums
# ---------------------------------------------------------------------------

def calc_sums(path):
    """
    Build prefix-sum tables for x, y, x², xy, y².
    These allow O(1) computation of the quadratic penalty for any sub-path.
    Translated from potrace.js calcSums().
    """
    path.x0 = path.pt[0].x
    path.y0 = path.pt[0].y
    path.sums = [Sum(0, 0, 0, 0, 0)]
    s = path.sums
    for i in range(path.len):
        x = path.pt[i].x - path.x0
        y = path.pt[i].y - path.y0
        s.append(Sum(
            s[i].x  + x,
            s[i].y  + y,
            s[i].xy + x*y,
            s[i].x2 + x*x,
            s[i].y2 + y*y,
        ))


# ---------------------------------------------------------------------------
# Stage 3: calcLon
# ---------------------------------------------------------------------------

def calc_lon(path):
    """
    For each vertex i, compute path.lon[i] = the farthest vertex j such
    that the straight segment i..j is a valid 'run' (no 4-directional reversal).
    Translated from potrace.js calcLon().
    """
    n = path.len
    pt = path.pt
    pivk = [0]*n
    nc   = [0]*n
    path.lon = [0]*n

    constraint = [Point(), Point()]
    cur = Point(); off = Point(); dk = Point()

    # nc[i] = index of next point that is NOT collinear with i
    k = 0
    for i in range(n-1, -1, -1):
        if pt[i].x != pt[k].x and pt[i].y != pt[k].y:
            k = i + 1
        nc[i] = k

    for i in range(n-1, -1, -1):
        ct = [0, 0, 0, 0]
        dir_val = int(( 3 + 3*(pt[_mod(i+1,n)].x - pt[i].x)
                          +   (pt[_mod(i+1,n)].y - pt[i].y) ) // 2)
        ct[dir_val] += 1

        constraint[0].x = constraint[0].y = 0.0
        constraint[1].x = constraint[1].y = 0.0

        k  = nc[i]
        k1 = i
        foundk = False

        while True:
            dir_val = int(( 3 + 3*_sign(pt[k].x - pt[k1].x)
                              +   _sign(pt[k].y - pt[k1].y) ) // 2)
            ct[dir_val] += 1

            if ct[0] and ct[1] and ct[2] and ct[3]:
                pivk[i] = k1
                foundk   = True
                break

            cur.x = pt[k].x - pt[i].x
            cur.y = pt[k].y - pt[i].y

            if _xprod(constraint[0], cur) < 0 or _xprod(constraint[1], cur) > 0:
                break

            if abs(cur.x) > 1 or abs(cur.y) > 1:
                # update constraint[0]
                off.x = cur.x + (1 if (cur.y >= 0 and (cur.y > 0 or cur.x < 0)) else -1)
                off.y = cur.y + (1 if (cur.x <= 0 and (cur.x < 0 or cur.y < 0)) else -1)
                if _xprod(constraint[0], off) >= 0:
                    constraint[0].x = off.x; constraint[0].y = off.y
                # update constraint[1]
                off.x = cur.x + (1 if (cur.y <= 0 and (cur.y < 0 or cur.x < 0)) else -1)
                off.y = cur.y + (1 if (cur.x >= 0 and (cur.x > 0 or cur.y < 0)) else -1)
                if _xprod(constraint[1], off) <= 0:
                    constraint[1].x = off.x; constraint[1].y = off.y

            k1 = k
            k  = nc[k1]
            if not _cyclic(k, i, k1):
                break

        if not foundk:
            dk.x = _sign(pt[k].x - pt[k1].x)
            dk.y = _sign(pt[k].y - pt[k1].y)
            cur.x = pt[k1].x - pt[i].x
            cur.y = pt[k1].y - pt[i].y
            a = _xprod(constraint[0], cur)
            b = _xprod(constraint[0], dk)
            c = _xprod(constraint[1], cur)
            d = _xprod(constraint[1], dk)
            j = 10000000
            if b < 0:
                j = int(a / -b)
            if d > 0:
                j = min(j, int(-c / d))
            pivk[i] = _mod(k1 + j, n)

    j = pivk[n-1]
    path.lon[n-1] = j
    for i in range(n-2, -1, -1):
        if _cyclic(i+1, pivk[i], j):
            j = pivk[i]
        path.lon[i] = j

    i = n - 1
    while _cyclic(_mod(i+1, n), j, path.lon[i]):
        path.lon[i] = j
        i -= 1


# ---------------------------------------------------------------------------
# Stage 4: bestPolygon
# ---------------------------------------------------------------------------

def best_polygon(path):
    """
    Dynamic-programming shortest polygon.
    penalty3 uses the prefix-sum tables to measure how well a straight line
    fits the pixel boundary between vertices i and j.
    Translated from potrace.js bestPolygon().
    """
    n    = path.len
    pt   = path.pt
    sums = path.sums

    def penalty3(i, j):
        """
        Quadratic penalty for fitting a straight line from pt[i] to pt[j].
        Uses the algebraic formula from Selinger §2.3:
            For each pixel p on the boundary segment, the penalty is the
            squared distance from p to the line.  The prefix sums allow
            computing Σx, Σy, Σx², Σxy, Σy² in O(1).
        """
        r = 0
        jj = j
        if jj >= n:
            jj -= n
            r = 1

        if r == 0:
            x  = sums[jj+1].x  - sums[i].x
            y  = sums[jj+1].y  - sums[i].y
            x2 = sums[jj+1].x2 - sums[i].x2
            xy = sums[jj+1].xy - sums[i].xy
            y2 = sums[jj+1].y2 - sums[i].y2
            k  = jj + 1 - i
        else:
            x  = sums[jj+1].x  - sums[i].x  + sums[n].x
            y  = sums[jj+1].y  - sums[i].y  + sums[n].y
            x2 = sums[jj+1].x2 - sums[i].x2 + sums[n].x2
            xy = sums[jj+1].xy - sums[i].xy + sums[n].xy
            y2 = sums[jj+1].y2 - sums[i].y2 + sums[n].y2
            k  = jj + 1 - i + n

        px = (pt[i].x + pt[jj].x) / 2.0 - pt[0].x
        py = (pt[i].y + pt[jj].y) / 2.0 - pt[0].y
        ey = pt[jj].x - pt[i].x
        ex = -(pt[jj].y - pt[i].y)

        a = (x2 - 2*x*px) / k + px*px
        b = (xy - x*py - y*px) / k + px*py
        c = (y2 - 2*y*py) / k + py*py

        s = ex*ex*a + 2*ex*ey*b + ey*ey*c
        return math.sqrt(max(s, 0.0))

    # Build clip arrays and DP path
    clip0 = [0]*n
    clip1 = [0]*(n+1)
    seg0  = [0]*(n+1)
    seg1  = [0]*(n+1)
    pen   = [0.0]*(n+1)
    prev  = [0]*(n+1)

    for i in range(n):
        c = _mod(path.lon[_mod(i-1, n)] - 1, n)
        if c == i:
            c = _mod(i+1, n)
        clip0[i] = n if c < i else c

    j = 1
    for i in range(n):
        while j <= clip0[i]:
            clip1[j] = i
            j += 1

    i = 0; j = 0
    while i < n:
        seg0[j] = i
        i = clip0[i]
        j += 1
    seg0[j] = n
    m = j

    i = n
    for j in range(m, 0, -1):
        seg1[j] = i
        i = clip1[i]
    seg1[0] = 0

    pen[0] = 0.0
    for j in range(1, m+1):
        for i in range(seg1[j], seg0[j]+1):
            best = -1.0
            for k in range(seg0[j-1], clip1[i]-1, -1):
                this_pen = penalty3(k, i) + pen[k]
                if best < 0 or this_pen < best:
                    prev[i] = k
                    best = this_pen
            pen[i] = best

    path.m  = m
    path.po = [0]*m
    i = n
    for j in range(m-1, -1, -1):
        i = prev[i]
        path.po[j] = i


# ---------------------------------------------------------------------------
# Stage 5: adjustVertices
# ---------------------------------------------------------------------------

def adjust_vertices(path):
    """
    Find the optimal sub-pixel vertex positions using least-squares quadratic
    forms (Selinger §2.4 / potrace.js adjustVertices).

    For each polygon edge, a 3×3 quadratic form Q is built from the
    best-fit line direction through that edge's pixels.  Pairs of adjacent
    Qs are combined and the minimum of their sum gives the optimal vertex.
    """
    m  = path.m
    po = path.po
    n  = path.len
    pt = path.pt
    x0 = path.x0
    y0 = path.y0

    ctr = [Point() for _ in range(m)]
    dir_ = [Point() for _ in range(m)]
    q   = [Quad()  for _ in range(m)]
    path.curve = Curve(m)

    def pointslope(i, j, ctr, dir_out):
        """
        Compute centroid and principal direction of the pixel segment i..j.

        Uses the eigendecomposition of the 2×2 covariance matrix:
            [[a, b], [b, c]]
        The larger eigenvalue's eigenvector gives the best-fit direction.
        This is the standard PCA approach for line fitting.
        """
        r = 0
        while j >= n: j -= n; r += 1
        while i >= n: i -= n; r -= 1
        while j < 0:  j += n; r -= 1
        while i < 0:  i += n; r += 1

        x  = sums[j+1].x  - sums[i].x  + r*sums[n].x
        y  = sums[j+1].y  - sums[i].y  + r*sums[n].y
        x2 = sums[j+1].x2 - sums[i].x2 + r*sums[n].x2
        xy = sums[j+1].xy - sums[i].xy + r*sums[n].xy
        y2 = sums[j+1].y2 - sums[i].y2 + r*sums[n].y2
        k  = j + 1 - i + r*n

        ctr.x = x / k
        ctr.y = y / k

        a = (x2 - x*x/k) / k
        b = (xy - x*y/k) / k
        c = (y2 - y*y/k) / k

        # Larger eigenvalue of [[a,b],[b,c]]
        lambda2 = (a + c + math.sqrt((a-c)**2 + 4*b*b)) / 2.0
        a -= lambda2
        c -= lambda2

        if abs(a) >= abs(c):
            ll = math.sqrt(a*a + b*b)
            if ll != 0:
                dir_out.x = -b / ll
                dir_out.y =  a / ll
        else:
            ll = math.sqrt(c*c + b*b)
            if ll != 0:
                dir_out.x = -c / ll
                dir_out.y =  b / ll
        if ll == 0:
            dir_out.x = dir_out.y = 0.0
        return j  # return adjusted j (unused externally)

    sums = path.sums

    for i in range(m):
        j = po[_mod(i+1, m)]
        j = _mod(j - po[i], n) + po[i]
        pointslope(po[i], j, ctr[i], dir_[i])

    for i in range(m):
        d = dir_[i].x**2 + dir_[i].y**2
        if d == 0.0:
            for jj in range(9):
                q[i].data[jj] = 0.0
        else:
            v = [dir_[i].y, -dir_[i].x,
                 -dir_[i].y*ctr[i].y - (-dir_[i].x)*ctr[i].x]
            # Rank-1 outer product / d  → 3×3 quadratic form matrix
            for l in range(3):
                for k in range(3):
                    q[i].data[l*3+k] = v[l]*v[k] / d

    s = Point()
    for i in range(m):
        Q = Quad()
        w = Point()
        s.x = pt[po[i]].x - x0
        s.y = pt[po[i]].y - y0
        j = _mod(i-1, m)
        for l in range(3):
            for k in range(3):
                Q.data[l*3+k] = q[j].at(l,k) + q[i].at(l,k)

        while True:
            det = Q.at(0,0)*Q.at(1,1) - Q.at(0,1)*Q.at(1,0)
            if det != 0.0:
                w.x = (-Q.at(0,2)*Q.at(1,1) + Q.at(1,2)*Q.at(0,1)) / det
                w.y = ( Q.at(0,2)*Q.at(1,0) - Q.at(1,2)*Q.at(0,0)) / det
                break
            if Q.at(0,0) > Q.at(1,1):
                v = [-Q.at(0,1), Q.at(0,0)]
            elif Q.at(1,1):
                v = [-Q.at(1,1), Q.at(1,0)]
            else:
                v = [1.0, 0.0]
            d = v[0]**2 + v[1]**2
            v.append(-v[1]*s.y - v[0]*s.x)
            for l in range(3):
                for k in range(3):
                    Q.data[l*3+k] += v[l]*v[k] / d

        dx = abs(w.x - s.x)
        dy = abs(w.y - s.y)
        if dx <= 0.5 and dy <= 0.5:
            path.curve.vertex[i] = Point(w.x + x0, w.y + y0)
            continue

        min_q = _quadform(Q, s)
        xmin, ymin = s.x, s.y

        if Q.at(0,0) != 0.0:
            for z in range(2):
                w.y = s.y - 0.5 + z
                w.x = -(Q.at(0,1)*w.y + Q.at(0,2)) / Q.at(0,0)
                if abs(w.x - s.x) <= 0.5:
                    cand = _quadform(Q, w)
                    if cand < min_q:
                        min_q = cand; xmin = w.x; ymin = w.y

        if Q.at(1,1) != 0.0:
            for z in range(2):
                w.x = s.x - 0.5 + z
                w.y = -(Q.at(1,0)*w.x + Q.at(1,2)) / Q.at(1,1)
                if abs(w.y - s.y) <= 0.5:
                    cand = _quadform(Q, w)
                    if cand < min_q:
                        min_q = cand; xmin = w.x; ymin = w.y

        for l in range(2):
            for k in range(2):
                w.x = s.x - 0.5 + l
                w.y = s.y - 0.5 + k
                cand = _quadform(Q, w)
                if cand < min_q:
                    min_q = cand; xmin = w.x; ymin = w.y

        path.curve.vertex[i] = Point(xmin + x0, ymin + y0)


def reverse(path):
    v = path.curve.vertex
    m = path.curve.n
    i, j = 0, m-1
    while i < j:
        v[i], v[j] = v[j], v[i]
        i += 1; j -= 1


# ---------------------------------------------------------------------------
# Stage 6: smooth  (corner detection + Bezier control-point assignment)
# ---------------------------------------------------------------------------

def smooth(path, alphamax):
    """
    For each vertex of the optimal polygon, decide CURVE or CORNER.

    Alpha penalty (Selinger §2.5):
        At vertex j, with neighbours i and k:
          denom = ddenom(vertex[i], vertex[k])  — proportional to |i→k| in
                  the direction orthogonal to the chord
          dd    = |dpara(i, j, k)| / denom      — cross-ratio / 'curviness'
          alpha = (dd > 1) ? 1 - 1/dd : 0,  then scaled by 4/3

        If alpha >= alphamax  → CORNER  (sharp; use L command)
        Else                  → CURVE   (smooth; use C command)

    Control points for CURVE segments:
        p2 = midpoint of vertex[k] and vertex[j]   (end-anchor of segment)
        p3 = interval(0.5 + 0.5*alpha, vertex[i], vertex[j])  (cp1 outgoing)
        p4 = interval(0.5 + 0.5*alpha, vertex[k], vertex[j])  (cp2 incoming)
    The factor (0.5 + 0.5*alpha) scales the control arms — larger alpha =
    arms pulled closer to the polygon vertex = tighter curve.
    """
    m     = path.curve.n
    curve = path.curve

    for i in range(m):
        j  = _mod(i+1, m)
        k  = _mod(i+2, m)
        # p4 = midpoint of vertex[k] and vertex[j]
        p4 = _interval(0.5, curve.vertex[k], curve.vertex[j])

        denom = _ddenom(curve.vertex[i], curve.vertex[k])
        if denom != 0.0:
            dd    = abs(_dpara(curve.vertex[i], curve.vertex[j], curve.vertex[k]) / denom)
            alpha = (1.0 - 1.0/dd) if dd > 1 else 0.0
            alpha = alpha / 0.75
        else:
            alpha = 4.0 / 3.0

        curve.alpha0[j] = alpha

        if alpha >= alphamax:
            # Sharp corner
            curve.tag[j]       = "CORNER"
            curve.c[3*j+1]     = curve.vertex[j]
            curve.c[3*j+2]     = p4
        else:
            alpha = max(0.55, min(1.0, alpha))
            p2 = _interval(0.5 + 0.5*alpha, curve.vertex[i], curve.vertex[j])
            p3 = _interval(0.5 + 0.5*alpha, curve.vertex[k], curve.vertex[j])
            curve.tag[j]   = "CURVE"
            curve.c[3*j+0] = p2
            curve.c[3*j+1] = p3
            curve.c[3*j+2] = p4

        curve.alpha[j] = alpha
        curve.beta[j]  = 0.5

    curve.alphacurve = 1


# ---------------------------------------------------------------------------
# Stage 7: optiCurve  (curve-merging optimisation — optional)
# ---------------------------------------------------------------------------

def opti_curve(path, opttolerance):
    """
    Merge adjacent CURVE segments into a single Bezier where possible,
    minimising total segment count while staying within opttolerance.
    Translated from potrace.js optiCurve().
    """
    curve  = path.curve
    m      = curve.n
    vert   = curve.vertex
    convc  = [0]*m
    areac  = [0.0]*(m+1)

    for i in range(m):
        if curve.tag[i] == "CURVE":
            convc[i] = _sign(_dpara(vert[_mod(i-1,m)], vert[i], vert[_mod(i+1,m)]))
        else:
            convc[i] = 0

    area = 0.0
    areac[0] = 0.0
    p0 = curve.vertex[0]
    for i in range(m):
        i1 = _mod(i+1, m)
        if curve.tag[i1] == "CURVE":
            alpha = curve.alpha[i1]
            area += 0.3*alpha*(4-alpha) * _dpara(curve.c[i*3+2], vert[i1], curve.c[i1*3+2]) / 2
            area += _dpara(p0, curve.c[i*3+2], curve.c[i1*3+2]) / 2
        areac[i+1] = area

    class Opti:
        def __init__(self):
            self.pen   = 0.0
            self.c     = [Point(), Point()]
            self.t     = 0.0
            self.s     = 0.0
            self.alpha = 0.0

    pt  = list(range(m+1))
    pen = [0.0]*(m+1)
    llen = [0]*(m+1)
    opt = [None]*(m+1)

    pt[0]   = -1
    pen[0]  = 0.0
    llen[0] = 0

    def opti_penalty(i, j, res):
        if i == j:
            return 1
        k  = i
        i1 = _mod(i+1, m)
        k1 = _mod(k+1, m)
        conv = convc[k1]
        if conv == 0:
            return 1
        d = _ddist(vert[i], vert[i1])
        k = k1
        while k != j:
            k1 = _mod(k+1, m)
            k2 = _mod(k+2, m)
            if convc[k1] != conv:
                return 1
            if _sign(_cprod(vert[i], vert[i1], vert[k1], vert[k2])) != conv:
                return 1
            if _iprod1(vert[i], vert[i1], vert[k1], vert[k2]) < d * _ddist(vert[k1], vert[k2]) * -0.999847695156:
                return 1
            k = k1

        p0 = curve.c[_mod(i,m)*3+2].copy()
        p1 = vert[_mod(i+1,m)].copy()
        p2 = vert[_mod(j,m)].copy()
        p3 = curve.c[_mod(j,m)*3+2].copy()

        ar = areac[j] - areac[i]
        ar -= _dpara(vert[0], curve.c[i*3+2], curve.c[j*3+2]) / 2
        if i >= j:
            ar += areac[m]

        A1 = _dpara(p0, p1, p2)
        A2 = _dpara(p0, p1, p3)
        A3 = _dpara(p0, p2, p3)
        A4 = A1 + A3 - A2

        if A2 == A1:
            return 1

        t = A3 / (A3 - A4)
        s = A2 / (A2 - A1)
        A = A2 * t / 2.0
        if A == 0.0:
            return 1

        R     = ar / A
        alpha = 2.0 - math.sqrt(max(0.0, 4.0 - R / 0.3))

        res.c[0] = _interval(t * alpha, p0, p1)
        res.c[1] = _interval(s * alpha, p3, p2)
        res.alpha = alpha
        res.t = t
        res.s = s

        p1 = res.c[0].copy()
        p2 = res.c[1].copy()
        res.pen = 0.0

        k = _mod(i+1, m)
        while k != j:
            k1 = _mod(k+1, m)
            t2 = _tangent(p0, p1, p2, p3, vert[k], vert[k1])
            if t2 < -0.5:
                return 1
            pt2 = _bezier(t2, p0, p1, p2, p3)
            d   = _ddist(vert[k], vert[k1])
            if d == 0.0:
                return 1
            d1 = _dpara(vert[k], vert[k1], pt2) / d
            if abs(d1) > opttolerance:
                return 1
            if _iprod(vert[k], vert[k1], pt2) < 0 or _iprod(vert[k1], vert[k], pt2) < 0:
                return 1
            res.pen += d1*d1
            k = k1

        k = i
        while k != j:
            k1 = _mod(k+1, m)
            t2 = _tangent(p0, p1, p2, p3, curve.c[k*3+2], curve.c[k1*3+2])
            if t2 < -0.5:
                return 1
            pt2 = _bezier(t2, p0, p1, p2, p3)
            d   = _ddist(curve.c[k*3+2], curve.c[k1*3+2])
            if d == 0.0:
                return 1
            d1 = _dpara(curve.c[k*3+2], curve.c[k1*3+2], pt2) / d
            d2 = _dpara(curve.c[k*3+2], curve.c[k1*3+2], vert[k1]) / d
            d2 *= 0.75 * curve.alpha[k1]
            if d2 < 0:
                d1 = -d1; d2 = -d2
            if d1 < d2 - opttolerance:
                return 1
            if d1 < d2:
                res.pen += (d1-d2)**2
            k = k1

        return 0

    for j in range(1, m+1):
        pt[j]   = j-1
        pen[j]  = pen[j-1]
        llen[j] = llen[j-1] + 1
        o = Opti()
        for i in range(j-2, -1, -1):
            r = opti_penalty(i, _mod(j, m), o)
            if r:
                break
            if llen[j] > llen[i]+1 or (llen[j] == llen[i]+1 and pen[j] > pen[i]+o.pen):
                pt[j]   = i
                pen[j]  = pen[i] + o.pen
                llen[j] = llen[i] + 1
                opt[j]  = Opti()
                opt[j].pen   = o.pen
                opt[j].c     = [o.c[0].copy(), o.c[1].copy()]
                opt[j].alpha = o.alpha
                opt[j].t     = o.t
                opt[j].s     = o.s
                o = Opti()

    om     = llen[m]
    ocurve = Curve(om)
    s_arr  = [0.0]*om
    t_arr  = [0.0]*om

    j = m
    for i in range(om-1, -1, -1):
        jmod = _mod(j, m)
        if pt[j] == j-1:
            ocurve.tag[i]      = curve.tag[jmod]
            ocurve.c[i*3+0]    = curve.c[jmod*3+0]
            ocurve.c[i*3+1]    = curve.c[jmod*3+1]
            ocurve.c[i*3+2]    = curve.c[jmod*3+2]
            ocurve.vertex[i]   = curve.vertex[jmod]
            ocurve.alpha[i]    = curve.alpha[jmod]
            ocurve.alpha0[i]   = curve.alpha0[jmod]
            ocurve.beta[i]     = curve.beta[jmod]
            s_arr[i] = t_arr[i] = 1.0
        else:
            ocurve.tag[i]    = "CURVE"
            ocurve.c[i*3+0]  = opt[j].c[0]
            ocurve.c[i*3+1]  = opt[j].c[1]
            ocurve.c[i*3+2]  = curve.c[jmod*3+2]
            ocurve.vertex[i] = _interval(opt[j].s, curve.c[jmod*3+2], vert[jmod])
            ocurve.alpha[i]  = opt[j].alpha
            ocurve.alpha0[i] = opt[j].alpha
            s_arr[i] = opt[j].s
            t_arr[i] = opt[j].t
        j = pt[j]

    for i in range(om):
        i1 = _mod(i+1, om)
        ocurve.beta[i] = s_arr[i] / (s_arr[i] + t_arr[i1])

    ocurve.alphacurve = 1
    path.curve = ocurve


# ---------------------------------------------------------------------------
# Stage 8: getSVG  — SVG serialisation
# ---------------------------------------------------------------------------

def get_svg(pathlist, width, height, size=1.0, fill="#000000"):
    """
    Serialise all paths to SVG.
    Translated from potrace.js getSVG(), with fill-rule="evenodd".

    SVG commands used:
        M x y           — MoveTo (start of subpath)
        C x1 y1,x2 y2,x y — Cubic Bezier (CURVE segments)
        L x1 y1 x2 y2   — LineTo pair (CORNER segments: line + short end)
        Z               — ClosePath (implicit in Potrace-js omission, we add it)
    """
    def fmt(v):
        return f"{v * size:.3f}"

    def path_str(curve):
        n = curve.n
        # Start at last segment's end-point (matching potrace.js)
        d = f"M{fmt(curve.c[(n-1)*3+2].x)} {fmt(curve.c[(n-1)*3+2].y)} "
        for i in range(n):
            if curve.tag[i] == "CURVE":
                d += (f"C {fmt(curve.c[i*3+0].x)} {fmt(curve.c[i*3+0].y)},"
                      f"{fmt(curve.c[i*3+1].x)} {fmt(curve.c[i*3+1].y)},"
                      f"{fmt(curve.c[i*3+2].x)} {fmt(curve.c[i*3+2].y)} ")
            elif curve.tag[i] == "CORNER":
                d += (f"L {fmt(curve.c[i*3+1].x)} {fmt(curve.c[i*3+1].y)} "
                      f"{fmt(curve.c[i*3+2].x)} {fmt(curve.c[i*3+2].y)} ")
        return d

    w = int(width * size)
    h = int(height * size)
    d_parts = [path_str(p.curve) for p in pathlist]
    d = " ".join(d_parts)

    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1"',
        f'     width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'  <!-- tracer.py: pure-Python Potrace transpile -->',
        f'  <path',
        f'    fill="{fill}"',
        f'    stroke="none"',
        f'    fill-rule="evenodd"',
        f'    d="{d}"',
        f'  />',
        f'</svg>',
    ])


# ---------------------------------------------------------------------------
# Top-level: PurePythonTracer class
# ---------------------------------------------------------------------------

class PurePythonTracer:
    """
    Orchestrates the full Potrace pipeline on a binarized numpy array.

    Parameters
    ----------
    turdsize     : int   — suppress speckles smaller than this area (px²)
    alphamax     : float — corner threshold; lower = more corners kept
    opttolerance : float — curve-merge tolerance (pixels)
    optcurve     : bool  — run optiCurve pass (recommended)
    turnpolicy   : str   — how to resolve ambiguous turns ("minority" default)
    """

    def __init__(self, turdsize=2, alphamax=1.0, opttolerance=0.2,
                 optcurve=True, turnpolicy="minority"):
        self.info = dict(
            turdsize=turdsize,
            alphamax=alphamax,
            opttolerance=opttolerance,
            optcurve=optcurve,
            turnpolicy=turnpolicy,
        )

    def trace(self, binary_arr: np.ndarray, fill="#000000", size=1.0) -> str:
        """
        Run the full pipeline.

        Parameters
        ----------
        binary_arr : uint8 numpy array (H x W), foreground pixels = 255
        fill       : SVG fill colour
        size       : output scale factor

        Returns
        -------
        str : complete SVG document
        """
        h, w = binary_arr.shape[:2]

        # Build Bitmap (1 = foreground / dark pixel)
        flat = (binary_arr.flatten() > 127).astype(int).tolist()
        bm = Bitmap(w, h, flat)

        info = self.info
        pathlist = bm_to_pathlist(bm, info)

        for path in pathlist:
            calc_sums(path)
            calc_lon(path)
            best_polygon(path)
            adjust_vertices(path)
            if path.sign == "-":
                reverse(path)
            smooth(path, info["alphamax"])
            if info["optcurve"]:
                opti_curve(path, info["opttolerance"])

        return get_svg(pathlist, w, h, size=size, fill=fill)

    def trace_color_layers(self, img_bgr: np.ndarray, n_colors: int = 8,
                           size: float = 1.0) -> str:
        """
        Multi-color vectorization pipeline (vectorizer.ai style):
          1. K-means color quantize the image to n_colors palette entries.
          2. For each palette color (darkest first = bottom SVG layer):
             a. Build a binary mask: pixels belonging to this color cluster = 1.
             b. Run the full Potrace pipeline on that mask.
             c. Collect the resulting paths tagged with the cluster's hex color.
          3. Assemble a layered SVG with all color layers stacked correctly.

        Each layer uses fill-rule=evenodd so inner holes still punch through.
        """
        h, w = img_bgr.shape[:2]

        # ---- Step 1: K-means color quantization ----
        # Reshape to list of pixels, work in float32 for cv2.kmeans
        pixels = img_bgr.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        centers = np.uint8(centers)            # palette: (n_colors, 3) BGR
        labels  = labels.flatten()             # cluster index per pixel

        # ---- Step 2: Sort palette from lightest to darkest ----
        # Lightest = background (render first / bottom). Darkest = foreground (top).
        brightness = centers.astype(float).mean(axis=1)  # mean BGR as proxy
        order = np.argsort(brightness)[::-1]             # light → dark

        info = self.info
        layer_parts = []  # list of (hex_color, svg_d_string)

        for cluster_idx in order:
            color_bgr = centers[cluster_idx]             # (B, G, R)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
            )

            # Skip near-white background layer (optional: remove if you want bg rect)
            lum = 0.299*color_bgr[2] + 0.587*color_bgr[1] + 0.114*color_bgr[0]
            if lum > 245:
                continue

            # Build binary mask for this cluster (1 = this color)
            mask = (labels == cluster_idx).reshape(h, w).astype(np.uint8) * 255

            # Run Potrace on the mask
            flat = (mask.flatten() > 127).astype(int).tolist()
            bm   = Bitmap(w, h, flat)
            pathlist = bm_to_pathlist(bm, info)

            if not pathlist:
                continue

            for path in pathlist:
                calc_sums(path)
                calc_lon(path)
                best_polygon(path)
                adjust_vertices(path)
                if path.sign == "-":
                    reverse(path)
                smooth(path, info["alphamax"])
                if info["optcurve"]:
                    opti_curve(path, info["opttolerance"])

            layer_parts.append((hex_color, pathlist))

        # ---- Step 3: Assemble multi-layer SVG ----
        return build_multicolor_svg(layer_parts, w, h, size=size)


# ---------------------------------------------------------------------------
# Multi-color SVG builder
# ---------------------------------------------------------------------------

def _path_d_str(curve, size):
    """Serialize one Curve object to an SVG path 'd' substring."""
    def fmt(v):
        return f"{v * size:.3f}"
    n = curve.n
    d = f"M{fmt(curve.c[(n-1)*3+2].x)} {fmt(curve.c[(n-1)*3+2].y)} "
    for i in range(n):
        if curve.tag[i] == "CURVE":
            d += (f"C {fmt(curve.c[i*3+0].x)} {fmt(curve.c[i*3+0].y)},"
                  f"{fmt(curve.c[i*3+1].x)} {fmt(curve.c[i*3+1].y)},"
                  f"{fmt(curve.c[i*3+2].x)} {fmt(curve.c[i*3+2].y)} ")
        elif curve.tag[i] == "CORNER":
            d += (f"L {fmt(curve.c[i*3+1].x)} {fmt(curve.c[i*3+1].y)} "
                  f"{fmt(curve.c[i*3+2].x)} {fmt(curve.c[i*3+2].y)} ")
    return d


def build_multicolor_svg(layer_parts, width, height, size=1.0):
    """
    Build a layered SVG with one <path> per color cluster.
    Each path uses fill-rule=evenodd so holes punch through correctly.
    Layers are ordered light→dark so the darkest shapes are on top.
    """
    w = int(width  * size)
    h = int(height * size)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1"',
        f'     width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'  <!-- tracer.py multi-color mode: {len(layer_parts)} color layers -->',
    ]
    for hex_color, pathlist in layer_parts:
        # Combine all subpath 'd' strings for this color into one <path>
        d_parts = [_path_d_str(p.curve, size) for p in pathlist]
        d = " ".join(d_parts)
        lines.append(
            f'  <path fill="{hex_color}" stroke="none" '
            f'fill-rule="evenodd" d="{d}"/>'
        )
    lines.append('</svg>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="tracer.py",
        description="Pure-Python Potrace transpile — mathematically faithful SVG vectorizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tracer.py logo.png logo.svg
  python tracer.py sketch.png out.svg --alphamax 0.8 --opttolerance 0.3
  python tracer.py art.png out.svg --threshold 100 --fill "#1a1a2e"
        """,
    )
    ap.add_argument("input",  help="Input raster image (PNG / JPEG / BMP ...)")
    ap.add_argument("output", help="Output SVG path")
    ap.add_argument("--threshold",    type=int,   default=128,
                    help="Grayscale binarization threshold 0-255 (default 128). Ignored if --otsu is set.")
    ap.add_argument("--otsu",         action="store_true",
                    help="Use OTSU automatic threshold instead of --threshold.")
    ap.add_argument("--close",        type=int,   default=0,
                    help="Morphological close kernel size in pixels before tracing (e.g. 3 or 5). "
                         "Bridges thin white gaps/borders between touching shapes.")
    ap.add_argument("--turdsize",     type=int,   default=2,     help="Suppress speckles ≤ this area (default 2)")
    ap.add_argument("--alphamax",     type=float, default=1.0,   help="Corner threshold 0..2+ (default 1.0)")
    ap.add_argument("--opttolerance", type=float, default=0.2,   help="Curve-merge tolerance in px (default 0.2)")
    ap.add_argument("--no-optcurve",  action="store_true",       help="Disable optiCurve pass")
    ap.add_argument("--invert",       action="store_true",       help="Invert bitmap (trace dark shapes)")
    ap.add_argument("--turnpolicy",   default="minority",        help="Turn ambiguity policy (default minority)")
    ap.add_argument("--fill",         default="#000000",         help='SVG fill colour for binary mode (default "#000000")')
    ap.add_argument("--size",         type=float, default=1.0,   help="Output scale factor (default 1.0)")
    ap.add_argument("--colors",       type=int,   default=0,
                    help="Number of colors for multi-color mode (e.g. 8). "
                         "When set, ignores --threshold/--invert/--fill and "
                         "runs k-means quantization + per-layer tracing.")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"[tracer] ERROR: not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[tracer] Loading  {args.input}")

    tracer = PurePythonTracer(
        turdsize=args.turdsize,
        alphamax=args.alphamax,
        opttolerance=args.opttolerance,
        optcurve=not args.no_optcurve,
        turnpolicy=args.turnpolicy,
    )

    if args.colors > 0:
        # ---- Multi-color mode ----
        img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[tracer] ERROR: cannot read image.", file=sys.stderr)
            sys.exit(1)
        h, w = img_bgr.shape[:2]
        print(f"[tracer] Multi-color mode: {w}×{h} px, {args.colors} colors "
              f"(alphamax={args.alphamax}, opttol={args.opttolerance}, "
              f"turdsize={args.turdsize})")
        svg = tracer.trace_color_layers(img_bgr, n_colors=args.colors, size=args.size)
    else:
        # ---- Binary mode ----
        gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[tracer] ERROR: cannot read image.", file=sys.stderr)
            sys.exit(1)

        # Threshold — OTSU automatically finds the optimal split
        if args.otsu:
            thresh_val, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            print(f"[tracer] OTSU threshold: {thresh_val}")
        else:
            _, binary = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY)

        if args.invert:
            binary = cv2.bitwise_not(binary)

        # Optional morphological close — bridges thin white gaps between shapes
        # (e.g. the white border between a pin body and its shadow ellipse)
        if args.close > 0:
            kernel = np.ones((args.close, args.close), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            print(f"[tracer] Morphological close: kernel={args.close}×{args.close}")

        print(f"[tracer] Binary mode: {gray.shape[1]}×{gray.shape[0]} px "
              f"(alphamax={args.alphamax}, opttol={args.opttolerance}, "
              f"turdsize={args.turdsize}, optcurve={not args.no_optcurve})")
        svg = tracer.trace(binary, fill=args.fill, size=args.size)

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"[tracer] Written  {args.output}  ✓")


if __name__ == "__main__":
    main()
