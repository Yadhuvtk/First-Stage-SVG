from __future__ import annotations

import math
from typing import List, Tuple

from yd_vector.models import ContourData, FittedPath, PathSegment, Point


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _angle(prev_pt: Point, curr_pt: Point, next_pt: Point) -> float:
    ax, ay = prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1]
    bx, by = next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]

    mag_a = math.hypot(ax, ay)
    mag_b = math.hypot(bx, by)
    if mag_a == 0 or mag_b == 0:
        return 180.0

    dot = ax * bx + ay * by
    cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_theta))


def detect_corners(points: List[Point], angle_threshold: float = 135.0) -> List[int]:
    n = len(points)
    if n < 3:
        return []

    corners: List[int] = []
    for i in range(n):
        prev_pt = points[(i - 1) % n]
        curr_pt = points[i]
        next_pt = points[(i + 1) % n]

        angle = _angle(prev_pt, curr_pt, next_pt)
        if angle < angle_threshold:
            corners.append(i)

    return corners


def _line_segment(start: Point, end: Point) -> PathSegment:
    return PathSegment(kind="line", start=start, end=end)


def fit_contour_as_lines(contour: ContourData) -> FittedPath:
    points = contour.points
    corners = detect_corners(points)

    if len(corners) < 2:
        segments = []
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            segments.append(_line_segment(start, end))
        return FittedPath(
            segments=segments,
            area=contour.area,
            parent_index=contour.parent_index,
            is_hole=contour.is_hole,
        )

    ordered = sorted(set(corners))
    segments: List[PathSegment] = []

    for i in range(len(ordered)):
        a = ordered[i]
        b = ordered[(i + 1) % len(ordered)]
        start = points[a]
        end = points[b]
        segments.append(_line_segment(start, end))

    return FittedPath(
        segments=segments,
        area=contour.area,
        parent_index=contour.parent_index,
        is_hole=contour.is_hole,
    )


def prepare_paths(contours: List[ContourData]) -> List[FittedPath]:
    return [fit_contour_as_lines(c) for c in contours]