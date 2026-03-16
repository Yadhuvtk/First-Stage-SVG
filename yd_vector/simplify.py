from __future__ import annotations

from typing import List

import cv2
import numpy as np

from yd_vector.models import ContourData


def simplify_contour(contour: ContourData, tolerance: float) -> ContourData:
    pts = np.array(contour.points, dtype=np.float32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(pts, epsilon=tolerance, closed=True)
    simplified_points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]

    return ContourData(
        points=simplified_points,
        area=contour.area,
        parent_index=contour.parent_index,
        is_hole=contour.is_hole,
    )


def simplify_contours(contours: List[ContourData], tolerance: float) -> List[ContourData]:
    return [simplify_contour(contour, tolerance) for contour in contours]