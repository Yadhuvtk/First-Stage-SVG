from __future__ import annotations

from typing import List

import cv2
import numpy as np

from yd_vector.models import ContourData


def _touches_border(points: list[tuple[float, float]], width: int, height: int) -> bool:
    for x, y in points:
        if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1:
            return True
    return False


def extract_contours(binary: np.ndarray, min_area: float) -> List[ContourData]:
    height, width = binary.shape[:2]

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )

    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]
    results: List[ContourData] = []

    for idx, contour in enumerate(contours):
        area = abs(cv2.contourArea(contour))
        if area < min_area:
            continue

        points = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]

        if _touches_border(points, width, height):
            continue

        parent_index = int(hierarchy[idx][3])
        is_hole = parent_index != -1

        results.append(
            ContourData(
                points=points,
                area=float(area),
                parent_index=parent_index,
                is_hole=is_hole,
            )
        )

    results.sort(key=lambda c: (c.is_hole, -c.area))
    return results