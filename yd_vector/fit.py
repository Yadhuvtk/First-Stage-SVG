from __future__ import annotations

from typing import List

from yd_vector.models import ContourData


def prepare_paths(contours: List[ContourData]) -> List[ContourData]:
    return contours