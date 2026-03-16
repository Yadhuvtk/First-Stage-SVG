from __future__ import annotations

from pathlib import Path
from typing import List

from yd_vector.models import ContourData
from yd_vector.utils import ensure_parent_dir


def _contour_to_path_d(contour: ContourData) -> str:
    if not contour.points:
        return ""

    first_x, first_y = contour.points[0]
    commands = [f"M {first_x:.2f} {first_y:.2f}"]

    for x, y in contour.points[1:]:
        commands.append(f"L {x:.2f} {y:.2f}")

    commands.append("Z")
    return " ".join(commands)


def write_svg(
    output_path: str | Path,
    width: int,
    height: int,
    contours: List[ContourData],
    fill: str = "black",
    stroke: str = "none",
) -> Path:
    output_path = ensure_parent_dir(output_path)

    path_strings = []
    for contour in contours:
        d = _contour_to_path_d(contour)
        if d:
            path_strings.append(d)

    compound_path = " ".join(path_strings)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path d="{compound_path}" fill="{fill}" stroke="{stroke}" fill-rule="evenodd"/>
</svg>
"""

    Path(output_path).write_text(svg, encoding="utf-8")
    return Path(output_path)