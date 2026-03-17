"""SVG builder with proper winding and hole support.

Accepts a list of ContourData objects (outer contours + holes) and produces
a single clean <svg> file with all paths merged into one compound path using
fill-rule="evenodd" so holes are correctly rendered as transparent cut-outs.

Outer contours are wound CCW; hole contours are wound CW — this is enforced
automatically inside contour_to_svg_path().
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from yd_vector.bezier import contour_to_svg_path
from yd_vector.models import ContourData
from yd_vector.utils import ensure_parent_dir


def build_svg(
    contours: List[ContourData],
    width: int,
    height: int,
    fill: str = "#000000",
    corner_threshold: float = 0.85,
    bezier_min_points: int = 3,
) -> str:
    """Build an SVG string from a list of outer and hole contours.

    All contours are merged into a single <path> element with
    fill-rule="evenodd" so that nested holes are cut out correctly.

    Args:
        contours:          Outer and hole ContourData objects.
        width:             Canvas width in pixels.
        height:            Canvas height in pixels.
        fill:              SVG fill colour (default ``"#000000"``).
        corner_threshold:  Cosine threshold for corner detection (default 0.85).
        bezier_min_points: Min segment points to use C instead of L (default 3).

    Returns:
        SVG string.
    """
    path_parts: List[str] = []
    for contour in contours:
        d = contour_to_svg_path(
            contour.points,
            corner_threshold=corner_threshold,
            is_hole=contour.is_hole,
            bezier_min_points=bezier_min_points,
        )
        if d:
            path_parts.append(d)

    if not path_parts:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"></svg>'
        )

    compound = " ".join(path_parts)

    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <path\n'
        f'    fill="{fill}"\n'
        f'    stroke="none"\n'
        f'    fill-rule="evenodd"\n'
        f'    d="{compound}"\n'
        f'  />\n'
        f'</svg>'
    )


def write_svg(
    output_path: str | Path,
    contours: List[ContourData],
    width: int,
    height: int,
    fill: str = "#000000",
    corner_threshold: float = 0.85,
    bezier_min_points: int = 3,
) -> Path:
    """Write an SVG file from multiple contours.

    Args:
        output_path:       Destination file path (parent dirs created as needed).
        contours:          Outer and hole ContourData objects.
        width:             Canvas width in pixels.
        height:            Canvas height in pixels.
        fill:              SVG fill colour (default ``"#000000"``).
        corner_threshold:  Cosine threshold for corner detection (default 0.85).
        bezier_min_points: Min segment points to use C instead of L (default 3).

    Returns:
        Resolved Path to the written SVG file.
    """
    out = ensure_parent_dir(output_path)
    svg = build_svg(contours, width, height, fill=fill,
                    corner_threshold=corner_threshold,
                    bezier_min_points=bezier_min_points)
    Path(out).write_text(svg, encoding="utf-8")
    return Path(out)
