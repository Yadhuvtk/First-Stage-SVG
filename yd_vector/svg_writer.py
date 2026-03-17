from __future__ import annotations

from pathlib import Path
from typing import List

from yd_vector.models import ContourData, FittedPath, PathSegment
from yd_vector.utils import ensure_parent_dir


def _contour_to_path_d(contour: ContourData) -> str:
    """Convert a ContourData directly to an SVG path data string (L commands only).

    This is a legacy helper retained for backward-compatible tests.
    For Bézier-quality output use yd_vector.bezier.contour_to_svg_path() instead.
    """
    points = contour.points
    if not points:
        return ""
    commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for pt in points[1:]:
        commands.append(f"L {pt[0]:.2f} {pt[1]:.2f}")
    commands.append("Z")
    return " ".join(commands)


def _segments_to_path_d(path: FittedPath) -> str:
    if not path.segments:
        return ""

    first = path.segments[0].start
    commands = [f"M {first[0]:.2f} {first[1]:.2f}"]

    for seg in path.segments:
        if seg.kind == "line":
            commands.append(f"L {seg.end[0]:.2f} {seg.end[1]:.2f}")
        elif seg.kind == "cubic":
            c1 = seg.ctrl1
            c2 = seg.ctrl2
            if c1 is None or c2 is None:
                commands.append(f"L {seg.end[0]:.2f} {seg.end[1]:.2f}")
            else:
                commands.append(
                    f"C {c1[0]:.2f} {c1[1]:.2f}, {c2[0]:.2f} {c2[1]:.2f}, {seg.end[0]:.2f} {seg.end[1]:.2f}"
                )

    commands.append("Z")
    return " ".join(commands)


def write_svg(
    output_path: str | Path,
    width: int,
    height: int,
    paths: List[FittedPath],
    fill: str = "black",
    stroke: str = "none",
) -> Path:
    output_path = ensure_parent_dir(output_path)

    d_parts = []
    for path in paths:
        d = _segments_to_path_d(path)
        if d:
            d_parts.append(d)

    compound_path = " ".join(d_parts)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path d="{compound_path}" fill="{fill}" stroke="{stroke}" fill-rule="evenodd"/>
</svg>
"""

    Path(output_path).write_text(svg, encoding="utf-8")
    return Path(output_path)