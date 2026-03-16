from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

Point = Tuple[float, float]
SegmentType = Literal["line", "cubic"]


@dataclass
class TraceOptions:
    input_path: Path
    output_path: Path
    threshold: int = 128
    invert: bool = False
    simplify_tolerance: float = 1.5
    min_area: float = 10.0
    fill: str = "black"
    stroke: str = "none"
    debug: bool = False


@dataclass
class ContourData:
    points: List[Point]
    area: float
    parent_index: int
    is_hole: bool = False


@dataclass
class PathSegment:
    kind: SegmentType
    start: Point
    end: Point
    ctrl1: Optional[Point] = None
    ctrl2: Optional[Point] = None


@dataclass
class FittedPath:
    segments: List[PathSegment]
    area: float
    parent_index: int
    is_hole: bool = False


@dataclass
class TraceResult:
    width: int
    height: int
    contours: List[ContourData] = field(default_factory=list)
    fitted_paths: List[FittedPath] = field(default_factory=list)
    svg_path: Optional[Path] = None
    debug_mask_path: Optional[Path] = None