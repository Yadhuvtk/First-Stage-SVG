from __future__ import annotations

import argparse
from pathlib import Path

from yd_vector.config import load_config
from yd_vector.models import TraceOptions
from yd_vector.utils import clamp_int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YD-Vector CLI bitmap-to-SVG tracer")
    parser.add_argument("--input", required=True, help="Input raster image path")
    parser.add_argument("--output", required=True, help="Output SVG path")
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--threshold", type=int, default=None, help="Binary threshold 0-255")
    parser.add_argument("--invert", action="store_true", help="Invert threshold result")
    parser.add_argument("--simplify", type=float, default=None, help="Contour simplification tolerance")
    parser.add_argument("--min-area", type=float, default=None, help="Minimum contour area")
    parser.add_argument("--fill", default=None, help="SVG fill color")
    parser.add_argument("--stroke", default=None, help="SVG stroke color")
    parser.add_argument("--debug", action="store_true", help="Save debug mask")
    return parser


def parse_args() -> TraceOptions:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 128)
    simplify_tolerance = args.simplify if args.simplify is not None else cfg.get("simplify_tolerance", 1.5)
    min_area = args.min_area if args.min_area is not None else cfg.get("min_area", 10.0)
    fill = args.fill if args.fill is not None else cfg.get("fill", "black")
    stroke = args.stroke if args.stroke is not None else cfg.get("stroke", "none")

    invert = args.invert or bool(cfg.get("invert", False))
    debug = args.debug or bool(cfg.get("debug", False))

    return TraceOptions(
        input_path=Path(args.input),
        output_path=Path(args.output),
        threshold=clamp_int(threshold, 0, 255),
        invert=invert,
        simplify_tolerance=float(simplify_tolerance),
        min_area=float(min_area),
        fill=str(fill),
        stroke=str(stroke),
        debug=debug,
    )