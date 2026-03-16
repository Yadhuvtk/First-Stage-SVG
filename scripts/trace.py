#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

# Allow running the script directly from the repo root structure
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yd_vector.pipeline import TraceOptions, run_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YD-Vector: bitmap -> SVG tracer"
    )

    parser.add_argument("input", help="Input raster image path")
    parser.add_argument("output", help="Output SVG path")

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binary threshold value (0-255)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert foreground/background before tracing",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=1.2,
        help="Contour simplification tolerance",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Minimum contour area to keep",
    )
    parser.add_argument(
        "--fill",
        default="#000000",
        help="SVG fill color",
    )
    parser.add_argument(
        "--stroke",
        default="none",
        help="SVG stroke color",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    options = TraceOptions(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        invert=args.invert,
        simplify=args.simplify,
        min_area=args.min_area,
        fill=args.fill,
        stroke=args.stroke,
    )

    print(f"[yd-vector] Tracing: {args.input}")
    run_trace(options)
    print(f"[yd-vector] SVG written to: {args.output}")


if __name__ == "__main__":
    main()