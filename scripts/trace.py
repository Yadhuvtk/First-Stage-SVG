#!/usr/bin/env python3
"""
scripts/trace.py — CLI wrapper for PurePythonTracer (tracer.py)
================================================================
This script replaces the old yd_vector-based CLI.

The old yd_vector pipeline (cv2.findContours + Douglas-Peucker + line-only
SVG) has been superseded by the Potrace-style bitmap-walk pipeline in
tracer.py. This script now delegates entirely to PurePythonTracer so there
is one canonical tracing entry point.

Usage (unchanged from before, args extended):
  python scripts/trace.py input.png output.svg [options]

Potrace-style parameters:
  --threshold INT    Grayscale binarization threshold (default 128)
  --otsu             Use OTSU automatic threshold (overrides --threshold)
  --invert           Invert bitmap (trace dark shapes on light background)
  --close INT        Morphological close kernel size to bridge gaps (default 0)
  --turdsize INT     Suppress speckles smaller than this area px^2 (default 2)
  --alphamax FLOAT   Corner threshold: lower = more corners (default 1.0)
  --opttolerance F   Curve-merge tolerance in pixels (default 0.2)
  --no-optcurve      Disable the optional optiCurve merging pass
  --fill COLOR       SVG fill color hex string (default #000000)
  --bg COLOR         SVG background rect color (omitted = transparent)
  --scale FLOAT      Output scale factor (default 1.0)
  --debug            Dump intermediate pipeline state as JSON to stdout
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running directly from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import potrace

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/trace.py",
        description="Potrace-style bitmap→SVG tracer (pure Python, no C-bindings).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/trace.py inputs/1.jpg outputs/1.svg --otsu --invert
  python scripts/trace.py logo.png out.svg --alphamax 0.8 --opttolerance 0.1
  python scripts/trace.py art.png out.svg --debug
        """,
    )

    parser.add_argument("input",  help="Input raster image (PNG / JPEG / BMP)")
    parser.add_argument("output", help="Output SVG file path")

    # --- Preprocessing ---
    pre = parser.add_argument_group("Preprocessing")
    pre.add_argument("--threshold", type=int,   default=128,
                     help="Grayscale threshold 0-255 (default: 128, ignored if --otsu)")
    pre.add_argument("--otsu",      action="store_true",
                     help="Use OTSU automatic threshold instead of --threshold")
    pre.add_argument("--invert",    action="store_true",
                     help="Invert bitmap (trace dark shapes on light background)")
    pre.add_argument("--close",     type=int,   default=0,
                     help="Morphological close kernel (pixels) to bridge thin gaps (default: 0)")

    # --- Potrace pipeline ---
    pt = parser.add_argument_group("Potrace pipeline")
    pt.add_argument("--turdsize",     type=int,   default=2,
                    help="Suppress speckles ≤ this area in px² (default: 2)")
    pt.add_argument("--alphamax",     type=float, default=1.0,
                    help="Corner threshold 0..4: lower=more corners (default: 1.0)")
    pt.add_argument("--opttolerance", type=float, default=0.2,
                    help="Curve-merge tolerance in pixels (default: 0.2)")
    pt.add_argument("--no-optcurve",  action="store_true",
                    help="Disable optiCurve merging pass")
    pt.add_argument("--turnpolicy",   default="minority",
                    choices=["minority","majority","right","black","white"],
                    help="Ambiguous-turn resolution policy (default: minority)")

    # --- Output ---
    out = parser.add_argument_group("Output")
    out.add_argument("--fill",  default="#000000",
                     help="SVG fill color (default: #000000)")
    out.add_argument("--bg",    default=None,
                     help="Optional background rect color (default: none = transparent)")
    out.add_argument("--scale", type=float, default=1.0,
                     help="Output scale factor (default: 1.0)")

    # --- Debug ---
    parser.add_argument("--debug", action="store_true",
                        help="Dump intermediate pipeline state (pathlist, polygons, "
                             "vertices, curve tags) as JSON lines to stdout")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[trace] ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ── Load & preprocess ─────────────────────────────────────────────────────
    print(f"[trace] Loading   {args.input}")
    gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[trace] ERROR: cannot read image.", file=sys.stderr)
        sys.exit(1)

    if args.otsu:
        thresh_val, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"[trace] OTSU threshold: {thresh_val}")
    else:
        _, binary = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY)

    if args.invert:
        binary = cv2.bitwise_not(binary)

    if args.close > 0:
        kernel = np.ones((args.close, args.close), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        print(f"[trace] Morphological close: {args.close}×{args.close}")

    h, w = gray.shape[:2]

    # ── Build tracer ───────────────────────────────────────────────────────────
    turn_policies = {
        "minority": potrace.TURNPOLICY_MINORITY,
        "majority": potrace.TURNPOLICY_MAJORITY,
        "right": potrace.TURNPOLICY_RIGHT,
        "black": potrace.TURNPOLICY_BLACK,
        "white": potrace.TURNPOLICY_WHITE,
    }
    policy = turn_policies.get(args.turnpolicy, potrace.TURNPOLICY_MINORITY)

    print(f"[trace] Tracing   {w}×{h} px  "
          f"(alphamax={args.alphamax}, opttol={args.opttolerance}, "
          f"turdsize={args.turdsize}, optcurve={not args.no_optcurve})")

    bmp = potrace.Bitmap(binary.astype(bool))
    path = bmp.trace(
        turdsize=args.turdsize,
        turnpolicy=policy,
        alphamax=args.alphamax,
        opticurve=not args.no_optcurve,
        opttolerance=args.opttolerance,
    )
    
    # Export SVG
    svg_paths = []
    s = args.scale
    for curve in path:
        parts = [f"M {curve.start_point.x * s:.3f} {curve.start_point.y * s:.3f}"]
        for segment in curve:
            if segment.is_corner:
                parts.append(f"L {segment.c.x * s:.3f} {segment.c.y * s:.3f} L {segment.end_point.x * s:.3f} {segment.end_point.y * s:.3f}")
            else:
                parts.append(
                    f"C {segment.c1.x * s:.3f} {segment.c1.y * s:.3f} "
                    f"{segment.c2.x * s:.3f} {segment.c2.y * s:.3f} "
                    f"{segment.end_point.x * s:.3f} {segment.end_point.y * s:.3f}"
                )
        parts.append("Z")
        svg_paths.append(" ".join(parts))

    d_string = " ".join(svg_paths)
    bg_rect = f'  <rect width="100%" height="100%" fill="{args.bg}"/>\n' if args.bg else ""
    sw_w, sw_h = int(w * s), int(h * s)
    svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n<svg xmlns="http://www.w3.org/2000/svg" width="{sw_w}" height="{sw_h}" viewBox="0 0 {sw_w} {sw_h}">\n{bg_rect}  <path d="{d_string}" fill="{args.fill}" fill-rule="evenodd" stroke="none"/>\n</svg>\n'

    # ── Write output ───────────────────────────────────────────────────────────
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"[trace] Written   {args.output}  ✓")


if __name__ == "__main__":
    main()