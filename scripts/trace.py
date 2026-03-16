#!/usr/bin/env python3
"""
Usage:
    python scripts/trace.py input.png output.svg
    python scripts/trace.py input.png output.svg --threshold 120 --turd 4 --alpha 1.2
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yd_vector.tracer import Params, trace, export_svg


def main():
    parser = argparse.ArgumentParser(description="YD-Vector: bitmap → SVG tracer")
    parser.add_argument("input",  help="Input image (PNG/JPG/BMP/etc)")
    parser.add_argument("output", help="Output SVG file")
    parser.add_argument("--threshold", type=int,   default=128,  help="Binarization threshold 0-255 (default: 128)")
    parser.add_argument("--turd",      type=int,   default=2,    help="Suppress speckles smaller than N pixels (default: 2)")
    parser.add_argument("--alpha",     type=float, default=1.0,  help="Corner smoothness 0..1.333 (default: 1.0)")
    parser.add_argument("--opti",      type=float, default=0.2,  help="Curve optimization tolerance (default: 0.2)")
    parser.add_argument("--scale",     type=float, default=1.0,  help="Output scale multiplier (default: 1.0)")
    parser.add_argument("--invert",    action="store_true",       help="Invert bitmap before tracing")
    parser.add_argument("--fg",        default="#000000",         help="Foreground color (default: #000000)")
    parser.add_argument("--bg",        default="white",           help="Background color (default: white)")
    parser.add_argument("--no-opti",   action="store_true",       help="Disable curve optimization")

    args = parser.parse_args()

    params = Params(
        threshold=args.threshold,
        turd_size=args.turd,
        alpha_max=args.alpha,
        opti_curve=not args.no_opti,
        opti_tolerance=args.opti,
        scale=args.scale,
        invert=args.invert,
    )

    print(f"[yd-vector] Tracing: {args.input}")
    paths_d, w, h = trace(args.input, params)
    export_svg(paths_d, w, h, args.output, foreground=args.fg, background=args.bg)


if __name__ == "__main__":
    main()