#!/usr/bin/env python3
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yd_vector.tracer import Params, trace, export_svg

def main():
    parser = argparse.ArgumentParser(description="YD-Vector: bitmap → SVG tracer")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--threshold", type=int,   default=128)
    parser.add_argument("--turd",      type=int,   default=2)
    parser.add_argument("--alpha",     type=float, default=1.0)
    parser.add_argument("--opti",      type=float, default=0.2)
    parser.add_argument("--scale",     type=float, default=1.0)
    parser.add_argument("--invert",    action="store_true")
    parser.add_argument("--fg",        default="#000000")
    parser.add_argument("--bg",        default="white")
    args = parser.parse_args()

    params = Params(
        threshold=args.threshold,
        turd_size=args.turd,
        alpha_max=args.alpha,
        opti_tolerance=args.opti,
        scale=args.scale,
        invert=args.invert,
    )

    print(f"[yd-vector] Tracing: {args.input}")
    paths, w, h = trace(args.input, params)
    export_svg(paths, w, h, args.output, foreground=args.fg, background=args.bg)

if __name__ == "__main__":
    main()