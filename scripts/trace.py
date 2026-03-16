from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.cli import parse_args
from yd_vector.pipeline import run_trace


def main() -> None:
    options = parse_args()
    result = run_trace(options)

    print(f"[YD-Vector] SVG saved to: {result.svg_path}")
    print(f"[YD-Vector] Size: {result.width}x{result.height}")
    print(f"[YD-Vector] Contours: {len(result.contours)}")

    if result.debug_mask_path:
        print(f"[YD-Vector] Debug mask saved to: {result.debug_mask_path}")


if __name__ == "__main__":
    main()