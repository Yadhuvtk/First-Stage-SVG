from __future__ import annotations

from pathlib import Path

from PIL import Image

from yd_vector.contours import extract_contours
from yd_vector.fit import prepare_paths
from yd_vector.models import TraceOptions, TraceResult
from yd_vector.preprocess import preprocess_image
from yd_vector.simplify import simplify_contours
from yd_vector.svg_builder import write_svg as _bezier_write_svg
from yd_vector.utils import ensure_parent_dir


def run_trace(options: TraceOptions) -> TraceResult:
    rgb, gray, binary = preprocess_image(
        image_path=str(options.input_path),
        threshold=options.threshold,
        invert=options.invert,
    )

    height, width = gray.shape[:2]

    contours = extract_contours(binary, min_area=options.min_area)
    contours = simplify_contours(contours, tolerance=options.simplify_tolerance)

    # Build fitted_paths for TraceResult back-compat (straight-line representation)
    fitted_paths = prepare_paths(contours)

    # Use Bézier-based svg_builder for high-quality SVG output
    svg_path = _bezier_write_svg(
        output_path=options.output_path,
        contours=contours,
        width=width,
        height=height,
        fill=options.fill,
        corner_threshold=getattr(options, "corner_threshold", 0.85),
        bezier_min_points=getattr(options, "bezier_min_points", 3),
    )

    debug_mask_path = None
    if options.debug:
        debug_mask_path = options.output_path.parent / f"{options.output_path.stem}_mask.png"
        ensure_parent_dir(debug_mask_path)
        Image.fromarray(binary).save(debug_mask_path)

    return TraceResult(
        width=width,
        height=height,
        contours=contours,
        fitted_paths=fitted_paths,
        svg_path=svg_path,
        debug_mask_path=debug_mask_path,
    )