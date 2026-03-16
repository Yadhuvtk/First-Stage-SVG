from __future__ import annotations

from pathlib import Path

from PIL import Image

from yd_vector.contours import extract_contours
from yd_vector.fit import prepare_paths
from yd_vector.models import TraceOptions, TraceResult
from yd_vector.preprocess import preprocess_image
from yd_vector.simplify import simplify_contours
from yd_vector.svg_writer import write_svg
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
    contours = prepare_paths(contours)

    svg_path = write_svg(
        output_path=options.output_path,
        width=width,
        height=height,
        contours=contours,
        fill=options.fill,
        stroke=options.stroke,
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
        svg_path=svg_path,
        debug_mask_path=debug_mask_path,
    )