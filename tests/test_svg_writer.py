import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Legacy test (yd_vector SVG writer) ───────────────────────────────────────
def test_contour_to_path_d():
    """Legacy: yd_vector.svg_writer still produces well-formed path data."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from yd_vector.models import ContourData
        from yd_vector.svg_writer import _contour_to_path_d

    contour = ContourData(
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
        area=100,
        parent_index=-1,
        is_hole=False,
    )
    d = _contour_to_path_d(contour)
    assert d.startswith("M ")
    assert d.endswith("Z")


# ── New: end-to-end Potrace pipeline SVG test ─────────────────────────────────
def test_tracer_end_to_end_circle():
    """
    PurePythonTracer.trace() on a synthetic circle binary image should produce:
    - A valid SVG string
    - fill-rule=evenodd (for correct hole handling)
    - At least one cubic Bezier 'C ' command
    """
    import cv2
    from tracer import PurePythonTracer

    # Synthetic 60×60 image with a filled circle
    img = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(img, (30, 30), 20, 255, -1)

    tracer = PurePythonTracer(turdsize=2, alphamax=1.0, opttolerance=0.2)
    svg = tracer.trace(img)

    assert isinstance(svg, str), "trace() should return a string"
    assert "<svg" in svg, "Output should be SVG"
    assert 'fill-rule="evenodd"' in svg, "SVG must include fill-rule=evenodd"
    assert "<path" in svg, "SVG must contain at least one path element"
    assert " C " in svg or "C " in svg, "SVG should contain cubic Bezier commands"


def test_tracer_end_to_end_square():
    """
    A simple square should produce CORNER-only segments (L commands, likely)
    and still be valid SVG.
    """
    from tracer import PurePythonTracer

    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255   # filled square

    tracer = PurePythonTracer(turdsize=1, alphamax=4.0)  # force all corners
    svg = tracer.trace(img)

    assert "<svg" in svg
    assert 'fill-rule="evenodd"' in svg
    assert "<path" in svg


def test_tracer_debug_mode_produces_json(capsys):
    """
    debug=True should print JSON lines to stdout for each pipeline stage.
    """
    import json
    import cv2
    from tracer import PurePythonTracer

    img = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(img, (20, 20), 10, 255, -1)

    tracer = PurePythonTracer(turdsize=1)
    tracer.trace(img, debug=True)

    captured = capsys.readouterr()
    lines = [l.strip() for l in captured.out.splitlines() if l.strip().startswith("{")]
    assert len(lines) > 0, "Debug mode should print JSON lines"
    for line in lines:
        obj = json.loads(line)   # must be valid JSON
        assert "stage" in obj, "Each debug JSON line must have a 'stage' key"