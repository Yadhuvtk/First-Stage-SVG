import numpy as np
import pytest

# ── Legacy test (yd_vector still works) ──────────────────────────────────────
def test_extract_contours_runs():
    """Legacy: yd_vector.contours.extract_contours still returns results."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from yd_vector.contours import extract_contours

    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[10:30, 10:30] = 255
    contours = extract_contours(binary, min_area=5)
    assert len(contours) >= 1


# ── New: Potrace bitmap path-walk ─────────────────────────────────────────────
def test_bitmap_pathwalk():
    """
    bm_to_pathlist uses the internal Bitmap pixel-walk (NOT cv2.findContours).
    A 30×30 binary image with a 10×10 square should produce at least 1 path.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tracer import Bitmap, bm_to_pathlist

    w, h = 30, 30
    data = [0] * (w * h)
    # Draw a 10×10 filled square at (5,5)
    for y in range(5, 15):
        for x in range(5, 15):
            data[y * w + x] = 1
    bm = Bitmap(w, h, data)

    info = {
        "turdsize": 1,
        "turnpolicy": "minority",
    }
    paths = bm_to_pathlist(bm, info)
    assert len(paths) >= 1, "Expected at least 1 path from a filled square bitmap"
    assert paths[0].len > 0, "Path should have pixel boundary points"


def test_bitmap_pathwalk_no_contours_on_blank():
    """An all-white bitmap should produce no paths."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tracer import Bitmap, bm_to_pathlist

    w, h = 20, 20
    bm = Bitmap(w, h, [0] * (w * h))
    info = {"turdsize": 0, "turnpolicy": "minority"}
    paths = bm_to_pathlist(bm, info)
    assert len(paths) == 0, "Blank bitmap should produce no paths"