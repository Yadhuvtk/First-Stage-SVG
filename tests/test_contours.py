import numpy as np

from yd_vector.contours import extract_contours


def test_extract_contours_runs():
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[10:30, 10:30] = 255
    contours = extract_contours(binary, min_area=5)
    assert len(contours) >= 1