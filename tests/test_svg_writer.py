from yd_vector.models import ContourData
from yd_vector.svg_writer import _contour_to_path_d


def test_contour_to_path_d():
    contour = ContourData(
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
        area=100,
        parent_index=-1,
        is_hole=False,
    )
    d = _contour_to_path_d(contour)
    assert d.startswith("M ")
    assert d.endswith("Z")