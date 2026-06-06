from vehicle_flow_ascend.counting.geometry import crossed_line, point_side
from vehicle_flow_ascend.types import Line, Point


def test_point_side_returns_signed_values_for_horizontal_line() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0))

    assert point_side(line, Point(5.0, 1.0)) > 0
    assert point_side(line, Point(5.0, -1.0)) < 0
    assert point_side(line, Point(5.0, 0.0)) == 0


def test_point_side_returns_signed_values_for_vertical_line() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(0.0, 10.0))

    assert point_side(line, Point(-1.0, 5.0)) > 0
    assert point_side(line, Point(1.0, 5.0)) < 0
    assert point_side(line, Point(0.0, 5.0)) == 0


def test_crossed_line_returns_true_when_points_are_on_opposite_sides() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0))

    assert crossed_line(line, Point(5.0, -1.0), Point(5.0, 1.0)) is True


def test_crossed_line_returns_false_when_points_are_on_same_side() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0))

    assert crossed_line(line, Point(5.0, 1.0), Point(6.0, 2.0)) is False


def test_crossed_line_returns_false_without_previous_point() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0))

    assert crossed_line(line, None, Point(5.0, 1.0)) is False


def test_crossed_line_returns_false_when_either_point_is_on_line() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0))

    assert crossed_line(line, Point(5.0, 0.0), Point(5.0, 1.0)) is False
    assert crossed_line(line, Point(5.0, -1.0), Point(5.0, 0.0)) is False


def test_crossed_line_returns_false_when_crossing_outside_finite_segment() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0))

    assert crossed_line(line, Point(20.0, -1.0), Point(20.0, 1.0)) is False


def test_crossed_line_returns_false_for_stationary_point() -> None:
    line = Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0))
    point = Point(5.0, 1.0)

    assert crossed_line(line, point, point) is False
