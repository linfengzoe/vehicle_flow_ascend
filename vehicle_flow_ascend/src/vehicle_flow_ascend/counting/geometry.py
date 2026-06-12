from __future__ import annotations

from vehicle_flow_ascend.types import Line, Point


def point_side(line: Line, point: Point) -> float:
    """Return the signed side of a point relative to a directed line."""
    line_dx = line.end.x - line.start.x
    line_dy = line.end.y - line.start.y
    point_dx = point.x - line.start.x
    point_dy = point.y - line.start.y

    return line_dx * point_dy - line_dy * point_dx


def _orientation(start: Point, end: Point, point: Point) -> float:
    segment_dx = end.x - start.x
    segment_dy = end.y - start.y
    point_dx = point.x - start.x
    point_dy = point.y - start.y

    return segment_dx * point_dy - segment_dy * point_dx


def point_on_segment(line: Line, point: Point) -> bool:
    """Return whether a point lies on the finite line segment."""
    if point_side(line, point) != 0:
        return False

    return (
        min(line.start.x, line.end.x) <= point.x <= max(line.start.x, line.end.x)
        and min(line.start.y, line.end.y) <= point.y <= max(line.start.y, line.end.y)
    )


def _movement_intersects_counting_segment(line: Line, previous: Point, current: Point) -> bool:
    line_start_side = _orientation(previous, current, line.start)
    line_end_side = _orientation(previous, current, line.end)

    return line_start_side * line_end_side <= 0


def crossed_line(line: Line, previous: Point | None, current: Point) -> bool:
    """Return whether movement crossed the finite line segment from one strict side to the other."""
    if previous is None or previous == current:
        return False

    previous_side = point_side(line, previous)
    current_side = point_side(line, current)

    if previous_side == 0 or current_side == 0:
        return False

    return previous_side * current_side < 0 and _movement_intersects_counting_segment(
        line,
        previous,
        current,
    )
