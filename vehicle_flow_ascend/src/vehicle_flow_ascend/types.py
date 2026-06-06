from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Line:
    start: Point
    end: Point


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    raw_class_id: int | None = None

    @property
    def centroid(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    previous_centroid: Point | None
    current_centroid: Point
    disappeared: int = 0
