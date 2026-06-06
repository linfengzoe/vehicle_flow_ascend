from __future__ import annotations

import cv2

from vehicle_flow_ascend.types import Detection, Line, Track


_CLASS_COLORS = {
    "car": (60, 220, 60),
    "bus": (255, 180, 40),
    "truck": (80, 160, 255),
    "two_wheeler": (220, 80, 220),
}


def draw_overlay(
    frame,
    detections: list[Detection],
    tracks: list[Track],
    line: Line,
    counts: dict[str, int],
    fps: float,
):
    output = frame.copy()
    cv2.line(
        output,
        (int(line.start.x), int(line.start.y)),
        (int(line.end.x), int(line.end.y)),
        (0, 255, 255),
        2,
    )

    track_by_centroid = {(track.current_centroid.x, track.current_centroid.y): track for track in tracks}
    for detection in detections:
        color = _CLASS_COLORS.get(detection.class_name, (255, 255, 255))
        cv2.rectangle(
            output,
            (int(detection.x1), int(detection.y1)),
            (int(detection.x2), int(detection.y2)),
            color,
            2,
        )
        track = track_by_centroid.get((detection.centroid.x, detection.centroid.y))
        track_label = f" #{track.track_id}" if track is not None else ""
        label = f"{detection.class_name}{track_label} {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (int(detection.x1), max(15, int(detection.y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    y = 25
    cv2.putText(output, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 25
    cv2.putText(output, f"Total: {counts.get('total', 0)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for class_name in ("car", "bus", "truck", "two_wheeler"):
        y += 22
        cv2.putText(
            output,
            f"{class_name}: {counts.get(class_name, 0)}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _CLASS_COLORS.get(class_name, (255, 255, 255)),
            2,
        )
    return output
