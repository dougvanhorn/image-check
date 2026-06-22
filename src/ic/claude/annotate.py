"""Write a copy of a video with YOLO bicycle bounding boxes drawn on each frame."""
from pathlib import Path

import cv2

from ic.video import models


BICYCLE_CLASS = 1
DEFAULT_CONF = 0.4
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 3


def annotate_video(
        source,
        output,
        conf=DEFAULT_CONF,
        box_color=BOX_COLOR,
        box_thickness=BOX_THICKNESS,
) -> int:
    """Read `source`, draw the largest bicycle box per frame, write to `output`.

    Preserves the source resolution and fps. Frames with no bike pass through
    unmodified. Returns the count of frames written.
    """
    source = str(source)
    output = str(output)

    yolo = models.yolo_26n

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for: {output}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, verbose=False,
                           classes=[BICYCLE_CLASS], conf=conf)

            best = None
            best_area = 0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = (x1, y1, x2, y2, float(box.conf[0]))

            if best is not None:
                x1, y1, x2, y2, c = best
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              box_color, box_thickness)
                cv2.putText(frame, f"bike {c:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            box_color, 2)

            writer.write(frame)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    return frame_count
