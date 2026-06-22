# Gather up utility functions.

from dataclasses import dataclass

import cv2
from ic.video import models


@dataclass
class FrameExtract:
    index: int
    timestamp_sec: float
    bgr: any


def extract_frames(video_path, target_fps=2, max_frames=60):
    """Extract frames at target_fps from a video.

    Arguments:
        video_path: Path to the video file.
        target_fps [2]: Approximate frames per second to sample.
        max_frames [60]: Maximum number of frames to extract.

    Returns:
        A list of FrameExtract instances with: index, timestamp_sec, and bgr.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / source_fps

    # How many source frames to skip between samples
    frame_stride = max(1, int(round(source_fps / target_fps)))

    print(f"Video: {duration_sec:.1f}s @ {source_fps:.1f}fps, "
          f"sampling every {frame_stride} frames (~{target_fps}fps)")

    frames = []
    frame_idx = 0
    sampled = 0

    while sampled < max_frames:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            timestamp = frame_idx / source_fps
            frame_extract = FrameExtract(
                index=frame_idx,
                timestamp_sec=timestamp,
                bgr=frame
            )
            frames.append(frame_extract)
            sampled += 1

        frame_idx += 1

    capture.release()
    return frames


@dataclass
class BicycleCrop:
    bgr: any
    box: tuple(int, int, int, int)  # (x1, y1, x2, y2)
    confidence: float


def detect_largest_bicycle(image_bgr, conf_threshold=0.4, padding_pct=0.05):
    """Detect and return the largest bicycle crop in the image.

    Arguments:
        image_bgr: Input image in BGR format (as loaded by OpenCV).
        conf_threshold: Minimum confidence to consider a detection valid.
        padding_pct: Optional padding around the detected box to capture edge details.

    Returns:
        A BicycleCrop instance with:
            - bgr is the cropped image of the detected bicycle (or None if not found)
            - box is the bounding box coordinates (x1, y1, x2, y2) (or None if not found)
            - confidence is the detection confidence (or 0.0 if not found)
    """
    yolo_26n = models.yolo_26n
    results = yolo_26n(image_bgr, verbose=False)

    largest_box = None
    largest_area = 0
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo_26n.names[cls] == 'bicycle' and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = (int(x1), int(y1), int(x2), int(y2))
                    best_conf = conf

    if largest_box is None:
        return BicycleCrop(
            bgr=None,
            box=None,
            confidence=0.0
        )

    # Pad the box slightly to capture edge details
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = largest_box
    pad_x = int(padding_pct * (x2 - x1))
    pad_y = int(padding_pct * (y2 - y1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]

    return BicycleCrop(
        bgr=crop,
        box=(x1, y1, x2, y2),
        confidence=best_conf
    )
