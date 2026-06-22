"""Image edits that model what a scammer does to a stolen listing photo.

Each attack takes a BGR image array and returns a modified BGR array. These are
the operations the watermark must survive to be useful for PB-658: recompression
(re-saving as JPEG), resizing, and cropping.
"""

import cv2
import numpy as np


def jpeg_recompress(image: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip the image through JPEG at the given quality (0-100)."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rescale(image: np.ndarray, scale: float) -> np.ndarray:
    """Resample the image by `scale` (0.5 = half size). Models a posted resize."""
    h, w = image.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (nw, nh), interpolation=interp)


def center_crop(image: np.ndarray, keep: float) -> np.ndarray:
    """Center-crop, keeping `keep` fraction of each dimension (0.9 = trim 10%)."""
    h, w = image.shape[:2]
    nh, nw = int(h * keep), int(w * keep)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    return image[y0:y0 + nh, x0:x0 + nw]


def default_gauntlet() -> list[tuple[str, "callable"]]:
    """Ordered (name, attack) pairs covering the realistic scammer edit space.

    Includes a no-op baseline so a clean decode is visible at the top, and a
    combined resize+recompress case (the worst realistic scenario).
    """
    return [
        ("baseline (no attack)", lambda im: im),
        ("jpeg q90", lambda im: jpeg_recompress(im, 90)),
        ("jpeg q75", lambda im: jpeg_recompress(im, 75)),
        ("jpeg q50", lambda im: jpeg_recompress(im, 50)),
        ("resize 75%", lambda im: rescale(im, 0.75)),
        ("resize 50%", lambda im: rescale(im, 0.5)),
        ("crop 90%", lambda im: center_crop(im, 0.9)),
        ("crop 75%", lambda im: center_crop(im, 0.75)),
        ("resize 50% + jpeg q75", lambda im: jpeg_recompress(rescale(im, 0.5), 75)),
    ]
