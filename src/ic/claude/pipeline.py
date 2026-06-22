"""Video-to-listing verification pipeline.

Takes a listing image and a verification video, uses YOLO to locate the bike
in each, compares with ORB keypoint matching, runs a homography-residual
liveness check against photo-of-photo spoofing, and writes a demo video
with a title card and per-frame status panel.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ic.video import models, settings


# ----- Tunables -----
TARGET_FPS = 2
MAX_FRAMES = 60
YOLO_CONF_THRESHOLD = 0.4
BBOX_PADDING_PCT = 0.05
CROP_WIDTH = 800

ORB_FEATURES = 2000
ORB_RATIO = 0.75

LIVENESS_PAIR_INTERVAL_SEC = 0.8
LIVENESS_FLAT_THRESHOLD = 0.85

VERDICT_MATCH_MIN_PCT = 8.0
VERDICT_PROBABLE_MIN_PCT = 3.0
VERDICT_DETECTION_RATE_MIN = 30.0

DEMO_FPS = 10
DEMO_HOLD_FRAMES = 5
DEMO_FINAL_HOLD_MULTIPLIER = 15
TITLE_CARD_SECONDS = 2.5

# ----- Colors (BGR) -----
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 50, 200)
COLOR_YELLOW = (0, 200, 220)
COLOR_GRAY = (120, 120, 120)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK = (30, 30, 30)


@dataclass
class FrameResult:
    frame_index: int
    timestamp_sec: float
    bike_detected: bool
    bike_confidence: float = 0.0
    bbox: Optional[tuple] = None
    orb_match_pct: float = 0.0
    orb_inliers: int = 0


@dataclass
class VerificationResult:
    video_path: str
    listing_path: str
    total_frames_sampled: int
    frames_with_bike: int
    frame_results: list = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    verdict: str = ""
    liveness: dict = field(default_factory=dict)


# ----- Frame extraction -----

def extract_frames(video_path, target_fps=TARGET_FPS, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(source_fps / target_fps)))

    frames = []
    idx = 0
    sampled = 0
    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append((idx, idx / source_fps, frame))
            sampled += 1
        idx += 1

    cap.release()
    return frames


# ----- YOLO detection -----

def detect_largest_bicycle(image_bgr, conf_threshold=YOLO_CONF_THRESHOLD,
                           padding_pct=BBOX_PADDING_PCT):
    yolo = models.yolo_26n
    results = yolo(image_bgr, verbose=False)

    largest = None
    largest_area = 0
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo.names[cls] != 'bicycle' or conf < conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest = (int(x1), int(y1), int(x2), int(y2))
                best_conf = conf

    if largest is None:
        return None, None, 0.0

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = largest
    pad_x = int(padding_pct * (x2 - x1))
    pad_y = int(padding_pct * (y2 - y1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return image_bgr[y1:y2, x1:x2], (x1, y1, x2, y2), best_conf


def resize_to_width(image, target_width=CROP_WIDTH):
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    return cv2.resize(image, (target_width, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# ----- ORB comparison -----

def compare_with_orb(crop1, crop2, ratio_threshold=ORB_RATIO):
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    if len(good) < 4:
        return 0.0, 0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    denom = min(len(kp1), len(kp2))
    match_pct = (inliers / denom) * 100 if denom > 0 else 0.0
    return match_pct, inliers


# ----- Homography liveness -----

def homography_inlier_ratio(crop_a, crop_b):
    """High ratio = single planar transform fits well = likely a flat image."""
    gray1 = cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [pair[0] for pair in matches
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]

    if len(good) < 20:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None

    return float(mask.sum() / len(mask))


@dataclass
class LivenessAccumulator:
    ratios: list = field(default_factory=list)
    previous_crop: Optional[np.ndarray] = None
    previous_timestamp: float = -999.0
    pair_interval_sec: float = LIVENESS_PAIR_INTERVAL_SEC

    def update(self, crop, timestamp):
        if self.previous_crop is None:
            self.previous_crop = crop
            self.previous_timestamp = timestamp
        elif (timestamp - self.previous_timestamp) >= self.pair_interval_sec:
            ratio = homography_inlier_ratio(self.previous_crop, crop)
            if ratio is not None:
                self.ratios.append(ratio)
            self.previous_crop = crop
            self.previous_timestamp = timestamp
        return self.current_state()

    def current_state(self):
        if not self.ratios:
            return {"status": "analyzing", "ratio": None, "samples": 0}
        avg = float(np.mean(self.ratios))
        if len(self.ratios) < 2:
            status = "analyzing"
        elif avg > LIVENESS_FLAT_THRESHOLD:
            status = "flat"
        else:
            status = "3d"
        return {"status": status, "ratio": avg, "samples": len(self.ratios)}


# ----- Aggregation + verdict -----

def aggregate_scores(frame_results):
    bike_frames = [f for f in frame_results if f.bike_detected]
    if not bike_frames:
        return {
            "bike_detection_rate": 0.0,
            "orb_max": 0.0, "orb_mean": 0.0, "orb_top3_mean": 0.0,
        }
    orb_scores = [f.orb_match_pct for f in bike_frames]
    orb_top3 = sorted(orb_scores, reverse=True)[:3]
    return {
        "bike_detection_rate": len(bike_frames) / len(frame_results) * 100,
        "orb_max": max(orb_scores),
        "orb_mean": float(np.mean(orb_scores)),
        "orb_top3_mean": float(np.mean(orb_top3)),
    }


def make_verdict(agg, liveness_state):
    if agg["bike_detection_rate"] < VERDICT_DETECTION_RATE_MIN:
        return "insufficient_bike_visibility"
    if liveness_state.get("status") == "flat":
        return "likely_spoof_flat_image"
    orb_top = agg["orb_top3_mean"]
    if orb_top >= VERDICT_MATCH_MIN_PCT:
        return "likely_same_bike"
    if orb_top >= VERDICT_PROBABLE_MIN_PCT:
        return "probably_same_bike"
    return "different_bike"


# ----- Demo frame rendering -----

def _draw_status_pill(img, x, y, label, value, color, width=220, height=44):
    cv2.rectangle(img, (x, y), (x + width, y + height), COLOR_DARK, -1)
    cv2.rectangle(img, (x, y), (x + 8, y + height), color, -1)
    cv2.putText(img, label, (x + 18, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
    cv2.putText(img, value, (x + 18, y + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _liveness_color(status):
    return {"3d": COLOR_GREEN, "flat": COLOR_RED}.get(status, COLOR_YELLOW)


def _match_color(orb_pct):
    if orb_pct >= VERDICT_MATCH_MIN_PCT:
        return COLOR_GREEN
    if orb_pct >= VERDICT_PROBABLE_MIN_PCT:
        return COLOR_YELLOW
    return COLOR_RED


def make_demo_frame(video_frame, listing_crop, bbox=None,
                    orb_pct=None, orb_inliers=None,
                    liveness_state=None, final_verdict=None):
    LISTING_WIDTH = 300
    PANEL_WIDTH = 260
    vh, vw = video_frame.shape[:2]

    lh, lw = listing_crop.shape[:2]
    listing_h = int(lh * (LISTING_WIDTH / lw))
    listing_resized = cv2.resize(listing_crop, (LISTING_WIDTH, min(listing_h, vh)))
    if listing_resized.shape[0] < vh:
        pad = np.zeros((vh - listing_resized.shape[0], LISTING_WIDTH, 3),
                       dtype=np.uint8)
        pad[:] = COLOR_DARK
        listing_panel = np.vstack([listing_resized, pad])
    else:
        listing_panel = listing_resized[:vh]

    annotated = video_frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if liveness_state and liveness_state.get("status") == "flat":
            box_color = COLOR_RED
        elif liveness_state and liveness_state.get("status") == "3d":
            box_color = COLOR_GREEN
        else:
            box_color = COLOR_YELLOW
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 4)

    panel = np.zeros((vh, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = COLOR_DARK
    cv2.putText(panel, "ANALYSIS", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    cv2.line(panel, (18, 38), (PANEL_WIDTH - 18, 38), COLOR_GRAY, 1)

    y = 54
    pill_w = PANEL_WIDTH - 36

    if orb_pct is not None:
        _draw_status_pill(panel, 18, y, "ORB MATCH",
                          f"{orb_pct:.1f}%", _match_color(orb_pct), width=pill_w)
        y += 54
        _draw_status_pill(panel, 18, y, "ORB INLIERS",
                          str(orb_inliers or 0),
                          _match_color(orb_pct), width=pill_w)
        y += 64
    else:
        _draw_status_pill(panel, 18, y, "ORB MATCH",
                          "no bike", COLOR_GRAY, width=pill_w)
        y += 64

    if liveness_state:
        status = liveness_state.get("status", "analyzing")
        ratio = liveness_state.get("ratio")
        if status == "3d":
            label_val = "LIVE 3D SCENE"
        elif status == "flat":
            label_val = "FLAT (SPOOF?)"
        elif ratio is not None:
            label_val = f"H-ratio: {ratio:.2f}"
        else:
            label_val = "analyzing..."
        _draw_status_pill(panel, 18, y, "LIVENESS",
                          label_val, _liveness_color(status), width=pill_w)

    combined = np.hstack([listing_panel, annotated, panel])
    total_w = combined.shape[1]

    header = np.zeros((34, total_w, 3), dtype=np.uint8)
    header[:] = COLOR_DARK
    cv2.putText(header, "LISTING PHOTO", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    cv2.putText(header, "VERIFICATION VIDEO", (LISTING_WIDTH + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    cv2.putText(header, "METRICS", (LISTING_WIDTH + vw + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

    banner = np.zeros((56, total_w, 3), dtype=np.uint8)
    if final_verdict:
        if final_verdict in ("likely_same_bike", "probably_same_bike"):
            banner[:] = (0, 100, 0)
            text = f"[PASS] {final_verdict}"
        elif "spoof" in final_verdict or final_verdict == "different_bike":
            banner[:] = (0, 0, 120)
            text = f"[FAIL] {final_verdict}"
        else:
            banner[:] = (60, 60, 60)
            text = f"[?] {final_verdict}"
        cv2.putText(banner, text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
    else:
        banner[:] = COLOR_DARK
        cv2.putText(banner, "Analyzing video...", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    return np.vstack([header, combined, banner])


def make_title_card(width, height, listing_path, video_path,
                    title="Pinkbike BuySell",
                    subtitle="Video Verification POC"):
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (20, 20, 28)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(title, font, 1.8, 3)
    ty = height // 2 - 40
    cv2.putText(card, title, ((width - tw) // 2, ty), font, 1.8,
                COLOR_WHITE, 3, cv2.LINE_AA)

    (sw, _), _ = cv2.getTextSize(subtitle, font, 0.8, 2)
    sy = ty + 40
    cv2.putText(card, subtitle, ((width - sw) // 2, sy), font, 0.8,
                (180, 180, 200), 2, cv2.LINE_AA)

    line_y = sy + 30
    cv2.line(card, (width // 2 - 80, line_y), (width // 2 + 80, line_y),
             (100, 140, 200), 2, cv2.LINE_AA)

    meta_color = (160, 160, 170)
    meta_y = height - 60
    cv2.putText(card, f"Listing: {Path(listing_path).name}",
                (30, meta_y), font, 0.5, meta_color, 1, cv2.LINE_AA)
    cv2.putText(card, f"Video:   {Path(video_path).name}",
                (30, meta_y + 20), font, 0.5, meta_color, 1, cv2.LINE_AA)
    cv2.putText(card, f"Run:     {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                (30, meta_y + 40), font, 0.5, meta_color, 1, cv2.LINE_AA)

    return card


# ----- Main orchestrator -----

def verify(
        video_path,
        listing_path,
        output_dir=None,
        demo_filename='demo_output.mp4',
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
        generate_demo=True,
) -> VerificationResult:
    output_dir = Path(output_dir) if output_dir else settings.VAR_DIR / "verify"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(demo_filename)
    demo_filename = path.parent / f'{path.stem}_{timestamp}{path.suffix}'
    demo_path = output_dir / demo_filename

    listing_image = cv2.imread(str(listing_path))
    if listing_image is None:
        raise FileNotFoundError(f"Could not read listing: {listing_path}")
    listing_crop, _, _ = detect_largest_bicycle(listing_image)
    if listing_crop is None:
        raise RuntimeError("No bicycle detected in listing image")
    listing_crop = resize_to_width(listing_crop)

    frames = extract_frames(video_path, target_fps=target_fps,
                            max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    frame_results = []
    liveness = LivenessAccumulator()
    demo_writer = None

    for i, (idx, ts, frame) in enumerate(frames):
        crop, bbox, conf = detect_largest_bicycle(frame)

        result = FrameResult(
            frame_index=idx, timestamp_sec=ts,
            bike_detected=crop is not None,
            bike_confidence=conf, bbox=bbox,
        )

        orb_pct = None
        orb_inliers = None
        motion_state = liveness.current_state()

        if crop is not None:
            crop_resized = resize_to_width(crop)
            pct, inliers = compare_with_orb(listing_crop, crop_resized)
            result.orb_match_pct = pct
            result.orb_inliers = inliers
            orb_pct = pct
            orb_inliers = inliers
            motion_state = liveness.update(crop_resized, ts)

        frame_results.append(result)

        if not generate_demo:
            continue

        is_last = (i == len(frames) - 1)
        final = None
        if is_last:
            agg = aggregate_scores(frame_results)
            final = make_verdict(agg, motion_state)

        demo_frame = make_demo_frame(
            video_frame=frame, listing_crop=listing_crop, bbox=bbox,
            orb_pct=orb_pct, orb_inliers=orb_inliers,
            liveness_state=motion_state, final_verdict=final,
        )

        if demo_writer is None:
            h, w = demo_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            demo_writer = cv2.VideoWriter(str(demo_path), fourcc,
                                          DEMO_FPS, (w, h))
            if not demo_writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {demo_path}")

            title_card = make_title_card(w, h, listing_path, video_path)
            for _ in range(int(TITLE_CARD_SECONDS * DEMO_FPS)):
                demo_writer.write(title_card)

        hold = DEMO_HOLD_FRAMES * (DEMO_FINAL_HOLD_MULTIPLIER if is_last else 1)
        for _ in range(hold):
            demo_writer.write(demo_frame)

    if demo_writer is not None:
        demo_writer.release()

    agg = aggregate_scores(frame_results)
    final_state = liveness.current_state()
    verdict = make_verdict(agg, final_state)

    return VerificationResult(
        video_path=str(video_path),
        listing_path=str(listing_path),
        total_frames_sampled=len(frame_results),
        frames_with_bike=sum(1 for f in frame_results if f.bike_detected),
        frame_results=frame_results,
        aggregate=agg,
        verdict=verdict,
        liveness=final_state,
    )
