# Pinkbike BuySell Video Verification POC

A complete design and implementation guide for a fraud-prevention system that verifies whether a video submitted by a seller matches the bike in their marketplace listing photo.

---

## Table of Contents

1. [High-Level Roadmap](#1-high-level-roadmap)
2. [YOLO for Bike Detection](#2-yolo-for-bike-detection)
3. [Comparing Two Images with YOLO + OpenCV](#3-comparing-two-images-with-yolo--opencv)
4. [Full Pipeline: Video-to-Listing Verification](#4-full-pipeline-video-to-listing-verification)
5. [Visualizing Video + Bounding Boxes for Demos](#5-visualizing-video--bounding-boxes-for-demos)
6. [Integrating `make_demo_frame` into the Pipeline](#6-integrating-make_demo_frame-into-the-pipeline)
7. [Debugging `cv2.VideoWriter`](#7-debugging-cv2videowriter)
8. [Why FPS=2 Breaks VideoWriter](#8-why-fps2-breaks-videowriter)
9. [Liveness / Anti-Spoofing Detection](#9-liveness--anti-spoofing-detection)
10. [Surfacing Liveness Results in the Demo Overlay](#10-surfacing-liveness-results-in-the-demo-overlay)
11. [Adding a Title Card](#11-adding-a-title-card)

---

## 1. High-Level Roadmap

### 1. Define the Verification Goal

Decide what "match" actually means. For a first POC, aim to verify that the bike in the video is *the same physical bike* (same frame, components, wear patterns, stickers, scratches) as the one in the listing photo — not just "a bike that looks similar."

### 2. Set Up the Environment

Install Python, OpenCV (`opencv-python`), and supporting libraries (NumPy, possibly PyTorch or TensorFlow). Decide early whether processing happens on-device (mobile) or server-side. Server-side is much easier for a POC.

### 3. Capture Inputs

Two artifacts: the listing image (already uploaded) and a short verification video (e.g., 10–20 seconds panning around the bike).

### 4. Preprocess the Media

Extract frames from the video at a sensible rate (2–5 fps). Normalize images — resize, correct orientation, possibly adjust lighting/contrast. Optionally run a bike detector (YOLO) to crop out the bike.

### 5. Extract Features for Comparison

- **Classical keypoint matching (start here):** ORB, SIFT, or AKAZE with Lowe's ratio test and RANSAC.
- **Deep embeddings:** Pretrained CNN (ResNet, CLIP, DINOv2) with cosine similarity.
- **Hybrid:** Deep embeddings for coarse matching + keypoint matching for identity-level details.

### 6. Score the Match

Aggregate matches across all extracted video frames. Define a scoring function.

### 7. Add Liveness / Anti-Spoofing Checks

- Parallax and motion consistency between frames
- Multiple viewpoints present
- Metadata sanity checks

### 8. Build a Minimal Interface

Simple web form: upload listing image and video, backend runs verification, returns match score with annotated frames.

### 9. Test with Real Data

- Same-bike pairs (should match)
- Different-bikes-same-model pairs (hard case)
- Obvious mismatches

### 10. Document Limitations

Lighting sensitivity, modified bikes (new wheels, new paint), adversarial attacks, near-identical bikes.

---

## 2. YOLO for Bike Detection

### What YOLO Does

YOLO ("You Only Look Once") is a family of real-time object detection models. It returns bounding boxes with class labels and confidence scores. "bicycle" is one of the 80 default COCO classes, so bike detection works out of the box.

### The YOLO Landscape

- **Ultralytics YOLO (YOLOv8, YOLOv11, YOLOv12)** — Most popular modern implementation. Dead simple Python API. AGPL-3.0 license.
- **YOLOv5 (Ultralytics)** — Older but still widely used. AGPL-3.0.
- **YOLOv7, YOLOv9, YOLOv10** — Research releases. Various licenses.
- **YOLO-NAS (Deci AI)** — Apache 2.0 architecture.

### Licensing

- **Free to download and use for POC/research.**
- **AGPL-3.0 means deploying commercially requires open-sourcing your entire app**, or buying a commercial license from Ultralytics.
- Commercial-friendly alternatives: **YOLOX** (Apache 2.0), **Detectron2** (Apache 2.0), **Torchvision detection models** (BSD), **MMDetection** (Apache 2.0).

### Install and Basic Usage

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 'n' = nano; also s, m, l, x
results = model('listing_photo.jpg')

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == 'bicycle':
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
```

Weights auto-download on first use. No API key, no signup.

### Practical Notes

- Use nano/small variants for POC (fast on CPU)
- Default confidence 0.25; raise to 0.5 to reduce false positives
- Handle multiple bikes by picking largest bounding box
- Fine-tune only if pretrained doesn't work for your specific bike types

---

## 3. Comparing Two Images with YOLO + OpenCV

### Setup

```bash
pip install ultralytics opencv-python numpy torch torchvision open-clip-torch
```

### Approach 1: ORB Keypoint Matching (Classical, Fast)

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def detect_largest_bicycle(image_path, conf_threshold=0.4):
    """Run YOLO and return the crop of the largest bicycle bounding box."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    results = model(image, verbose=False)

    largest_box = None
    largest_area = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls] == 'bicycle' and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = (int(x1), int(y1), int(x2), int(y2))

    if largest_box is None:
        return None, None

    x1, y1, x2, y2 = largest_box
    crop = image[y1:y2, x1:x2]
    return crop, largest_box


def compare_with_orb(crop1, crop2, ratio_threshold=0.75):
    """Compare two image crops using ORB features. Returns match percentage."""
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return 0.0, len(good_matches), 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    denom = min(len(kp1), len(kp2))
    match_pct = (inliers / denom) * 100 if denom > 0 else 0.0
    return match_pct, len(good_matches), inliers


def compare_bikes(image_path1, image_path2):
    crop1, box1 = detect_largest_bicycle(image_path1)
    crop2, box2 = detect_largest_bicycle(image_path2)

    if crop1 is None or crop2 is None:
        print("No bicycle found in one or both images")
        return

    match_pct, good, inliers = compare_with_orb(crop1, crop2)
    print(f"Good matches: {good}, geometric inliers: {inliers}")
    print(f"Match percentage: {match_pct:.1f}%")
    return match_pct
```

**ORB score interpretation:**
- Same bike, similar angle: 15–60%
- Same bike, different angle: 3–15%
- Different bikes: usually <3%

### Approach 2: Visualize Matches (for debugging)

```python
def visualize_matches(crop1, crop2, output_path="matches.jpg"):
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])

    good = sorted(good, key=lambda m: m.distance)[:50]

    matched_image = cv2.drawMatches(
        crop1, kp1, crop2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, matched_image)
```

### Approach 3: CLIP Deep Embeddings (More Robust)

```python
import torch
import open_clip
from PIL import Image
import numpy as np

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model.eval()

def embed_with_clip(crop_bgr):
    """Get a CLIP embedding for a BGR image crop."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    img_tensor = clip_preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().flatten()


def compare_with_clip(crop1, crop2):
    """Cosine similarity of CLIP embeddings, scaled to 0-100%."""
    e1 = embed_with_clip(crop1)
    e2 = embed_with_clip(crop2)
    cosine_sim = float(np.dot(e1, e2))
    return max(0.0, cosine_sim) * 100
```

**CLIP score interpretation:**
- Same bike: 90–98%
- Same model different bike: 85–93% (hard case)
- Different mountain bikes: 75–88%
- Bike vs non-bike: 50–70%

### Approach 4: Combined Verdict

```python
def verify_match(image_path1, image_path2):
    crop1, _ = detect_largest_bicycle(image_path1)
    crop2, _ = detect_largest_bicycle(image_path2)

    if crop1 is None or crop2 is None:
        return {"verdict": "no_bike_detected", "orb": 0, "clip": 0}

    orb_score, _, inliers = compare_with_orb(crop1, crop2)
    clip_score = compare_with_clip(crop1, crop2)

    if clip_score > 85 and orb_score > 5:
        verdict = "likely_same_bike"
    elif clip_score > 80:
        verdict = "same_model_uncertain_identity"
    else:
        verdict = "different_bike"

    return {
        "verdict": verdict,
        "orb_match_pct": round(orb_score, 1),
        "orb_inliers": inliers,
        "clip_similarity_pct": round(clip_score, 1),
    }
```

### Practical Tips

- **Crop padding** (expand 5–10% before cropping) helps preserve edge keypoints
- **Resize to common width** (e.g., 800px) for fair comparison
- **Use multiple frames** and aggregate (max CLIP, max ORB)
- **SIFT** more accurate than ORB but slower (patent expired in 2020, now free)
- **Precompute listing embedding** — save with listing, verify only new video frames

---

## 4. Full Pipeline: Video-to-Listing Verification

Takes a `.mov` video, extracts frames, detects bikes, scores matches, aggregates results.

```python
import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import os

# Models loaded once
print("Loading YOLO...")
yolo_model = YOLO('yolov8n.pt')

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model.eval()


@dataclass
class FrameResult:
    frame_index: int
    timestamp_sec: float
    bike_detected: bool
    bike_confidence: float = 0.0
    bbox: Optional[tuple] = None
    orb_match_pct: float = 0.0
    orb_inliers: int = 0
    clip_similarity_pct: float = 0.0
    crop_path: Optional[str] = None


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


def extract_frames(video_path, target_fps=2, max_frames=60):
    """Extract frames at target_fps. Returns [(frame_index, timestamp_sec, frame_bgr), ...]"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / source_fps
    frame_stride = max(1, int(round(source_fps / target_fps)))

    print(f"Video: {duration_sec:.1f}s @ {source_fps:.1f}fps, "
          f"sampling every {frame_stride} frames")

    frames = []
    frame_idx = 0
    sampled = 0

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            timestamp = frame_idx / source_fps
            frames.append((frame_idx, timestamp, frame))
            sampled += 1
        frame_idx += 1

    cap.release()
    return frames


def detect_largest_bicycle(image_bgr, conf_threshold=0.4, padding_pct=0.05):
    """Returns (crop, bbox, confidence) or (None, None, 0)."""
    results = yolo_model(image_bgr, verbose=False)

    largest_box = None
    largest_area = 0
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo_model.names[cls] == 'bicycle' and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = (int(x1), int(y1), int(x2), int(y2))
                    best_conf = conf

    if largest_box is None:
        return None, None, 0.0

    # Pad the box slightly
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = largest_box
    pad_x = int(padding_pct * (x2 - x1))
    pad_y = int(padding_pct * (y2 - y1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2), best_conf


def resize_to_width(image, target_width=800):
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def compare_with_orb(crop1, crop2, ratio_threshold=0.75):
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return 0.0, 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    denom = min(len(kp1), len(kp2))
    match_pct = (inliers / denom) * 100 if denom > 0 else 0.0
    return match_pct, inliers


def embed_with_clip(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    img_tensor = clip_preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def cosine_similarity_pct(e1, e2):
    return max(0.0, float(np.dot(e1, e2))) * 100


def aggregate_scores(frame_results):
    """Summary statistics from per-frame results."""
    bike_frames = [f for f in frame_results if f.bike_detected]

    if not bike_frames:
        return {
            "bike_detection_rate": 0.0,
            "orb_max": 0.0, "orb_mean": 0.0, "orb_top3_mean": 0.0,
            "clip_max": 0.0, "clip_mean": 0.0, "clip_top3_mean": 0.0,
        }

    orb_scores = [f.orb_match_pct for f in bike_frames]
    clip_scores = [f.clip_similarity_pct for f in bike_frames]

    orb_top3 = sorted(orb_scores, reverse=True)[:3]
    clip_top3 = sorted(clip_scores, reverse=True)[:3]

    return {
        "bike_detection_rate": len(bike_frames) / len(frame_results) * 100,
        "orb_max": max(orb_scores),
        "orb_mean": float(np.mean(orb_scores)),
        "orb_top3_mean": float(np.mean(orb_top3)),
        "clip_max": max(clip_scores),
        "clip_mean": float(np.mean(clip_scores)),
        "clip_top3_mean": float(np.mean(clip_top3)),
    }


def make_verdict(agg):
    """Convert aggregate scores to a verdict. Tune thresholds with real data."""
    if agg["bike_detection_rate"] < 30:
        return "insufficient_bike_visibility"

    clip_top = agg["clip_top3_mean"]
    orb_top = agg["orb_top3_mean"]

    if clip_top > 88 and orb_top > 8:
        return "likely_same_bike"
    elif clip_top > 88 and orb_top > 3:
        return "probably_same_bike"
    elif clip_top > 82:
        return "same_model_identity_uncertain"
    else:
        return "different_bike"
```

**Top-3 mean** is the most useful metric: rewards videos with a few genuinely good viewpoints without being dragged down by motion-blurred frames.

### Tuning Knobs

- `target_fps` — 2 fps for 10–20s videos; 4–5 for short; 1 for long
- `max_frames` — 40–60 is usually enough
- YOLO `conf_threshold` — 0.4 default; lower to 0.25 if bikes get missed
- Verdict thresholds — calibrate against known match/mismatch pairs

### Performance Notes

CPU: ~0.5–1.5s per frame (YOLO + CLIP + ORB). 40 frames = 20–60s total. GPU reduces to a few seconds with batching.

---

## 5. Visualizing Video + Bounding Boxes for Demos

### Option 1: Ultralytics Built-In (Easiest)

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model(
    'bike_video.mov',
    save=True,           # saves to runs/detect/predict/
    classes=[1],         # COCO class 1 = bicycle
    conf=0.4,
    show=False,
)
```

### Option 2: Custom OpenCV Annotation (Most Flexible)

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def annotate_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, classes=[1], conf=0.4)

        largest_box = None
        largest_area = 0
        best_conf = 0.0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = (int(x1), int(y1), int(x2), int(y2))
                    best_conf = float(box.conf[0])

        if largest_box:
            x1, y1, x2, y2 = largest_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"BIKE {best_conf:.2f}"
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
```

### Option 3: Side-by-Side Comparison (for Demos)

```python
def make_demo_frame(video_frame, listing_crop, match_score, verdict, bbox=None):
    """Stitch listing + annotated video frame with verdict banner."""
    vh, vw = video_frame.shape[:2]
    lh, lw = listing_crop.shape[:2]
    new_lw = int(lw * (vh / lh))
    listing_resized = cv2.resize(listing_crop, (new_lw, vh))

    annotated = video_frame.copy()
    if bbox:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if "same" in verdict else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)

    combined = np.hstack([listing_resized, annotated])

    cv2.putText(combined, "LISTING PHOTO", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "VERIFICATION VIDEO", (new_lw + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    h, w = combined.shape[:2]
    banner = np.zeros((80, w, 3), dtype=np.uint8)
    banner_color = (0, 100, 0) if "same" in verdict else (0, 50, 100)
    banner[:] = banner_color

    cv2.putText(banner, f"Match: {match_score:.1f}%",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(banner, f"Verdict: {verdict}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return np.vstack([combined, banner])
```

### Option 4: Supervision Library (Roboflow)

```bash
pip install supervision
```

```python
import supervision as sv
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_scale=1.0)

def callback(frame, _):
    results = model(frame, verbose=False, classes=[1])[0]
    detections = sv.Detections.from_ultralytics(results)
    labels = [f"bike {conf:.2f}" for conf in detections.confidence]
    annotated = box_annotator.annotate(frame, detections)
    return label_annotator.annotate(annotated, detections, labels=labels)

sv.process_video(
    source_path='bike_video.mov',
    target_path='demo_output.mp4',
    callback=callback,
)
```

### Option 5: Gradio Web UI

```bash
pip install gradio
```

```python
import gradio as gr

def verify(listing_image, video):
    result = verify_video_against_listing(video, listing_image, save_crops=True)
    return "demo_output.mp4", result.verdict, result.aggregate

demo = gr.Interface(
    fn=verify,
    inputs=[gr.Image(type="filepath"), gr.Video()],
    outputs=[gr.Video(), gr.Text(label="Verdict"), gr.JSON(label="Scores")],
    title="Pinkbike BuySell Verification POC",
)
demo.launch()
```

---

## 6. Integrating `make_demo_frame` into the Pipeline

The demo frame is built **inside the per-frame loop, after scoring completes**. At that point you have: the original frame, bounding box in original coordinates, match scores, and listing crop.

### Key Design Decisions

- Demo writer is **lazy-initialized on first frame** — you don't know output dimensions until `make_demo_frame` runs once.
- Use the **original `frame`, not resized crop** — full resolution for viewing.
- `bbox` is in **original frame coordinates** (before any resizing).
- `running_verdict=True` updates verdict per-frame; False shows "analyzing..." until end.

### Integration Skeleton

```python
def verify_video_against_listing(
    video_path, listing_path,
    target_fps=2, max_frames=60,
    save_crops=False, output_dir="verification_output",
    generate_demo_video=False, demo_video_path="demo_output.mp4",
    running_verdict=True,
):
    # Setup...
    os.makedirs(output_dir, exist_ok=True)
    listing_image = cv2.imread(listing_path)
    listing_crop, _, _ = detect_largest_bicycle(listing_image)
    listing_crop = resize_to_width(listing_crop, 800)
    listing_embedding = embed_with_clip(listing_crop)

    frames = extract_frames(video_path, target_fps=target_fps, max_frames=max_frames)

    demo_writer = None
    demo_output_full_path = os.path.join(output_dir, demo_video_path)
    frame_results = []

    for i, (frame_idx, timestamp, frame) in enumerate(frames):
        crop, bbox, conf = detect_largest_bicycle(frame)

        result = FrameResult(
            frame_index=frame_idx, timestamp_sec=timestamp,
            bike_detected=crop is not None, bike_confidence=conf, bbox=bbox,
        )

        if crop is not None:
            crop_resized = resize_to_width(crop, 800)
            orb_pct, inliers = compare_with_orb(listing_crop, crop_resized)
            frame_embedding = embed_with_clip(crop_resized)
            clip_pct = cosine_similarity_pct(listing_embedding, frame_embedding)

            result.orb_match_pct = orb_pct
            result.orb_inliers = inliers
            result.clip_similarity_pct = clip_pct

        frame_results.append(result)

        # ========== DEMO VIDEO GENERATION ==========
        if generate_demo_video:
            display_score = result.clip_similarity_pct
            if running_verdict:
                interim_agg = aggregate_scores(frame_results)
                interim_verdict = (make_verdict(interim_agg)
                                   if interim_agg["bike_detection_rate"] > 0
                                   else "analyzing...")
            else:
                interim_verdict = "analyzing..."

            demo_frame = make_demo_frame(
                video_frame=frame,
                listing_crop=listing_crop,
                match_score=display_score,
                verdict=interim_verdict,
                bbox=bbox,
            )

            # Lazy init writer
            if demo_writer is None:
                demo_h, demo_w = demo_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                demo_writer = cv2.VideoWriter(
                    demo_output_full_path, fourcc, 10, (demo_w, demo_h)
                )

            demo_writer.write(demo_frame)
        # ============================================

    if demo_writer is not None:
        demo_writer.release()

    agg = aggregate_scores(frame_results)
    verdict = make_verdict(agg)

    return VerificationResult(
        video_path=video_path, listing_path=listing_path,
        total_frames_sampled=len(frame_results),
        frames_with_bike=sum(1 for f in frame_results if f.bike_detected),
        frame_results=frame_results, aggregate=agg, verdict=verdict,
    )
```

---

## 7. Debugging `cv2.VideoWriter`

### Common Cause: Frame Size Mismatch

`VideoWriter.write()` **silently fails** when frame size doesn't match constructor dimensions. Note: `VideoWriter` takes `(width, height)`, NumPy arrays are `(height, width, channels)`.

```python
if demo_writer is None:
    demo_h, demo_w = demo_frame.shape[:2]
    print(f"Demo frame shape: {demo_frame.shape}")
    print(f"Initializing writer with size: ({demo_w}, {demo_h})")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    demo_writer = cv2.VideoWriter(output_path, fourcc, 10, (demo_w, demo_h))
    if not demo_writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open")

# Every frame must match
assert demo_frame.shape[:2] == (demo_h, demo_w)
demo_writer.write(demo_frame)
```

### Codec/Container Compatibility

| Container | FourCC |
|-----------|--------|
| `.mp4` | `'mp4v'` or `'avc1'` |
| `.avi` | `'XVID'` or `'MJPG'` |
| `.mkv` | `'X264'` |

**`MJPG` to `.avi` is the most universally reliable combo.**

### Frame dtype/Channels

Must be 3-channel BGR `uint8`. `np.zeros()` defaults to `float64` — specify `dtype=np.uint8` explicitly.

### Why Green Specifically?

Codecs use YUV internally. Invalid frame data → U and V default to 128, Y stays low → bright green in RGB. "All green" = codec received something it can't interpret.

### Diagnostic Script

```python
import cv2
import numpy as np

w, h = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('test.mp4', fourcc, 10, (w, h))
print(f"Writer opened: {writer.isOpened()}")

for i in range(30):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (i * 8 % 255, 100, 200)
    cv2.putText(frame, f"Frame {i}", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    writer.write(frame)

writer.release()
```

---

## 8. Why FPS=2 Breaks VideoWriter

Many video codecs have a **minimum framerate** (usually 5–10 fps). `mp4v` in particular produces corrupted (green) output below ~5 fps.

### Why This Happens

- **Keyframe intervals are framerate-dependent** — at 2 fps, 2s keyframe interval = 4 frames, breaks buffering assumptions.
- **Rate control divides by fps** — edge cases at low fps cause overflow/underflow.
- **Timestamp quantization bugs** in some encoder implementations.
- **OpenCV codec backend is system-dependent** (FFmpeg, GStreamer, AVFoundation, Media Foundation).

### Fix 1: Write at Higher FPS, Duplicate Frames (Recommended)

```python
PLAYBACK_FPS = 10
HOLD_FRAMES = 5  # each scored frame held 0.5s at 10fps

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
demo_writer = cv2.VideoWriter(path, fourcc, PLAYBACK_FPS, (demo_w, demo_h))

for _ in range(HOLD_FRAMES):
    demo_writer.write(demo_frame)
```

### Fix 2: Use MJPG/AVI (Frame-Independent Codec)

```python
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
demo_writer = cv2.VideoWriter(path.replace('.mp4', '.avi'),
                              fourcc, 2, (demo_w, demo_h))
```

Works at any fps. Larger files, less universally playable.

### Fix 3: Try avc1 (H.264)

```python
fourcc = cv2.VideoWriter_fourcc(*'avc1')
```

More tolerant than `mp4v` when available.

### Broader Lesson

`VideoWriter.isOpened()` returning `True` only means file creation succeeded, not that codec accepted settings. For unusual configs, test with a short clip first. For bulletproof low-fps output, consider `imageio-ffmpeg` or shell out to `ffmpeg` directly.

---

## 9. Liveness / Anti-Spoofing Detection

### Part 1: Detecting Real 3D Motion (Parallax)

**Key insight:** Flat scenes (photo of a photo) have every pixel moving together. Real 3D scenes have different parts moving at different rates.

### Method A: Homography Residual Test (Best for POC)

If a single homography fits frame N to frame N+1 with high inlier ratio, the scene is probably flat.

```python
def homography_fit_error(frame1_crop, frame2_crop):
    """Returns flatness score 0-1. High = likely flat image."""
    gray1 = cv2.cvtColor(frame1_crop, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_crop, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < 20:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None

    inlier_ratio = mask.sum() / len(mask)
    return float(inlier_ratio)

# Interpretation:
# > 0.85 -> suspiciously flat (possible spoof)
# 0.5-0.85 -> normal 3D scene
# < 0.5 -> too 3D or too much motion/blur
```

### Method B: Fundamental Matrix vs Homography

```python
def geometric_consistency_test(frame1_crop, frame2_crop):
    # ... keypoint extraction ...
    H, H_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    F, F_mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0)

    if H is None or F is None:
        return None

    H_inliers = H_mask.sum() / len(H_mask)
    F_inliers = F_mask.sum() / len(F_mask)
    flatness = H_inliers / max(F_inliers, 0.01)

    return {
        "homography_inlier_ratio": float(H_inliers),
        "fundamental_inlier_ratio": float(F_inliers),
        "flatness_ratio": float(flatness),
        "likely_flat": flatness > 0.95,
    }
```

### Method C: Optical Flow Divergence

```python
def optical_flow_variance(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mean_flow = magnitude.mean()
    if mean_flow < 1.0:
        return None

    cv_score = magnitude.std() / mean_flow
    return float(cv_score)

# < 0.15 -> uniform flow, possibly flat
# > 0.3 -> lots of variation, real 3D
```

### Method D: Structure from Motion (Heavy)

Use COLMAP, OpenSfM, or pycolmap. Overkill for POC.

### Part 2: Viewpoint Coverage

### Embedding Clustering

```python
from sklearn.cluster import KMeans

def analyze_viewpoint_coverage(frame_embeddings, n_expected_views=4):
    if len(frame_embeddings) < n_expected_views:
        return {"sufficient_coverage": False, "reason": "too_few_frames"}

    embeddings = np.array(frame_embeddings)
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    mask = ~np.eye(n, dtype=bool)
    mean_pairwise_sim = sim_matrix[mask].mean()

    kmeans = KMeans(n_clusters=min(n_expected_views, n), n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    cluster_sizes = np.bincount(labels)

    centroids = kmeans.cluster_centers_
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    inter_cluster_sim = (centroids @ centroids.T)
    inter_cluster_sim_mean = inter_cluster_sim[~np.eye(len(centroids), dtype=bool)].mean()

    return {
        "mean_pairwise_similarity": float(mean_pairwise_sim),
        "inter_cluster_similarity": float(inter_cluster_sim_mean),
        "cluster_sizes": cluster_sizes.tolist(),
        "likely_static": mean_pairwise_sim > 0.96,
        "sufficient_coverage": mean_pairwise_sim < 0.94 and min(cluster_sizes) >= 2,
    }
```

### Zero-Shot View Classification (CLIP)

```python
def classify_bike_view(crop_bgr, clip_model, clip_preprocess, tokenizer):
    view_prompts = [
        "a photo of a mountain bike from the side",
        "a photo of a mountain bike from the front",
        "a photo of a mountain bike from the back",
        "a close-up of a bicycle drivetrain and gears",
        "a close-up of bicycle handlebars",
        "a close-up of a bicycle wheel",
    ]

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = clip_preprocess(Image.fromarray(crop_rgb)).unsqueeze(0)
    text_tokens = tokenizer(view_prompts)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)

    best_idx = similarity.argmax().item()
    return {
        "view": view_prompts[best_idx].replace("a photo of a mountain bike from the ", "")
                                     .replace("a close-up of ", ""),
        "confidence": float(similarity[best_idx]),
        "all_scores": dict(zip(view_prompts, similarity.tolist())),
    }
```

### Combined Liveness Assessment

```python
@dataclass
class LivenessResult:
    motion_consistency: dict = field(default_factory=dict)
    viewpoint_coverage: dict = field(default_factory=dict)
    verdict: str = ""
    is_likely_live: bool = False


def assess_liveness(frame_results, frame_crops, frame_embeddings,
                    original_frames, clip_tools=None):
    # Motion/parallax
    bike_frames = [(i, fr) for i, fr in enumerate(frame_results) if fr.bike_detected]
    flatness_scores = []

    for i in range(len(bike_frames) - 2):
        idx_a = bike_frames[i][0]
        idx_b = bike_frames[i + 2][0]
        if idx_a >= len(frame_crops) or idx_b >= len(frame_crops):
            continue
        score = homography_fit_error(frame_crops[idx_a], frame_crops[idx_b])
        if score is not None:
            flatness_scores.append(score)

    if flatness_scores:
        avg_flatness = float(np.mean(flatness_scores))
        motion_result = {
            "mean_homography_inlier_ratio": avg_flatness,
            "samples": len(flatness_scores),
            "likely_flat_image": avg_flatness > 0.85,
        }
    else:
        motion_result = {"error": "insufficient_frame_pairs"}

    coverage_result = analyze_viewpoint_coverage(frame_embeddings)

    if clip_tools and frame_crops:
        view_result = assess_view_coverage(frame_crops, *clip_tools)
        coverage_result["view_classification"] = view_result

    is_flat = motion_result.get("likely_flat_image", False)
    is_static = coverage_result.get("likely_static", False)
    has_coverage = coverage_result.get("sufficient_coverage", False)

    if is_flat:
        verdict = "likely_spoof_flat_image"
    elif is_static and not has_coverage:
        verdict = "insufficient_viewpoints"
    elif has_coverage and not is_flat:
        verdict = "live_multi_view_capture"
    else:
        verdict = "ambiguous"

    return LivenessResult(
        motion_consistency=motion_result,
        viewpoint_coverage=coverage_result,
        verdict=verdict,
        is_likely_live=verdict == "live_multi_view_capture",
    )
```

### Calibration Needed

Collect real data to tune thresholds:
- 20 real walkaround videos
- 20 spoof videos (phone filming a photo/screen, static shots)
- 10 edge cases (low light, hand shake only, busy backgrounds)

### Priority

Method A (homography residual) has the best effort-to-payoff ratio. 30 lines of code, catches the dominant spoof attack.

---

## 10. Surfacing Liveness Results in the Demo Overlay

### Enhanced Demo Frame with Status Panel

```python
# Color constants (BGR)
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 50, 200)
COLOR_YELLOW = (0, 200, 220)
COLOR_GRAY = (120, 120, 120)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK = (30, 30, 30)


def status_color(status):
    if status in ("pass", "live", "3d"):
        return COLOR_GREEN
    elif status in ("fail", "spoof", "flat"):
        return COLOR_RED
    elif status in ("pending", "analyzing", "unknown"):
        return COLOR_YELLOW
    return COLOR_GRAY


def draw_status_pill(img, x, y, label, value, status, width=220, height=44):
    color = status_color(status)
    cv2.rectangle(img, (x, y), (x + width, y + height), COLOR_DARK, -1)
    cv2.rectangle(img, (x, y), (x + 8, y + height), color, -1)
    cv2.putText(img, label, (x + 18, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
    cv2.putText(img, value, (x + 18, y + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def make_demo_frame_v2(
    video_frame, listing_crop,
    bbox=None, match_score=None,
    liveness_state=None, view_state=None,
    final_verdict=None,
):
    LISTING_WIDTH = 300
    vh, vw = video_frame.shape[:2]
    lh, lw = listing_crop.shape[:2]
    listing_h = int(lh * (LISTING_WIDTH / lw))
    listing_resized = cv2.resize(listing_crop, (LISTING_WIDTH, min(listing_h, vh)))

    if listing_resized.shape[0] < vh:
        pad = np.zeros((vh - listing_resized.shape[0], LISTING_WIDTH, 3), dtype=np.uint8)
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

    PANEL_WIDTH = 260
    panel = np.zeros((vh, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = COLOR_DARK

    cv2.putText(panel, "ANALYSIS", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    cv2.line(panel, (18, 38), (PANEL_WIDTH - 18, 38), COLOR_GRAY, 1)

    y_cursor = 54

    if match_score:
        clip_val = match_score.get("clip", 0)
        orb_val = match_score.get("orb", 0)
        match_status = "pass" if clip_val > 85 else "pending" if clip_val > 75 else "fail"
        draw_status_pill(panel, 18, y_cursor,
                         "VISUAL MATCH (CLIP)", f"{clip_val:.1f}%",
                         match_status, width=PANEL_WIDTH - 36)
        y_cursor += 54
        draw_status_pill(panel, 18, y_cursor,
                         "IDENTITY MATCH (ORB)", f"{orb_val:.1f}%",
                         "pass" if orb_val > 5 else ("pending" if orb_val > 2 else "fail"),
                         width=PANEL_WIDTH - 36)
        y_cursor += 64

    if liveness_state:
        status = liveness_state.get("status", "analyzing")
        ratio = liveness_state.get("homography_inlier_ratio")
        if status == "3d":
            label_val = "LIVE 3D SCENE"
        elif status == "flat":
            label_val = "FLAT (SPOOF?)"
        elif ratio is not None:
            label_val = f"H-ratio: {ratio:.2f}"
        else:
            label_val = "analyzing..."
        draw_status_pill(panel, 18, y_cursor,
                         "LIVENESS", label_val, status,
                         width=PANEL_WIDTH - 36)
        y_cursor += 64

    if view_state:
        seen = set(view_state.get("views_seen", []))
        required = set(view_state.get("required", ["front", "side", "back"]))
        covered = seen & required
        coverage_text = f"{len(covered)}/{len(required)} views"
        vstatus = "pass" if covered >= required else ("pending" if covered else "fail")
        draw_status_pill(panel, 18, y_cursor,
                         "VIEW COVERAGE", coverage_text, vstatus,
                         width=PANEL_WIDTH - 36)
        y_cursor += 54

        for view_name in sorted(required):
            seen_it = view_name in seen
            symbol = "[X]" if seen_it else "[ ]"
            color = COLOR_GREEN if seen_it else COLOR_GRAY
            cv2.putText(panel, f"{symbol} {view_name}",
                        (28, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_cursor += 22

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

    BANNER_H = 56
    banner = np.zeros((BANNER_H, total_w, 3), dtype=np.uint8)
    if final_verdict:
        if "same_bike" in final_verdict and "spoof" not in final_verdict:
            banner_color = (0, 100, 0)
            verdict_text = f"[PASS] VERIFIED: {final_verdict}"
        elif "spoof" in final_verdict or "fail" in final_verdict:
            banner_color = (0, 0, 120)
            verdict_text = f"[FAIL] FLAGGED: {final_verdict}"
        else:
            banner_color = (60, 60, 60)
            verdict_text = f"[?] {final_verdict}"
        banner[:] = banner_color
        cv2.putText(banner, verdict_text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
    else:
        banner[:] = COLOR_DARK
        cv2.putText(banner, "Analyzing video...", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    return np.vstack([header, combined, banner])
```

### Incremental Liveness Accumulator

```python
@dataclass
class LivenessAccumulator:
    homography_ratios: list = field(default_factory=list)
    views_seen: set = field(default_factory=set)
    previous_crop: Optional[np.ndarray] = None
    previous_crop_timestamp: float = -999.0
    pair_interval_sec: float = 0.8

    def update_motion(self, crop, timestamp):
        if self.previous_crop is not None and \
           (timestamp - self.previous_crop_timestamp) >= self.pair_interval_sec:
            ratio = homography_fit_error(self.previous_crop, crop)
            if ratio is not None:
                self.homography_ratios.append(ratio)
            self.previous_crop = crop
            self.previous_crop_timestamp = timestamp
        elif self.previous_crop is None:
            self.previous_crop = crop
            self.previous_crop_timestamp = timestamp
        return self.current_motion_state()

    def current_motion_state(self):
        if not self.homography_ratios:
            return {"status": "analyzing", "homography_inlier_ratio": None}
        avg = float(np.mean(self.homography_ratios))
        if len(self.homography_ratios) < 2:
            status = "analyzing"
        elif avg > 0.85:
            status = "flat"
        else:
            status = "3d"
        return {
            "status": status,
            "homography_inlier_ratio": avg,
            "samples": len(self.homography_ratios),
        }

    def update_views(self, crop, clip_tools):
        clip_model, clip_preprocess, tokenizer = clip_tools
        result = classify_bike_view(crop, clip_model, clip_preprocess, tokenizer)
        if result["confidence"] > 0.22:
            view = result["view"].split()[-1] if result["view"] else "unknown"
            if view in ("drivetrain", "gears", "handlebars", "wheel"):
                self.views_seen.add("detail")
            else:
                self.views_seen.add(view)
        return self.current_view_state()

    def current_view_state(self):
        return {
            "views_seen": list(self.views_seen),
            "required": ["front", "side", "back"],
        }
```

### Wired Into the Main Loop

```python
def verify_video_against_listing(
    video_path, listing_path,
    target_fps=2, max_frames=60,
    generate_demo_video=False, demo_video_path="demo_output.mp4",
    output_dir="verification_output",
):
    os.makedirs(output_dir, exist_ok=True)
    listing_image = cv2.imread(listing_path)
    listing_crop, _, _ = detect_largest_bicycle(listing_image)
    listing_crop = resize_to_width(listing_crop, 800)
    listing_embedding = embed_with_clip(listing_crop)

    frames = extract_frames(video_path, target_fps=target_fps, max_frames=max_frames)

    frame_results = []
    liveness = LivenessAccumulator()
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_tools = (clip_model, clip_preprocess, clip_tokenizer)
    demo_writer = None
    demo_output_full_path = os.path.join(output_dir, demo_video_path)

    DEMO_FPS = 10
    DEMO_HOLD_FRAMES = 5

    for i, (frame_idx, timestamp, frame) in enumerate(frames):
        crop, bbox, conf = detect_largest_bicycle(frame)

        match_score = None
        motion_state = liveness.current_motion_state()
        view_state = liveness.current_view_state()

        result = FrameResult(
            frame_index=frame_idx, timestamp_sec=timestamp,
            bike_detected=crop is not None, bike_confidence=conf, bbox=bbox,
        )

        if crop is not None:
            crop_resized = resize_to_width(crop, 800)
            orb_pct, inliers = compare_with_orb(listing_crop, crop_resized)
            frame_embedding = embed_with_clip(crop_resized)
            clip_pct = cosine_similarity_pct(listing_embedding, frame_embedding)
            result.orb_match_pct = orb_pct
            result.orb_inliers = inliers
            result.clip_similarity_pct = clip_pct

            match_score = {"clip": clip_pct, "orb": orb_pct}

            motion_state = liveness.update_motion(crop_resized, timestamp)
            view_state = liveness.update_views(crop_resized, clip_tools)

        frame_results.append(result)

        if generate_demo_video:
            is_last = (i == len(frames) - 1)
            final = None
            if is_last:
                agg = aggregate_scores(frame_results)
                match_verdict = make_verdict(agg)
                if motion_state["status"] == "flat":
                    final = "likely_spoof_flat_image"
                elif len(liveness.views_seen & {"front", "side", "back"}) < 2:
                    final = "insufficient_viewpoints"
                else:
                    final = match_verdict

            demo_frame = make_demo_frame_v2(
                video_frame=frame, listing_crop=listing_crop, bbox=bbox,
                match_score=match_score, liveness_state=motion_state,
                view_state=view_state, final_verdict=final,
            )

            if demo_writer is None:
                h, w = demo_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                demo_writer = cv2.VideoWriter(
                    demo_output_full_path, fourcc, DEMO_FPS, (w, h)
                )
                if not demo_writer.isOpened():
                    raise RuntimeError("Could not open demo video writer")

            hold = DEMO_HOLD_FRAMES * (15 if is_last else 1)
            for _ in range(hold):
                demo_writer.write(demo_frame)

    if demo_writer is not None:
        demo_writer.release()

    agg = aggregate_scores(frame_results)
    match_verdict = make_verdict(agg)

    return VerificationResult(
        video_path=video_path, listing_path=listing_path,
        total_frames_sampled=len(frame_results),
        frames_with_bike=sum(1 for f in frame_results if f.bike_detected),
        frame_results=frame_results, aggregate=agg, verdict=match_verdict,
        liveness={
            "motion": liveness.current_motion_state(),
            "views": liveness.current_view_state(),
        },
    )
```

### Demo Tips

- Record real bike + spoof attempt (photo of photo), run both, show side-by-side
- Bounding box color change (yellow → green/red) is the most visually obvious signal
- Hold final frame for ~7.5 seconds so viewers can read the verdict

---

## 11. Adding a Title Card

### Title Card Builder

```python
from datetime import datetime
import os

def make_title_card(
    width, height,
    listing_path, video_path,
    subtitle="Video Verification POC",
    title="Pinkbike BuySell",
):
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (20, 20, 28)

    # Title
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.8
    title_thickness = 3
    (tw, th), _ = cv2.getTextSize(title, title_font, title_scale, title_thickness)
    tx = (width - tw) // 2
    ty = height // 2 - 40
    cv2.putText(card, title, (tx, ty), title_font, title_scale,
                (255, 255, 255), title_thickness, cv2.LINE_AA)

    # Subtitle
    sub_font = cv2.FONT_HERSHEY_SIMPLEX
    sub_scale = 0.8
    sub_thickness = 2
    (sw, sh), _ = cv2.getTextSize(subtitle, sub_font, sub_scale, sub_thickness)
    sx = (width - sw) // 2
    sy = ty + 40
    cv2.putText(card, subtitle, (sx, sy), sub_font, sub_scale,
                (180, 180, 200), sub_thickness, cv2.LINE_AA)

    # Separator
    line_y = sy + 30
    cv2.line(card, (width // 2 - 80, line_y), (width // 2 + 80, line_y),
             (100, 140, 200), 2, cv2.LINE_AA)

    # Metadata
    meta_color = (160, 160, 170)
    meta_y = height - 60
    cv2.putText(card, f"Listing: {os.path.basename(listing_path)}", (30, meta_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, meta_color, 1, cv2.LINE_AA)
    cv2.putText(card, f"Video:   {os.path.basename(video_path)}", (30, meta_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, meta_color, 1, cv2.LINE_AA)
    cv2.putText(card, f"Run:     {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                (30, meta_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, meta_color, 1, cv2.LINE_AA)

    return card
```

### Integrating into the Pipeline

Title card is written **after the writer is initialized** (so dimensions match) but **before the first real demo frame** (so it plays first).

```python
# Constants
DEMO_FPS = 10
DEMO_HOLD_FRAMES = 5
TITLE_CARD_SECONDS = 2.5
TITLE_FADE_SECONDS = 0.3

# Inside the loop, at demo writer initialization:
if demo_writer is None:
    h, w = demo_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    demo_writer = cv2.VideoWriter(
        demo_output_full_path, fourcc, DEMO_FPS, (w, h)
    )
    if not demo_writer.isOpened():
        raise RuntimeError("Could not open demo video writer")

    # TITLE CARD
    title_card = make_title_card(
        width=w, height=h,
        listing_path=listing_path,
        video_path=video_path,
    )
    title_frames_total = int(TITLE_CARD_SECONDS * DEMO_FPS)
    for _ in range(title_frames_total):
        demo_writer.write(title_card)
```

### Why Build the Title Card After the First Demo Frame?

`VideoWriter` requires exact dimensions. The demo frame dimensions depend on the input video frame. Building a real demo frame first, then initializing the writer to match, then writing the title card at those same dimensions avoids the green-frame corruption problem.

### Optional Fade-In

```python
title_card = make_title_card(width=w, height=h,
                              listing_path=listing_path, video_path=video_path)

fade_frames = int(TITLE_FADE_SECONDS * DEMO_FPS)
hold_frames = int(TITLE_CARD_SECONDS * DEMO_FPS) - fade_frames

# Fade from black
for i in range(fade_frames):
    alpha = (i + 1) / fade_frames
    faded = (title_card * alpha).astype(np.uint8)
    demo_writer.write(faded)

# Hold at full opacity
for _ in range(hold_frames):
    demo_writer.write(title_card)
```

### Preview Without Running Full Pipeline

```python
if __name__ == "__main__":
    card = make_title_card(
        width=1460, height=700,
        listing_path="test_listing.jpg",
        video_path="test_video.mov",
    )
    cv2.imshow("Title Card Preview", card)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

## Summary: Project Status

At the end of this session, the POC includes:

- **YOLO-based bike detection** (largest bike in frame, with padding)
- **ORB keypoint matching** for identity-level verification
- **CLIP embeddings** for robust semantic similarity
- **Video frame extraction** from `.mov` files at configurable fps
- **Score aggregation** using max, mean, and top-3 mean
- **Full demo video generation** with side-by-side listing/video layout
- **Status panel overlay** showing match, liveness, view coverage
- **Liveness detection** via homography residual (flat-image spoof detection)
- **Viewpoint coverage** via CLIP zero-shot view classification
- **Title card** with metadata and optional fade-in

## Next Steps / Open Items

Possible directions to continue:

1. **Side-by-side spoof comparison demo** — run real and spoof videos in parallel, produce a comparison video
2. **End card** with summary stats (total frames, detection rate, final scores)
3. **Drawing ORB keypoint match lines** between listing and frame as overlay
4. **Small running score chart** in the corner of the demo video
5. **Gradio web UI** wrapping the full pipeline
6. **Calibration harness** — run pipeline against labeled dataset (real/spoof/same-model), tune thresholds
7. **Fine-tuning YOLO** on mountain bike specific data if generic detection misses full-suspension frames
8. **Store reference embeddings** with listings so verification only requires new video processing
9. **Optical flow liveness check** (Method C) as an additional signal
10. **Metadata checks** on uploaded videos (duration, resolution, EXIF)
11. **Per-view capture guidance** — tell user "please film the drivetrain" based on missing views
12. **Production licensing migration** — swap Ultralytics YOLO for YOLOX/Torchvision if commercializing

## Dependencies

```bash
pip install ultralytics opencv-python numpy torch torchvision open-clip-torch scikit-learn
```

Optional:
- `supervision` (Roboflow annotation library)
- `gradio` (web UI)
- `imageio-ffmpeg` (alternative video writer for unusual fps/codec needs)
