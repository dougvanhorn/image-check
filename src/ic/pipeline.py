import os
import pathlib

import cv2
import numpy as np
import torch
import open_clip

from PIL import Image
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional

from ic import settings
from ic import utils


# ----- Models loaded once at module level -----
print("Loading YOLO...")
yolo_model = YOLO(settings.YOLO_26N)
model = yolo_model

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)
clip_model.eval()


# ----- Data classes for clean result handling -----
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


# ----- Step 1: Extract frames from .mov -----
def extract_frames(video_path, target_fps=2, max_frames=60):
    """
    Extract frames at target_fps from a video. Returns list of (frame_index, timestamp_sec, frame_bgr).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / source_fps

    # How many source frames to skip between samples
    frame_stride = max(1, int(round(source_fps / target_fps)))

    print(f"Video: {duration_sec:.1f}s @ {source_fps:.1f}fps, "
          f"sampling every {frame_stride} frames (~{target_fps}fps)")

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
    print(f"Extracted {len(frames)} frames")
    return frames


# ----- Step 2: Detect largest bike with optional padding -----
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
    return crop, (x1, y1, x2, y2), best_conf


# ----- Step 3: Resize crops to a common width for fair comparison -----
def resize_to_width(image, target_width=800):
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


# ----- Step 4: ORB matching -----
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


# ----- Step 5: CLIP embeddings (precompute listing embedding once) -----
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


# ----- Step 6: Aggregate scores across all frames -----
def aggregate_scores(frame_results):
    """Pull useful summary statistics from per-frame results."""
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
    """Convert aggregate scores to a human verdict. Tune thresholds with real data."""
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


_cached_frames = None

# ----- Main pipeline -----
def verify_video_against_listing_nodemovideo(
    video_path: pathlib.Path,
    listing_path: pathlib.Path,
    target_fps: int = 2,
    max_frames: int = 60,
    save_crops: bool = False,
    output_dir: str = "verification_output",
) -> VerificationResult:
    """
    Verify whether the bike in a video matches the bike in a listing image.
    """
    if save_crops:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: Process the listing image once
    print(f"\nProcessing listing: {listing_path}")
    listing_image = cv2.imread(listing_path)
    if listing_image is None:
        raise FileNotFoundError(listing_path)

    listing_crop, listing_box, listing_conf = detect_largest_bicycle(listing_image)
    if listing_crop is None:
        raise ValueError("No bicycle detected in listing image")

    listing_crop = resize_to_width(listing_crop, 800)
    listing_embedding = embed_with_clip(listing_crop)
    print(f"Listing bike detected (conf {listing_conf:.2f}), "
          f"crop size {listing_crop.shape[1]}x{listing_crop.shape[0]}")

    if save_crops:
        cv2.imwrite(os.path.join(output_dir, "listing_crop.jpg"), listing_crop)

    # Step 2: Extract frames from video
    print(f"\nProcessing video: {video_path}")
    frames = extract_frames(video_path, target_fps=target_fps, max_frames=max_frames)

    # Step 3: Process each frame
    frame_results = []
    for i, (frame_idx, timestamp, frame) in enumerate(frames):
        crop, bbox, conf = detect_largest_bicycle(frame)

        result = FrameResult(
            frame_index=frame_idx,
            timestamp_sec=timestamp,
            bike_detected=crop is not None,
            bike_confidence=conf,
            bbox=bbox,
        )

        if crop is not None:
            crop = resize_to_width(crop, 800)
            orb_pct, inliers = compare_with_orb(listing_crop, crop)
            frame_embedding = embed_with_clip(crop)
            clip_pct = cosine_similarity_pct(listing_embedding, frame_embedding)

            result.orb_match_pct = orb_pct
            result.orb_inliers = inliers
            result.clip_similarity_pct = clip_pct

            if save_crops:
                crop_path = os.path.join(output_dir, f"frame_{i:03d}_t{timestamp:.1f}s.jpg")
                cv2.imwrite(crop_path, crop)
                result.crop_path = crop_path

        frame_results.append(result)
        marker = "✓" if result.bike_detected else "✗"
        print(f"  Frame {i+1}/{len(frames)} (t={timestamp:5.1f}s) {marker} "
              f"bike_conf={conf:.2f} orb={result.orb_match_pct:5.1f}% "
              f"clip={result.clip_similarity_pct:5.1f}%")

    # Step 4: Aggregate
    agg = aggregate_scores(frame_results)
    verdict = make_verdict(agg)

    return VerificationResult(
        video_path=video_path,
        listing_path=listing_path,
        total_frames_sampled=len(frame_results),
        frames_with_bike=sum(1 for f in frame_results if f.bike_detected),
        frame_results=frame_results,
        aggregate=agg,
        verdict=verdict,
    )

# ----- Modified main pipeline with demo video generation -----
# This is the main loop.
def verify_video_against_listing(
    video_path: str,
    listing_path: str,
    target_fps: int = 2,
    max_frames: int = 60,
    save_crops: bool = False,
    output_dir: str = "verification_output",
    generate_demo_video: bool = False,        # NEW
    demo_video_path: str = "demo_output.mp4", # NEW
    running_verdict: bool = True,             # NEW: update verdict per frame vs final
) -> VerificationResult:
    """
    Verify whether the bike in a video matches the bike in a listing image.
    Optionally generate a side-by-side demo video showing the matching process.
    """
    if save_crops or generate_demo_video:
        os.makedirs(output_dir, exist_ok=True)

    # ===== Process listing image once =====
    print(f"\nProcessing listing: {listing_path}")
    listing_image = cv2.imread(listing_path)
    if listing_image is None:
        raise FileNotFoundError(listing_path)

    listing_crop, listing_box, listing_conf = detect_largest_bicycle(listing_image)
    if listing_crop is None:
        raise ValueError("No bicycle detected in listing image")

    listing_crop = resize_to_width(listing_crop, 800)
    listing_embedding = embed_with_clip(listing_crop)

    if save_crops:
        cv2.imwrite(os.path.join(output_dir, "listing_crop.jpg"), listing_crop)

    # ===== Extract frames =====
    print(f"\nProcessing video: {video_path}")
    frames = extract_frames(video_path, target_fps=target_fps, max_frames=max_frames)

    # ===== Set up demo video writer if requested =====
    # NOTE: we need to know the output frame size BEFORE creating the writer.
    # Easiest way is to build the first demo frame, then initialize writer with its size.
    demo_writer = None
    demo_output_full_path = os.path.join(output_dir, demo_video_path) if generate_demo_video else None
    # We use this to get past fps limitations in the mp4 codec.
    PLAYBACK_FPS = 10.0  # Show each scored frame for 0.5 seconds
    HOLD_FRAMES = 5

    # ===== Process each frame =====
    frame_results = []
    for i, (frame_idx, timestamp, frame) in enumerate(frames):
        crop, bbox, conf = detect_largest_bicycle(frame)

        result = FrameResult(
            frame_index=frame_idx,
            timestamp_sec=timestamp,
            bike_detected=crop is not None,
            bike_confidence=conf,
            bbox=bbox,
        )

        if crop is not None:
            crop_resized = resize_to_width(crop, 800)
            orb_pct, inliers = compare_with_orb(listing_crop, crop_resized)
            frame_embedding = embed_with_clip(crop_resized)
            clip_pct = cosine_similarity_pct(listing_embedding, frame_embedding)

            result.orb_match_pct = orb_pct
            result.orb_inliers = inliers
            result.clip_similarity_pct = clip_pct

            if save_crops:
                crop_path = os.path.join(output_dir, f"frame_{i:03d}_t{timestamp:.1f}s.jpg")
                cv2.imwrite(crop_path, crop_resized)
                result.crop_path = crop_path

        frame_results.append(result)

        # ========== DEMO VIDEO GENERATION HAPPENS HERE ==========
        # Right after we've scored this frame, while we still have:
        # - the original `frame` (full resolution, untouched)
        # - the `bbox` in original frame coordinates
        # - the `clip_pct` and `orb_pct` for THIS frame
        # - the `listing_crop` already loaded
        if generate_demo_video:
            # Pick which score to display. CLIP is more visually intuitive
            # because it doesn't drop to zero on bad frames.
            display_score = result.clip_similarity_pct

            # Decide on a per-frame label. With running_verdict=True we recompute
            # an interim verdict using only the frames seen so far.
            if running_verdict:
                interim_agg = aggregate_scores(frame_results)
                interim_verdict = make_verdict(interim_agg) if interim_agg["bike_detection_rate"] > 0 else "analyzing..."
            else:
                interim_verdict = "analyzing..."

            demo_frame = make_demo_frame(
                video_frame=frame,           # the ORIGINAL frame, not the crop
                listing_crop=listing_crop,
                match_score=display_score,
                verdict=interim_verdict,
                bbox=bbox,                   # in original frame coordinates
            )
            # cv2.imshow('Demo Frame', demo_frame)
            # cv2.waitKey(3000)  # Show the final frame for a moment

            # Initialize writer on the first frame, now that we know the output size
            if demo_writer is None:
                demo_h, demo_w = demo_frame.shape[:2]
                print(f"Demo frame shape: {demo_frame.shape}")  # (H, W, 3)
                print(f"Initializing writer with size: ({demo_w}, {demo_h})")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Lower playback fps so each scored frame is visible (2 fps = each frame shown for 0.5s)
                # Bump to 4-5 for snappier demos.
                demo_writer = cv2.VideoWriter(
                    demo_output_full_path,
                    fourcc,
                    PLAYBACK_FPS,  # fps
                    (demo_w, demo_h)
                )
                if not demo_writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {demo_output_full_path}")

            assert demo_frame.shape[:2] == (demo_h, demo_w), \
                f"Frame size {demo_frame.shape[:2]} != writer size {(demo_h, demo_w)}"

            # print('Writing frame to demo file.')
            # cv2.imshow('Demo Frame', demo_frame)
            # cv2.waitKey(1000)  # Show each frame for a moment (1000 ms = 1 second)

            # We'll write the image multiple times at the higher framerate to keep it on screen
            # longer.
            for _ in range(HOLD_FRAMES):
                demo_writer.write(demo_frame)
        # ==========================================================

        marker = "✓" if result.bike_detected else "✗"
        print(f"  Frame {i+1}/{len(frames)} (t={timestamp:5.1f}s) {marker} "
              f"orb={result.orb_match_pct:5.1f}% clip={result.clip_similarity_pct:5.1f}%")

    # ===== Finalize demo video =====
    if demo_writer is not None:
        demo_writer.release()
        print(f"\nDemo video saved to {demo_output_full_path}")

    # ===== Aggregate final scores =====
    agg = aggregate_scores(frame_results)
    verdict = make_verdict(agg)

    # ===== Optional: append a "FINAL VERDICT" hold-frame to the demo =====
    # Useful so the demo video ends on a clear summary screen rather than the last frame.
    # if generate_demo_video and frames:
    #     append_final_verdict_frames(
    #         demo_output_full_path, listing_crop, frames[-1][2], frame, bbox,
    #         agg, verdict, hold_seconds=3
    #     )

    return VerificationResult(
        video_path=video_path,
        listing_path=listing_path,
        total_frames_sampled=len(frame_results),
        frames_with_bike=sum(1 for f in frame_results if f.bike_detected),
        frame_results=frame_results,
        aggregate=agg,
        verdict=verdict,
    )


def append_final_verdict_frames(
    video_path, listing_crop, last_frame, last_video_frame, last_bbox,
    agg, verdict, hold_seconds=3, fps=2
):
    """Append a few seconds of a static 'final verdict' frame to the end of the demo."""
    # Reopen the video to append (simpler: read all frames and rewrite)
    cap = cv2.VideoCapture(video_path)
    existing_frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        existing_frames.append(f)
    cap.release()

    if not existing_frames:
        return

    h, w = existing_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for f in existing_frames:
        writer.write(f)

    # Build the verdict hold frame
    final_frame = make_demo_frame(
        video_frame=last_video_frame,
        listing_crop=listing_crop,
        match_score=agg["clip_top3_mean"],
        verdict=f"FINAL: {verdict}",
        bbox=last_bbox,
    )

    for _ in range(hold_seconds * fps):
        writer.write(final_frame)

    writer.release()


def print_report(result: VerificationResult):
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(f"Listing:  {result.listing_path}")
    print(f"Video:    {result.video_path}")
    print(f"Frames sampled:        {result.total_frames_sampled}")
    print(f"Frames with bike:      {result.frames_with_bike} "
          f"({result.aggregate['bike_detection_rate']:.0f}%)")
    print()
    print("ORB keypoint matching (identity-level):")
    print(f"  max:        {result.aggregate['orb_max']:.1f}%")
    print(f"  mean:       {result.aggregate['orb_mean']:.1f}%")
    print(f"  top-3 mean: {result.aggregate['orb_top3_mean']:.1f}%")
    print()
    print("CLIP semantic similarity:")
    print(f"  max:        {result.aggregate['clip_max']:.1f}%")
    print(f"  mean:       {result.aggregate['clip_mean']:.1f}%")
    print(f"  top-3 mean: {result.aggregate['clip_top3_mean']:.1f}%")
    print()
    print(f"VERDICT: {result.verdict}")
    print("=" * 60)

"""
Additional Notes from Claude:

## What This Does, Step by Step
The pipeline reads the listing image once, runs YOLO to find the largest bike, crops it, and
computes its CLIP embedding. Then it walks through the video at roughly 2 frames per second
(configurable), running YOLO on each frame to find the largest bike there. For each frame where a
bike is detected, it runs both ORB keypoint matching and CLIP cosine similarity against the listing
crop.

After all frames are processed, it aggregates the scores using max, mean, and top-3 mean. The top-3
mean is the most useful metric in practice — it represents how well the best few frames matched,
which mirrors how a human would verify (you don't need every frame to be a great match, just a few
clear ones).

## Why Top-3 Mean Beats Plain Max or Mean
A single max can be a fluke (one lucky frame, one unlucky frame). The mean drags down good matches
because most video frames will be motion-blurred, oddly angled, or partially occluded. Top-3 mean
splits the difference: it rewards videos that have a few genuinely good viewpoints of the bike
without being thrown off by garbage frames.

## Tuning Knobs Worth Knowing

A few parameters you'll likely adjust:

* *`target_fps`* — 2 fps is a reasonable default for a 10–20 second pan video. Bump it to 4–5 fps  
  for short videos, drop to 1 fps for long ones.
* *`max_frames`* — caps total processing time. 40–60 frames is usually enough for a good verdict.
* *YOLO `conf_threshold`* — 0.4 is conservative. Lower it to 0.25 if bikes get missed in odd  
  angles; raise it to 0.6 if you get false positives on bike-shaped clutter.
* *Verdict thresholds in `make_verdict`* — these are placeholders. Run the pipeline against 20–30
* known-match and known-mismatch pairs, look at the score distributions, and pick thresholds that  
  give clean separation.

## Performance Notes
On a CPU, expect roughly 0.5–1.5 seconds per frame (YOLO + CLIP + ORB combined). A 20-second video
sampled at 2 fps means ~40 frames, so 20–60 seconds total. With a GPU, this drops to a few seconds
total. For a production system you'd batch the YOLO and CLIP inferences across frames for a big
speedup.

## Liveness / Anti-Spoofing Hooks
For a real fraud-prevention system, you'd want to add checks before trusting the match score. A few
ideas you could plug into this pipeline:

* Compute the variance of the YOLO bounding box position and size across frames — a real handheld  
  video will have natural jitter; a static photo of a photo won't.
* Check that the bike appears at meaningfully different scales/angles across frames (compare CLIP  
  embeddings between video frames; if they're all 99% similar to each other, it's probably a static  
  image being filmed).
* Look at frame-to-frame optical flow to confirm 3D parallax rather than 2D translation.

I left these out to keep the example focused, but they're the natural next layer. Want me to add a
basic liveness check, or build out the "save annotated frames with matched keypoints drawn on"
debug visualization?
"""



# # Visualization.
# def make_demo_frame(video_frame, listing_crop, match_score, verdict, bbox=None):
#     """Stitch listing image + annotated video frame side by side with verdict overlay."""
#     # Resize listing crop to match video frame height
#     vh, vw = video_frame.shape[:2]
#     lh, lw = listing_crop.shape[:2]
#     new_lw = int(lw * (vh / lh))
#     listing_resized = cv2.resize(listing_crop, (new_lw, vh))

#     # Draw bbox on a copy of the video frame
#     annotated = video_frame.copy()
#     if bbox:
#         x1, y1, x2, y2 = bbox
#         # Color based on verdict
#         color = (0, 255, 0) if "same" in verdict else (0, 165, 255)
#         cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)

#     # Stitch horizontally
#     combined = np.hstack([listing_resized, annotated])

#     # Add labels
#     cv2.putText(combined, "LISTING PHOTO", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     cv2.putText(combined, "VERIFICATION VIDEO", (new_lw + 10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     # Bottom banner with match info
#     h, w = combined.shape[:2]
#     banner_height = 80
#     banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
#     banner_color = (0, 100, 0) if "same" in verdict else (0, 50, 100)
#     banner[:] = banner_color

#     cv2.putText(banner, f"Match: {match_score:.1f}%",
#                 (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
#     cv2.putText(banner, f"Verdict: {verdict}",
#                 (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     return np.vstack([combined, banner])




def annotate_video(input_path, output_path, listing_crop=None):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4v works well for .mp4; use 'XVID' for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on this frame
        results = model(frame, verbose=False, classes=[1], conf=0.4)

        # Find largest bike box
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

        # Draw the box
        if largest_box:
            x1, y1, x2, y2 = largest_box
            # Green box, 3px thick
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Label background
            label = f"BIKE {best_conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_h - 10),
                          (x1 + text_w + 10, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Header banner with frame info
        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {frame_idx}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {output_path}")


## Call this from the video loop.
# This builds an image that shows the video and listing together.
def make_demo_frame(video_frame, listing_crop, match_score, verdict, bbox=None):
    """Stitch listing image + annotated video frame side by side with verdict overlay."""
    # Resize listing crop to match video frame height
    vh, vw = video_frame.shape[:2]
    lh, lw = listing_crop.shape[:2]
    new_lw = int(lw * (vh / lh))
    listing_resized = cv2.resize(listing_crop, (new_lw, vh))

    # Draw bbox on a copy of the video frame
    annotated = video_frame.copy()
    if bbox:
        x1, y1, x2, y2 = bbox
        # Color based on verdict
        color = (0, 255, 0) if "same" in verdict else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)

    # cv2.imwrite("debug_video_frame.jpg", annotated)  # Debug: save the annotated frame to disk
    # cv2.imwrite("debug_listing_frame.jpg", listing_resized)  # Debug: save the listing frame to disk

    # Stitch horizontally
    combined = np.hstack([listing_resized, annotated])

    # Add labels
    cv2.putText(combined, "LISTING PHOTO", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "VERIFICATION VIDEO", (new_lw + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Bottom banner with match info
    h, w = combined.shape[:2]
    banner_height = 80
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner_color = (0, 100, 0) if "same" in verdict else (0, 50, 100)
    banner[:] = banner_color

    cv2.putText(banner, f"Match: {match_score:.1f}%",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(banner, f"Verdict: {verdict}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return np.vstack([combined, banner])