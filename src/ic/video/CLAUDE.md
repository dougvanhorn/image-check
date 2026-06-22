# CLAUDE.md

Guidance for working on this project in future Claude Code sessions.

## Project Overview

This is a **fraud-prevention proof of concept** for the Pinkbike BuySell mountain-bike marketplace. The system verifies that a short video submitted by a seller shows the *same physical bike* as the photo in their listing — combating a common scam where fraudsters list a bike using photos they don't actually own.

The core question the system answers: "Is the bike in this video the same individual bike as the one in this listing photo, and was the video actually filmed live (not a photo of a photo)?"

This is a POC aimed at a **stakeholder demonstration** (5-minute presentation format), not production. Non-technical, plain-language framing of the technical components matters for that audience.

## Tech Stack

- **Python** — primary language
- **Ultralytics YOLOv8** (`yolov8n.pt`) — bike detection / bounding boxes
- **OpenCV** (`opencv-python`) — ORB keypoint matching, homography, video I/O
- **CLIP** via `open-clip-torch` (`ViT-B-32`, `openai` weights) — semantic similarity + zero-shot view classification
- **PyTorch** — backs CLIP
- **NumPy** — array handling
- **scikit-learn** — KMeans for viewpoint clustering

Install:
```bash
pip install ultralytics opencv-python numpy torch torchvision open-clip-torch scikit-learn
```

## Architecture / How It Works

The pipeline (`verify_video_against_listing`) does the following:

1. **Listing image** is processed once: detect largest bike (YOLO), crop with ~5% padding, resize to 800px width, compute CLIP embedding.
2. **Video frames** are extracted from a `.mov` at ~2 fps (configurable), capped at `max_frames`.
3. **Per frame**: detect largest bike, crop/resize, then score against the listing with both ORB (identity-level) and CLIP (semantic).
4. **Liveness** is assessed incrementally as frames process (homography residual for flat-image spoofing; CLIP zero-shot for viewpoint coverage).
5. **Aggregate** scores using max, mean, and **top-3 mean** (the primary metric).
6. **Verdict** combines match scores + liveness + viewpoint coverage.
7. Optionally generates a **demo video** with side-by-side layout, status panel, and verdict banner.

### Two scoring signals, combined

- **ORB** (OpenCV keypoint matching) catches *identity-level* detail — same stickers, scratches, cable routing. High specificity, struggles across large angle changes.
- **CLIP** (deep embeddings) catches *semantic* similarity — robust to angle/lighting but can't distinguish two bikes of the same model.

Neither alone is sufficient; the verdict logic uses both.

## Key Design Decisions

- **Server-side processing** (not on-device) — simpler for a POC.
- **Top-3 mean aggregation** is the primary metric — rewards a few good viewpoints without being dragged down by motion-blurred frames.
- **Largest bounding box** wins when multiple bikes are detected.
- **Homography residual** is the chosen liveness method — best effort-to-payoff ratio, catches the dominant spoof (filming a screen/printout).
- **Crops padded ~5%** before comparison to preserve edge keypoints; **resized to common 800px width** for fair matching.

## Gotchas & Hard-Won Learnings

- **`cv2.VideoWriter` fails silently with all-green output when fps is too low.** `mp4v` corrupts below ~5 fps. Fix: write at **10 fps and duplicate each frame** (`DEMO_HOLD_FRAMES`) to get slow visual playback with a codec-friendly file. Alternative: `MJPG`/`.avi`, which tolerates any fps but produces large files.
- **`VideoWriter` takes `(width, height)`; NumPy arrays are `(height, width, channels)`.** Mixing these up is the #1 cause of corrupt video.
- **`VideoWriter.isOpened()` returning True only means the file was created** — not that the codec accepted the settings. Validate with a short test clip for unusual configs.
- **Frames must be 3-channel BGR `uint8`.** `np.zeros()` defaults to `float64` — always pass `dtype=np.uint8`.
- **Lazy-initialize the demo `VideoWriter`** on the first built frame, since output dimensions aren't known until `make_demo_frame_v2` runs once. The title card must be written *after* writer init (so dims match) but *before* the first real frame (so it plays first).
- **All thresholds in the code are placeholders.** They need calibration against a real labeled dataset (real walkarounds / spoofs / same-model-different-bike) before they mean anything.

## Licensing — Important

- **Ultralytics YOLOv8 is AGPL-3.0.** Fine for this POC, but a **production blocker**: commercial deployment would require open-sourcing the whole app or buying a commercial license from Ultralytics.
- **Production path:** switch to **YOLOX** (Apache 2.0) or a Torchvision detector (BSD), or purchase the commercial license.
- CLIP via `open-clip-torch` and OpenCV are permissively licensed.

Flag this whenever the conversation drifts toward production/commercialization.

## Conventions

- Functions return structured results via `@dataclass` (`FrameResult`, `VerificationResult`, `LivenessResult`, `LivenessAccumulator`).
- Models (YOLO, CLIP) are loaded **once at module level**, not per-call.
- Color constants are BGR (OpenCV convention), not RGB.
- Verdict strings are snake_case identifiers (e.g. `likely_same_bike`, `likely_spoof_flat_image`, `insufficient_viewpoints`).

## Demo Presentation Notes

When producing demo material for stakeholders:

- The **bounding-box color change** (yellow → green/red) is the most legible signal for non-technical viewers — they don't read the side panel closely.
- **Hold the final verdict frame ~7.5s** so viewers can read it.
- Plain-language definitions that have worked:
  - **ORB:** picks out distinctive visual landmarks (a sticker corner, a scratch, where a cable meets the frame) and uses them like a fingerprint to confirm two images show the same physical bike, not just a look-alike.
  - **Homography:** a test for whether the camera is looking at something flat (a photo/screen) vs a real 3D object, by checking whether everything moves together (flat) or at different rates by depth (real).

## Current Status

Implemented: YOLO detection, ORB + CLIP matching, frame extraction, score aggregation, demo video with status-panel overlay, homography liveness, CLIP viewpoint coverage, title card (with optional fade-in).

## Open Items / Next Steps

1. Side-by-side spoof comparison demo (real vs spoof video in one output)
2. End card with summary stats
3. Draw ORB keypoint match lines between listing and frame
4. Running score chart in a corner of the demo video
5. Gradio web UI wrapping the pipeline
6. Calibration harness against a labeled dataset to set real thresholds
7. Fine-tune YOLO on mountain-bike data if full-suspension frames get missed
8. Store reference embeddings with listings (verify only new video)
9. Add optical-flow liveness as a second signal
10. Video metadata sanity checks (duration, resolution, EXIF)
11. Per-view capture guidance ("please also film the drivetrain")
12. Production licensing migration (YOLOX / Torchvision)
