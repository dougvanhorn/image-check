"""
# Liveness Detection: Parallax, Motion, and Viewpoint Coverage

These are two distinct problems with different solutions, so I'll tackle them separately and then
show how to combine them.

## Part 1: Detecting Real 3D Motion (Parallax)

The goal is to distinguish a real 3D bike being filmed from a flat image being filmed (someone
pointing their phone at a photo on a screen or printout). Both produce video with motion — the
question is whether that motion is 3D-consistent.

### The Key Insight: Parallax Breaks on Flat Scenes

When you film a real 3D object and move the camera, different parts of the object move at different
rates relative to each other. The handlebars might occlude part of the frame momentarily, then
reveal something behind them. The rear wheel and front wheel change relative position. A spoke
visible from one angle isn't visible from another.

When you film a flat photo, everything moves together. Every pixel translates, rotates, or scales
uniformly because the scene is mathematically a 2D plane.

This is testable. Four practical methods, roughly in order of sophistication:

### Method A: Homography Residual Test (Simple and Effective)

If you can fit a single homography (2D perspective transform) to map frame N to frame N+1 with very
low error across the entire bike, the scene is likely flat. Real 3D scenes can't be described by a
single homography — you'd need per-pixel depth.
"""

import cv2
import numpy as np
import rich

from ic.video import settings
from ic.video import utils

# Given a video, let's extract crops of the bike and test homography fit error between consecutive
# frames.


def run(filename=None):
    """Test homography fit error between consecutive video frames.

    Arguments:
        filename: Optional path to a video file. If not provided, uses a default test video.
    """

    rich.print(f'[bold]Testing homography fit error on video: [blue]{filename}[/blue][/bold]')

    # Get a list of FrameExtract instances from the video.
    frames = utils.extract_frames(filename, target_fps=2, max_frames=40)

    if not frames:
        rich.print("[red]No frames extracted from video.[/red]")
        return

    # Extract the bike crop for every frame.

    # Sampling would be better if we used frames 1 second apart, we'll get clearer parallax.
    prev_frame = frames[0]
    for idx, frame in enumerate(frames[1:], start=1):
        # Assume we have a function that detects the bike and returns a crop of it.
        prev_crop = utils.detect_largest_bicycle(prev_frame.bgr)
        curr_crop = utils.detect_largest_bicycle(frame.bgr)
        # We're now holding two BicycleCrop instances.
        # print(prev_crop)
        # print(curr_crop)

        if prev_crop.bgr is not None and curr_crop.bgr is not None:
            # We'll need to develop a rubric on this, how much "good" vs. "bad" vs. "n/a" do we expect.
            # Probaly ask Wen.
            score = homography_fit_error(prev_crop.bgr, curr_crop.bgr)
            if score > 0.85:
                rich.print(f"Frame {idx}: [red]Suspiciously flat scene detected![/red] Homography fit score: [bold]{score:.2f}[/bold]")
            elif score > .5:
                rich.print(f"Frame {idx}: [green]Normal 3D scene with parallax.[/green] Homography fit score: [bold]{score:.2f}[/bold]")
            else:
                rich.print(f"Frame {idx}: [grey]Very 3D scene or too much motion/blur to tell.[/grey] Homography fit score: [bold]{score:.2f}[/bold]")

        prev_frame = frame


def homography_fit_error(frame1_crop, frame2_crop):
    """Returns a 'flatness score' between 0 (very 3D) and 1 (perfectly flat).

    High scores suggest the bike might be a flat image.

    Interpretation:
        > 0.85  -> suspiciously flat (possible spoof)
        0.5-0.85 -> normal 3D scene with parallax
        < 0.5   -> either very 3D or too much motion/blur to tell

    Arguments:
        frame1_crop: BGR image crop of the bike from frame N.
        frame2_crop: BGR image crop of the bike from frame N+1.

    Returns:
        A float score between 0 and 1 indicating how well a homography fits the transformation
        between the two crops. Higher means more consistent with a flat image.
    """
    gray1 = cv2.cvtColor(frame1_crop, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_crop, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None  # Can't determine

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < 20:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Fit homography with RANSAC
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None

    inlier_ratio = mask.sum() / len(mask)
    # High inlier ratio = scene is well-described by a 2D transform = probably flat
    return float(inlier_ratio)


# Claude suggestion.
# @dataclass
# class LivenessResult:
#     motion_consistency: dict = field(default_factory=dict)
#     viewpoint_coverage: dict = field(default_factory=dict)
#     verdict: str = ""
#     is_likely_live: bool = False


# def assess_liveness(frame_results, frame_crops, frame_embeddings,
#                     original_frames, clip_tools=None):
#     """
#     Run all liveness checks on a video's frames.

#     frame_results: from your pipeline (has bboxes)
#     frame_crops: list of bike crops aligned to frame_results
#     frame_embeddings: CLIP embeddings of each crop
#     original_frames: full frames for flow analysis
#     clip_tools: (model, preprocess, tokenizer) if doing view classification
#     """
#     # --- Motion/parallax check ---
#     bike_frames = [(i, fr) for i, fr in enumerate(frame_results) if fr.bike_detected]
#     flatness_scores = []

#     # Sample pairs roughly 1 second apart for clearer parallax
#     for i in range(len(bike_frames) - 2):
#         idx_a = bike_frames[i][0]
#         idx_b = bike_frames[i + 2][0]
#         if idx_a >= len(frame_crops) or idx_b >= len(frame_crops):
#             continue
#         score = homography_fit_error(frame_crops[idx_a], frame_crops[idx_b])
#         if score is not None:
#             flatness_scores.append(score)

#     motion_result = {}
#     if flatness_scores:
#         avg_flatness = float(np.mean(flatness_scores))
#         motion_result = {
#             "mean_homography_inlier_ratio": avg_flatness,
#             "samples": len(flatness_scores),
#             "likely_flat_image": avg_flatness > 0.85,
#         }
#     else:
#         motion_result = {"error": "insufficient_frame_pairs"}

#     # --- Viewpoint coverage check ---
#     coverage_result = analyze_viewpoint_coverage(frame_embeddings)

#     # Optional: zero-shot view classification
#     if clip_tools and len(frame_crops) > 0:
#         view_result = assess_view_coverage(frame_crops, *clip_tools)
#         coverage_result["view_classification"] = view_result

#     # --- Combined verdict ---
#     is_flat = motion_result.get("likely_flat_image", False)
#     is_static = coverage_result.get("likely_static", False)
#     has_coverage = coverage_result.get("sufficient_coverage", False)

#     if is_flat:
#         verdict = "likely_spoof_flat_image"
#     elif is_static and not has_coverage:
#         verdict = "insufficient_viewpoints"
#     elif has_coverage and not is_flat:
#         verdict = "live_multi_view_capture"
#     else:
#         verdict = "ambiguous"

#     return LivenessResult(
#         motion_consistency=motion_result,
#         viewpoint_coverage=coverage_result,
#         verdict=verdict,
#         is_likely_live=verdict == "live_multi_view_capture",
#     )