"""
# Part 2: Viewpoint Coverage Detection

Here the goal is different: confirm the video actually shows multiple angles of the bike (front,
side, rear, drivetrain) rather than one static view. This both raises the bar for fraud (scammer
needs multi-angle photos of the right bike) and improves matching reliability.


## Calibration Notes

These tests are worthless without tuning on real data. Before trusting them, collect:

* ~20 real walkaround videos of different bikes in different settings
* ~20 spoof videos: phone filming a photo on another screen, phone filming a printed photo, static  
  videos of a bike
* ~10 edge cases: very dim lighting, hand shake but no walkaround, bike with busy background

Run the full pipeline across all of them, look at the score distributions, and set thresholds where
the distributions separate cleanly. The numbers I gave in the code comments (0.85, 0.94, 0.96,
etc.) are starting points based on general experience, not ground truth for your specific setup.

## Practical Prioritization

If you have limited time, the liveness check with the best effort-to-payoff ratio is Method A
(homography residual). It's 30 lines of code, runs in milliseconds per frame pair, and catches the
dominant spoofing attack (filming a screen or printed photo). Every other check is a nice-to-have
on top of that.

View classification is more about improving match confidence than detecting fraud — it's most
useful for telling the user "please also film the drivetrain" during capture, rather than as a
post-hoc security check.

Want me to show how to surface these liveness results in the demo video overlay, so reviewers can
see the system flagging a fake in real time?
"""

import numpy as np
from sklearn.cluster import KMeans


def analyze_viewpoint_coverage(frame_embeddings, n_expected_views=4):
    """
    Given a list of CLIP embeddings from video frames, estimate how many
    distinct viewpoints are represented.

    ## Approach: Embedding Clustering

    Compute a visual embedding for each frame's bike crop, then cluster them. A proper walkaround
    produces embeddings spread across "view space"; a static shot produces tightly clustered
    embeddings.

    Interpretation: if mean pairwise CLIP similarity between frames is above ~0.96, the video is
    essentially one view. Below ~0.94, you have genuine viewpoint diversity. The exact thresholds
    need calibration against your real data.

    Arguments:
        frame_embeddings: List of CLIP embeddings (one per frame with a detected bike).
        n_expected_views: Approximate number of distinct views we expect in a genuine walkaround.

    Returns:
        dict: Coverage metrics including mean pairwise similarity, inter-cluster similarity,
              cluster sizes, and flags for likely static video and sufficient coverage.
    """
    if len(frame_embeddings) < n_expected_views:
        return {"sufficient_coverage": False, "reason": "too_few_frames"}

    embeddings = np.array(frame_embeddings)

    # Pairwise cosine similarity
    sim_matrix = embeddings @ embeddings.T  # assumes pre-normalized

    # Mean off-diagonal similarity: how self-similar is the video?
    n = len(embeddings)
    mask = ~np.eye(n, dtype=bool)
    mean_pairwise_sim = sim_matrix[mask].mean()

    # Cluster into N groups and see how spread out the clusters are
    kmeans = KMeans(n_clusters=min(n_expected_views, n), n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    cluster_sizes = np.bincount(labels)

    # Compute inter-cluster distance
    centroids = kmeans.cluster_centers_
    # Normalize centroids for cosine comparison
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    inter_cluster_sim = (centroids @ centroids.T)
    inter_cluster_sim_mean = inter_cluster_sim[~np.eye(len(centroids), dtype=bool)].mean()

    # Heuristics
    # A static video has mean_pairwise_sim > 0.97 and inter_cluster_sim ~= mean_pairwise_sim
    # A proper walkaround has mean_pairwise_sim around 0.88-0.94 and inter_cluster_sim around 0.85-0.92
    return {
        "mean_pairwise_similarity": float(mean_pairwise_sim),
        "inter_cluster_similarity": float(inter_cluster_sim_mean),
        "cluster_sizes": cluster_sizes.tolist(),
        "likely_static": mean_pairwise_sim > 0.96,
        "sufficient_coverage": mean_pairwise_sim < 0.94 and min(cluster_sizes) >= 2,
    }


import open_clip
import torch


def classify_bike_view(crop_bgr, clip_model, clip_preprocess, tokenizer):
    """Zero-shot classify a bike crop into canonical viewpoints.

    ## Approach: Explicit View Classification (Advanced)

    If you want to specifically verify front/side/rear coverage, train a small classifier that
    labels each frame as one of a few canonical bike views. You can do this cheaply with a handful
    of labeled reference images and zero-shot CLIP:

    Zero-shot CLIP isn't as accurate as a trained classifier, but it's a zero-training-cost way to
    get decent view labels for a POC.

    Arguments:
    crop_bgr: The cropped bike image in BGR format.
    clip_model: Pre-loaded CLIP model for encoding.
    clip_preprocess: CLIP preprocessing function.
    tokenizer: CLIP tokenizer for text prompts.

    Returns:
    dict with:
        - view: The classified view (e.g., "front", "side", "rear", "drivetrain", "handlebars")
        - confidence: CLIP similarity score for the best-matching view prompt
        - all_scores: Similarity scores for all view prompts (for debugging and threshold tuning)
    """
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


def assess_view_coverage(frame_crops, clip_model, clip_preprocess, tokenizer):
    """Classify each frame and report which canonical views were seen."""
    views_seen = set()
    frame_views = []

    for crop in frame_crops:
        result = classify_bike_view(crop, clip_model, clip_preprocess, tokenizer)
        frame_views.append(result["view"])
        if result["confidence"] > 0.25:  # threshold
            views_seen.add(result["view"])

    required_views = {"side", "front", "back"}
    return {
        "views_detected": list(views_seen),
        "frame_by_frame": frame_views,
        "required_views_covered": required_views.issubset(views_seen),
        "coverage_fraction": len(views_seen & required_views) / len(required_views),
    }