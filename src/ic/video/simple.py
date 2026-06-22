# Simple stuff, playground.

# Use YOLO to build a prediction on an image.
import cv2
import numpy as np
from ultralytics import YOLO

from ic import settings


# Load YOLO once and reuse
model = YOLO(settings.YOLO_26N)


def simple_box ():
    model = YOLO(settings.YOLO_26N)
    results = model(settings.IMAGE_DIR / 'bike-2019-haibike-02.jpg')
    print(results)
    print('Done.')
    return results


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
                # Consider expanding the box 5-10% to capture edge keypoints.
                # x1 = max(0, x1 - int(0.05 * (x2-x1)))
                # y1 = max(0, y1 - int(0.05 * (y2-y1)))
                # x2 = min(image.shape[1], x2 + int(0.05 * (x2-x1)))
                # y2 = min(image.shape[0], y2 + int(0.05 * (y2-y1)))
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
    """
    Compare two image crops using ORB features.
    Returns a match percentage (0-100) based on geometrically consistent matches.
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0, 0

    # Brute-force matcher with Hamming distance (correct for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test to filter ambiguous matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return 0.0, len(good_matches), 0

    # Use RANSAC homography to keep only geometrically consistent matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    # Match percentage: inliers as a fraction of the smaller keypoint set
    denom = min(len(kp1), len(kp2))
    match_pct = (inliers / denom) * 100 if denom > 0 else 0.0

    return match_pct, len(good_matches), inliers


def compare_bikes(image_path1, image_path2):
    crop1, box1 = detect_largest_bicycle(image_path1)
    crop2, box2 = detect_largest_bicycle(image_path2)

    if crop1 is None:
        print(f"No bicycle found in {image_path1}")
        return
    if crop2 is None:
        print(f"No bicycle found in {image_path2}")
        return

    print(f"Image 1 bike crop: {crop1.shape}, box: {box1}")
    print(f"Image 2 bike crop: {crop2.shape}, box: {box2}")

    # First visualize the matches for debug inspection.
    print("Visualizing matches...")
    visualize_matches(crop1, crop2)

    # Then do the actual comparison and print results.
    match_pct, good, inliers = compare_with_orb(crop1, crop2)
    print(f"Good matches: {good}, geometric inliers: {inliers}")
    print(f"Match percentage: {match_pct:.1f}%")

    # Optional: save the crops for visual inspection
    cv2.imwrite(settings.VAR_DIR / "crop1.jpg", crop1)
    cv2.imwrite(settings.VAR_DIR / "crop2.jpg", crop2)

    return match_pct


def visualize_matches(crop1, crop2, output_path=settings.VAR_DIR / "matches.jpg"):
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

    # Sort by distance and keep the best 50 for clarity
    good = sorted(good, key=lambda m: m.distance)[:50]

    matched_image = cv2.drawMatches(
        crop1, kp1, crop2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, matched_image)
    print(f"Saved match visualization to {output_path}")



# CLIP Compare is better for semantic similarity, e.g. differing angles.
# Because we'll compare against video frames this won't be as useful.
# import torch
# import open_clip
# from PIL import Image
# import numpy as np

# # Load CLIP once
# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#     'ViT-B-32', pretrained='openai'
# )
# clip_model.eval()

# def embed_with_clip(crop_bgr):
#     """Get a CLIP embedding for a BGR image crop."""
#     # Convert BGR -> RGB -> PIL
#     crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(crop_rgb)
#     img_tensor = clip_preprocess(pil_img).unsqueeze(0)

#     with torch.no_grad():
#         embedding = clip_model.encode_image(img_tensor)
#         embedding = embedding / embedding.norm(dim=-1, keepdim=True)

#     return embedding.cpu().numpy().flatten()


# def compare_with_clip(crop1, crop2):
#     """Cosine similarity of CLIP embeddings, scaled to a 0-100 percentage."""
#     e1 = embed_with_clip(crop1)
#     e2 = embed_with_clip(crop2)
#     cosine_sim = float(np.dot(e1, e2))  # already normalized
#     # CLIP cosine sim for similar images usually lands in 0.7-0.95
#     # Map [0, 1] to [0, 100] for a readable percentage
#     return max(0.0, cosine_sim) * 100

# clip_score = compare_with_clip(crop1, crop2)
# print(f"CLIP similarity: {clip_score:.1f}%")

# This uses ORB and CLIP together for a more robust comparison, but for the POC we'll just do ORB
# since it's more explainable and we care about exact matches.
# def verify_match(image_path1, image_path2):
#     crop1, _ = detect_largest_bicycle(image_path1)
#     crop2, _ = detect_largest_bicycle(image_path2)

#     if crop1 is None or crop2 is None:
#         return {"verdict": "no_bike_detected", "orb": 0, "clip": 0}

#     orb_score, _, inliers = compare_with_orb(crop1, crop2)
#     clip_score = compare_with_clip(crop1, crop2)

#     # Simple decision logic — tune these thresholds with real data
#     if clip_score > 85 and orb_score > 5:
#         verdict = "likely_same_bike"
#     elif clip_score > 80:
#         verdict = "same_model_uncertain_identity"
#     else:
#         verdict = "different_bike"

#     return {
#         "verdict": verdict,
#         "orb_match_pct": round(orb_score, 1),
#         "orb_inliers": inliers,
#         "clip_similarity_pct": round(clip_score, 1),
#     }


# result = verify_match("listing.jpg", "video_frame.jpg")
# print(result)


# Functions to support video frame extraction and aggregate comparisons.