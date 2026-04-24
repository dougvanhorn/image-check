# Pinkbike BuySell Fraud Prevention POC: Video-to-Listing Verification

## Original Question

> I want to explore using the opencv library to combat fraud on the public marketplace Pinkbike BuySell. I want the user to use a camera on their phone to video a mountain bike, and use opencv to verify that the item in the video matches an image the user has uploaded to a marketplace listing.
>
> Provide a high level list of the steps needed to get a working proof of concept showcasing the above verification behavior.

---

## High-Level Roadmap

### 1. Define the Verification Goal

Decide what "match" actually means. For a first POC, aim to verify that the bike in the video is *the same physical bike* (same frame, components, wear patterns, stickers, scratches) as the one in the listing photo — not just "a bike that looks similar." This distinction drives every technical choice downstream.

### 2. Set Up the Environment

Install Python, OpenCV (`opencv-python`), and supporting libraries (NumPy, possibly PyTorch or TensorFlow if you go the deep-learning route). Decide early whether processing happens on-device (mobile) or server-side (user uploads media, backend verifies). Server-side is much easier for a POC.

### 3. Capture Inputs

You need two artifacts: the listing image (already uploaded) and a short verification video (e.g., 10–20 seconds panning around the bike). For the POC, you can simulate the mobile capture with any phone camera and upload both files to your backend.

### 4. Preprocess the Media

Extract frames from the video at a sensible rate (say, 2–5 fps) to get a set of still images. Normalize all images — resize, correct orientation, possibly adjust lighting/contrast. Optionally run a bike detector (YOLO or similar) to crop out the bike and ignore the background, which dramatically improves matching reliability.

### 5. Extract Features for Comparison

This is the core of the verification. A few approaches, in rough order of sophistication:

- **Classical keypoint matching (start here):** Use ORB, SIFT, or AKAZE in OpenCV to detect keypoints and descriptors in the listing photo and each video frame. Match descriptors with a brute-force or FLANN matcher, then filter matches using Lowe's ratio test and RANSAC for geometric consistency.
- **Deep embeddings (better accuracy):** Pass images through a pretrained CNN (ResNet, CLIP, DINOv2) to get feature vectors, then compare via cosine similarity.
- **Hybrid:** Use deep embeddings for coarse matching and keypoint matching to confirm identity-level details like sticker placement or scratches.

### 6. Score the Match

Aggregate matches across all extracted video frames. Define a scoring function — e.g., percentage of frames with enough geometrically consistent keypoint matches above a threshold. Decide pass/fail cutoffs empirically.

### 7. Add Liveness / Anti-Spoofing Checks

Critical for fraud prevention, since a scammer could just video someone else's listing photo on a screen. Check for:
- Parallax and motion consistency between frames (a real 3D bike looks different from different angles; a photo of a photo doesn't)
- Multiple viewpoints actually present (front, side, drivetrain, etc.)
- Metadata sanity checks (video duration, resolution, EXIF if available)

### 8. Build a Minimal Interface

A simple web form where you upload the listing image and the video, the backend runs verification, and returns a match score with annotated frames showing the matched keypoints. This makes the POC demo-able.

### 9. Test with Real Data

Gather a small dataset: same-bike pairs (should match), different-bikes-same-model pairs (should ideally fail — this is the hard case), and obvious mismatches. Tune thresholds against this set.

### 10. Document Limitations

Be upfront about what the POC can and can't do: lighting sensitivity, handling of modified bikes (new wheels, new paint), adversarial attacks, and cases where two bikes genuinely look near-identical. This guides what a production version would need.

---

A reasonable first milestone: steps 1–6 with ORB keypoint matching, on a handful of test bikes, running server-side. You can have something demonstrable in a day or two, then layer on the anti-spoofing and deep embeddings once the baseline works.

---

## Follow-up: YOLO for Bike Detection

> Say more about the bike detection step and the YOLO libraries. Are they free, do they cost money, how can i get a hold of them?

### What YOLO Actually Does Here

YOLO ("You Only Look Once") is a family of real-time object detection models. You feed it an image, and it returns bounding boxes with class labels and confidence scores — e.g., "bicycle at pixels (120, 80, 540, 420), confidence 0.94." For your POC, you'd run YOLO on the listing image and on each extracted video frame, crop to the bike's bounding box, and then do keypoint matching only on the cropped regions. This eliminates background noise (garage clutter, grass, other objects) that would otherwise generate spurious feature matches.

Conveniently, "bicycle" is one of the 80 default classes in the COCO dataset that most YOLO models are pretrained on, so you get bike detection out of the box with zero custom training.

### The YOLO Landscape

YOLO isn't a single library — it's a lineage of models, and different versions come from different groups with different licenses. Here are the ones that matter:

**Ultralytics YOLO (YOLOv8, YOLOv11, YOLOv12, etc.)**
The most popular modern implementation. Maintained by the company Ultralytics. Dead simple Python API. Actively developed. This is what most tutorials use today.

**YOLOv5 (also Ultralytics)**
Older but still widely used and stable. Same company, same license situation.

**YOLOv7, YOLOv9, YOLOv10**
Research releases from academic groups, each with their own repos. Often released under GPL or MIT.

**YOLO-NAS (by Deci AI)**
A newer architecture, available through their `super-gradients` library.

For a POC, **Ultralytics YOLOv8 or YOLOv11 is the obvious pick** — best documentation, easiest API, and pretrained weights are one line of code away.

### The Cost and Licensing Question

This is where it gets important, especially since you're building something for a commercial marketplace.

**The code and pretrained models are free to download.** You can `pip install ultralytics` right now and be running bike detection in five minutes with no payment.

**But the license matters.** Ultralytics YOLO is released under **AGPL-3.0**, which is a strong copyleft license. In practice this means:

- Free for personal use, research, experimentation, and your POC — no problem at all.
- If you deploy it as part of a commercial product or a network service (like a Pinkbike backend), AGPL requires that you open-source your *entire* application under AGPL too. This is usually a dealbreaker for commercial use.
- Ultralytics sells a **commercial license** that removes the AGPL requirement. You'd need to contact them for pricing; it's typically negotiated per-deployment and can range from modest to significant depending on scale.

**Alternatives that avoid AGPL** if you're thinking beyond the POC:

- **YOLOv5** older releases had more permissive licensing in some forks, but current Ultralytics-maintained YOLOv5 is also AGPL.
- **YOLOX** (from Megvii) is Apache 2.0 — commercially friendly.
- **YOLO-NAS** from Deci is Apache 2.0 for the architecture, though pretrained weights have their own terms.
- **Detectron2** (from Meta) uses Apache 2.0 and includes object detection models, though not branded YOLO.
- **Torchvision's built-in detection models** (Faster R-CNN, RetinaNet, etc.) are BSD-licensed and come with PyTorch. Slower than YOLO but no licensing headaches.
- **MMDetection** is Apache 2.0 and includes YOLO-family implementations trained from scratch.

My honest recommendation: use Ultralytics YOLOv8 for the POC since it's fastest to get running, then if the project progresses toward production, either buy a commercial license or migrate to YOLOX or a Torchvision detector.

### How to Get It

For Ultralytics YOLO, it's a one-liner:

```bash
pip install ultralytics
```

Then in Python:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 'n' = nano, smallest/fastest; also s, m, l, x
results = model('listing_photo.jpg')

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == 'bicycle':
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            # crop the image using these coordinates
```

The first time you load a model, it auto-downloads the weights file (a few MB to a few hundred MB depending on model size). No API key, no account, no signup.

For YOLOX:

```bash
pip install yolox
```

Docs and pretrained weights are on their GitHub repo (Megvii-BaseDetection/YOLOX).

### Practical Notes for Your POC

- **Use the "nano" or "small" variants** (`yolov8n.pt` or `yolov8s.pt`) for the POC. They're fast, run fine on CPU, and bicycle detection is an easy task that doesn't need a huge model.
- **Confidence threshold:** default is usually 0.25. You might raise it to 0.5 or higher to avoid false bike detections in busy backgrounds.
- **Multiple bikes in frame:** if there's more than one bicycle detected, you'll need logic to pick the right one — probably the largest bounding box, or the most centered.
- **Fine-tuning:** if generic bicycle detection isn't precise enough for mountain bikes specifically (e.g., it sometimes misses bikes at odd angles or with full-suspension frames), you can fine-tune on a small custom dataset. But try the pretrained model first — it's usually fine.

For the POC phase, AGPL is a non-issue since you're not distributing the software. Just be aware of it before the project turns into something real.
