"""Geometry-tolerant image-matching signals for PB-658 (spike).

Watermarking (see `ic.stega`) survives recompression but is destroyed by resize
and crop. These signals take the opposite approach: instead of *embedding* a
mark, they derive a descriptor from the image content itself and match by
similarity -- which is inherently tolerant to resize, crop, and recompression.

Two candidates, exposing a common interface (`descriptor` + `similarity`):

  - PerceptualHash (pHash): DCT-based hash. Fast, dependency-light, no model.
    Resize-tolerant and recompression-tolerant; moderately crop-tolerant.
  - ClipEmbedding: semantic embedding via open_clip ViT-B-32 (the same model the
    `ic.video` package uses). Very robust to appearance changes, but heavy
    (torch) and matches by *meaning* -- so two different bikes of the same model
    will also score high. That's a false-positive risk to weigh, not a bug.

Similarity is always returned in [0, 1], higher = more similar.
"""

import cv2
import numpy as np
from PIL import Image


class PerceptualHash:
    """DCT perceptual hash via the `imagehash` library."""

    name = "phash"
    # Hamming-similarity at/above which two images are considered the same.
    # Placeholder -- calibrate against a labeled set before trusting it.
    default_threshold = 0.85

    def descriptor(self, image_bgr: np.ndarray):
        import imagehash  # local import: keeps the dependency lazy
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return imagehash.phash(Image.fromarray(rgb))

    def similarity(self, a, b) -> float:
        # `a - b` is the Hamming distance between the two hashes.
        bits = a.hash.size
        return 1.0 - (a - b) / bits


class ClipEmbedding:
    """Normalized CLIP image embedding; matched by cosine similarity."""

    name = "clip"
    default_threshold = 0.90  # placeholder; CLIP cosine runs high, needs calibration

    _model = None
    _preprocess = None

    @classmethod
    def _load(cls):
        """Load ViT-B-32 / openai once, lazily (torch import is expensive)."""
        if cls._model is None:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model.eval()
            cls._model, cls._preprocess = model, preprocess
        return cls._model, cls._preprocess

    def descriptor(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch
        model, preprocess = self._load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return max(0.0, float(np.dot(a, b)))


SIGNALS = {sig.name: sig for sig in (PerceptualHash, ClipEmbedding)}
DEFAULT_SIGNAL = "phash"


def get_signal(name: str):
    return SIGNALS[name]()
