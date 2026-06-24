"""Robustness harness for the geometry-tolerant signals.

Mirrors `ic.stega.harness`, but the question is inverted. Watermarking asks "can
I still recover the embedded mark?"; a matching signal asks "does the attacked
copy still match the original reference?". We reuse the *same* attack gauntlet so
the two approaches can be compared directly on identical edits.
"""

import dataclasses
import pathlib
import re

import cv2
import numpy as np

from ic.stega import attacks  # shared, generic image-edit gauntlet
from ic.phash import signals


@dataclasses.dataclass
class MatchOutcome:
    """Whether the attacked copy still matches the original under one attack."""
    name: str
    similarity: float
    matched: bool
    saved_path: str | None = None
    error: str | None = None


def _slug(name: str) -> str:
    """Turn an attack label into a filesystem-safe stem, e.g. 'resize 50%' -> 'resize_50'."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def run_gauntlet(original: np.ndarray, signal, threshold: float,
                 gauntlet=None, save_dir: str | None = None) -> list[MatchOutcome]:
    """Compute the reference descriptor, then for each attack measure whether the
    attacked image still matches the original above `threshold`.

    If `save_dir` is given, each attacked image is written there as PNG (lossless,
    so the saved file exactly matches the pixels that were scored).
    """
    gauntlet = gauntlet or attacks.default_gauntlet()
    reference = signal.descriptor(original)

    out_dir = None
    if save_dir:
        out_dir = pathlib.Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    outcomes = []
    for i, (name, attack) in enumerate(gauntlet):
        try:
            attacked = attack(original)
            saved = None
            if out_dir is not None:
                path = out_dir / f"{i:02d}_{_slug(name)}.png"
                cv2.imwrite(str(path), attacked)
                saved = str(path)
            sim = signal.similarity(reference, signal.descriptor(attacked))
            outcomes.append(MatchOutcome(name, sim, sim >= threshold, saved_path=saved))
        except Exception as exc:  # noqa: BLE001 -- POC: log and keep going
            outcomes.append(MatchOutcome(name, 0.0, False, error=str(exc)))
    return outcomes
