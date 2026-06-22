"""Robustness harness: embed a watermark, attack it, measure survival.

This is the heart of the PB-658 evaluation. Neither invisible-watermark nor any
other tool guarantees the watermark survives the resize/recompress/crop edits
scammers apply, so the only honest way to choose an approach is to measure it.

Each attack is also run against the *un-watermarked* original (the "control") to
establish the bit-accuracy noise floor -- the score a clean photo gets against
our signature by chance. The presence threshold only means something relative to
that floor.
"""

import dataclasses

import numpy as np

from ic.stega import attacks
from ic.stega import watermark


@dataclasses.dataclass
class AttackOutcome:
    """How the watermark fared against a single attack, plus the control floor."""
    name: str
    result: watermark.DetectionResult  # watermarked image, attacked
    control_accuracy: float            # un-watermarked image, same attack


def run_gauntlet(original: np.ndarray, signature: bytes = watermark.PB_SIGNATURE,
                 method: str = watermark.DEFAULT_METHOD,
                 threshold: float = watermark.PRESENCE_THRESHOLD,
                 gauntlet=None) -> list[AttackOutcome]:
    """Embed into `original`, then for each attack score the watermarked image
    and the un-watermarked control under the identical edit.
    """
    gauntlet = gauntlet or attacks.default_gauntlet()
    watermarked = watermark.embed_watermark(original, signature, method)

    outcomes = []
    for name, attack in gauntlet:
        try:
            result = watermark.detect_watermark(attack(watermarked), signature, method, threshold)
        except Exception as exc:  # noqa: BLE001 -- POC: log the failure, keep going
            result = watermark.DetectionResult(signature, None, False, 0.0, False, error=str(exc))
        try:
            control = watermark.detect_watermark(attack(original), signature, method, threshold).bit_accuracy
        except Exception:  # noqa: BLE001
            control = float("nan")
        outcomes.append(AttackOutcome(name=name, result=result, control_accuracy=control))
    return outcomes
