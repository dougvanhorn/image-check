"""Invisible watermark embed/detect, built on the `invisible-watermark` library.

PB-658 proof of concept. We embed an invisible, recompression-resistant
signature into BuySell images so reused/duplicated listings can be detected
even after EXIF stripping, resizing, and recompression.

We use frequency-domain watermarking (DWT-DCT-SVD) rather than LSB steganography:
LSB hides data in raw pixel low-bits, which any resize or JPEG recompression
destroys -- exactly the edits scammers apply. Frequency-domain methods spread the
signal across the image's structure, which survives those edits far better.

Two ideas layered on top of the raw library:

  1. Presence detection (zero-bit). The real question for PB-658 is "has PB seen
     this image?", not "what 32 bits were stored". An image with no PB watermark
     decodes to ~50% bit-accuracy against our signature (chance); a watermarked
     image -- even after an attack -- scores well above that. So we threshold on
     bit-accuracy rather than demanding a bit-perfect decode.

  2. Repetition ECC. invisible-watermark carries raw bits with no error
     correction, so a couple of flipped bits after recompression break an exact
     match. We tile the signature N times and majority-vote the copies on decode,
     which recovers the signature through near-misses (e.g. JPEG q75).
"""

import dataclasses

import numpy as np

from imwatermark import WatermarkEncoder, WatermarkDecoder


# dwtDctSvd is the default: on real photos plain dwtDct is too marginal to
# recover even without an attack, while dwtDctSvd is bit-perfect through JPEG
# recompression. The ~3x embed-time cost is irrelevant here (we embed once, at
# upload). See the robustness harness for the measurements behind this choice.
DEFAULT_METHOD = "dwtDctSvd"
METHODS = ("dwtDct", "dwtDctSvd")

# Logical signature embedded in every PB image. Kept short (16 bits) so the
# repetition layer can fit several copies in the payload while staying small
# enough to survive. A decoded payload matching this means "PB has seen this".
#
# The bit pattern is deliberately BALANCED (8 ones, 8 zeros: 0xA5 0x5A). The
# library's decoder returns a zero-biased payload when no watermark is present,
# so a zero-heavy signature would match an un-marked image ~75% by chance and
# wreck the presence threshold. A balanced signature pins the noise floor at the
# true ~50% chance level. (Not ASCII text -- it's a signature, not a message.)
PB_SIGNATURE = b"\xa5\x5a"

# How many times the signature is tiled into the payload before embedding.
# Odd, so the per-bit majority vote never ties. 3 copies of a 16-bit signature
# is a 48-bit payload -- comfortably within dwtDctSvd's reliable capacity.
REPETITIONS = 3

# A recovered payload whose bit-accuracy against the (tiled) signature meets or
# exceeds this is treated as "watermark present". 0.5 is chance; an un-marked
# photo scores ~0.5, so this leaves margin above the noise floor. Calibrate
# against the harness's control column on real data before trusting it.
PRESENCE_THRESHOLD = 0.75


def bit_length(payload: bytes) -> int:
    return len(payload) * 8


def _to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits).tobytes()


def bit_accuracy(expected: bytes, actual: bytes) -> float:
    """Fraction of bits matching between expected and recovered payloads.

    ~0.5 is chance (no watermark recovered); ~1.0 is a clean recovery. This is
    the presence signal -- more informative than exact match alone.
    """
    eb, ab = _to_bits(expected), _to_bits(actual)
    n = min(eb.size, ab.size)
    if n == 0:
        return 0.0
    return float((eb[:n] == ab[:n]).mean())


# --- Raw single-shot embed/decode (no ECC) ----------------------------------

def embed(image: np.ndarray, payload: bytes, method: str = DEFAULT_METHOD) -> np.ndarray:
    """Embed raw `payload` bytes into a BGR image; return the watermarked array."""
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", payload)
    return encoder.encode(image, method)


def decode(image: np.ndarray, payload_bits: int, method: str = DEFAULT_METHOD) -> bytes:
    """Recover a `payload_bits`-bit payload from a (possibly attacked) image."""
    decoder = WatermarkDecoder("bytes", payload_bits)
    return decoder.decode(image, method)


# --- Repetition ECC over the signature --------------------------------------

def encode_signature(signature: bytes = PB_SIGNATURE, reps: int = REPETITIONS) -> bytes:
    """Tile the signature `reps` times into the payload that actually gets embedded."""
    return _to_bytes(np.tile(_to_bits(signature), reps))


def _majority_vote(raw: bytes, signature: bytes, reps: int) -> bytes:
    """Recover the signature by majority-voting its repeated copies in `raw`."""
    n = bit_length(signature)
    grid = _to_bits(raw)[:n * reps].reshape(reps, n)
    return _to_bytes((grid.mean(axis=0) >= 0.5).astype(np.uint8))


@dataclasses.dataclass
class DetectionResult:
    """Outcome of decoding and scoring a watermark against the expected signature."""
    signature: bytes
    recovered: bytes | None   # ECC majority-voted bytes
    exact_match: bool         # recovered == signature (full payload survived)
    bit_accuracy: float       # raw decode vs tiled signature (the presence signal)
    present: bool             # bit_accuracy >= threshold
    error: str | None = None


# --- Deployment-level embed/detect (ECC + presence) -------------------------

def embed_watermark(image: np.ndarray, signature: bytes = PB_SIGNATURE,
                    method: str = DEFAULT_METHOD, reps: int = REPETITIONS) -> np.ndarray:
    """Embed the ECC-coded PB signature into an image."""
    return embed(image, encode_signature(signature, reps), method)


def detect_watermark(image: np.ndarray, signature: bytes = PB_SIGNATURE,
                     method: str = DEFAULT_METHOD,
                     threshold: float = PRESENCE_THRESHOLD,
                     reps: int = REPETITIONS) -> DetectionResult:
    """Decode, ECC-correct, and score the watermark against `signature`.

    Returns both a presence verdict (bit-accuracy >= threshold) and an exact
    payload-recovery flag. Decode failures (e.g. an over-cropped image) are
    captured as an `error` and scored as a miss rather than raised.
    """
    expected = encode_signature(signature, reps)
    try:
        raw = decode(image, bit_length(expected), method)
    except Exception as exc:  # noqa: BLE001 -- POC: any failure is just a miss
        return DetectionResult(signature, None, False, 0.0, False, error=str(exc))
    acc = bit_accuracy(expected, raw)
    recovered = _majority_vote(raw, signature, reps)
    return DetectionResult(
        signature=signature,
        recovered=recovered,
        exact_match=recovered == signature,
        bit_accuracy=acc,
        present=acc >= threshold,
    )
