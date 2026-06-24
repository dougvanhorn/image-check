# PB-658 Session Summary — Image Duplicate Detection POC

Handoff doc for continuing in a fresh context. Covers everything done this session.

## The problem (Linear PB-658)

Scammers reuse stolen Pinkbike BuySell listing photos. They strip EXIF and
**resize/crop** images to defeat traditional duplicate checks. The ticket asked
to explore steganography/watermarking (OpenStego, `stegano`) to embed an
invisible signature so reused images can be detected on upload.

## What we concluded

The threat model (resize/crop/recompress) splits cleanly across two technique
families, and **neither alone is sufficient — they're complementary**:

1. **Invisible watermarking** (`ic.stega`) — survives recompression/re-upload,
   carries an identifying payload, near-zero false positives. **Destroyed by any
   resize or crop.** Only works going forward (must embed at upload); useless on
   the existing photo backlog.
2. **Geometry-tolerant content signals** (`ic.phash`) — perceptual hash + CLIP
   embedding. Match by similarity, so inherently tolerant to resize/crop/
   recompress. Work on the existing backlog. CLIP has a same-model-different-bike
   false-positive risk (matches by *meaning*).

**LSB steganography (`stegano`, OpenStego data-hiding) was discarded** — destroyed
by any JPEG re-save/resize/crop.

## Packages built (all under `src/ic/`)

### `ic.stega` — invisible watermarking (CLI: `stega`)
- `watermark.py` — DWT-DCT-SVD via the `invisible-watermark` library. Embed/decode,
  repetition ECC (3x), presence detection by bit-accuracy threshold.
- `attacks.py` — generic image-edit gauntlet (jpeg recompress, resize, crop). **Shared**
  by `ic.phash`.
- `harness.py` — embed → attack → measure recovery, with un-watermarked control.
- `cli.py` — `stega embed | detect | attack`.

Key implementation notes:
- Default method is **`dwtDctSvd`** (plain `dwtDct` is too marginal — failed even at
  baseline on a real photo).
- `PB_SIGNATURE = b"\xa5\x5a"` is **balanced** (8 ones / 8 zeros) on purpose: the
  decoder returns a zero-biased payload when no watermark is present, so a zero-heavy
  signature gives a false ~75% noise floor. Balanced pins the floor at true ~50%.

### `ic.phash` — geometry-tolerant signals (CLI: `phash`)
- `signals.py` — two swappable signals with a common `descriptor`/`similarity` interface:
  - `PerceptualHash` (pHash via `imagehash`) — light, no model.
  - `ClipEmbedding` (open_clip `ViT-B-32` / `openai`, lazy-loaded; same model as `ic.video`).
- `harness.py` — reuses `ic.stega.attacks`; measures whether an attacked copy still
  matches the original. Has `--save-dir` support (writes tested images as lossless PNG).
- `cli.py` — `phash hash | compare | attack`.

## Measured results (on `var/stega/real-bike-01.jpg`, 800x600)

Same attack gauntlet across all three approaches:

| Attack | stega watermark | phash | clip |
|---|---|---|---|
| jpeg q90 | exact | 100% | 99.5% |
| jpeg q75 | exact | 100% | 97.5% |
| jpeg q50 | **gone** | 100% | 97.8% |
| resize 75% | **gone** | 100% | 99.9% |
| resize 50% | **gone** | 100% | 100% |
| crop 90% | **gone** | 90.6% | 98.8% |
| crop 75% | **gone** | **62.5% (fail)** | 95.2% |
| resize 50% + q75 | **gone** | 100% | 95.3% |

- Watermark: binary — perfect through JPEG q75, then gone. Dies on all geometry.
- pHash: survives everything except heavy crop (crop-75% is its one failure).
- CLIP: survives everything, including heavy crop. Strictly more robust, but heavy
  (torch) and semantic (false-positive risk).

Tested images saved in `var/phash/test-images/` (00–08, lossless PNG = exact pixels scored).

## Scale architecture (for hundreds of thousands of existing photos)

Don't do O(N) pairwise. Build a searchable **index** of descriptors; each upload is
one **approximate-nearest-neighbor (ANN)** query against the whole corpus (like
reverse-image search). At this scale it fits in RAM, queries in single-digit ms.

- pHash: 64-bit int, Hamming distance — brute-force popcount handles 1M in ms.
- CLIP: 512-dim vector, ~2 GB at 1M — FAISS (HNSW/IVF), hnswlib, or **pgvector**
  (co-locate with listing data; easiest incremental inserts / lowest ops).

**4-step pipeline (recall → verify):**
1. **Watermark check** (`ic.stega`) — present+match = known image, zero FP. Forward-only.
2. **pHash index** — catch exact / near-identical re-uploads. Cheap, instant.
3. **CLIP ANN index** — recall layer for resized/cropped/re-encoded. Returns top-k candidates.
4. **Verify candidates** — precise identity check (ORB keypoint match from `ic.video`,
   and/or watermark) to confirm *same physical bike* and kill same-model false positives.

Net: index = recall against all photos; expensive same-bike confirmation only on the
few candidates returned.

## Open items / next steps

1. **False-positive measurement (blocker).** All `phash` numbers so far are recall only
   (true copy still matches). Need a **second, different bike photo (ideally same model)**
   dropped in `var/stega/` or `var/phash/` to measure separation between
   "same image edited" vs "different bike same model". This decides whether a usable
   threshold exists. Use `phash compare A B`.
2. **Threshold calibration** — set CLIP cosine cutoff + top-k against labeled pairs.
3. **Build the index POC** — extend `ic.phash` with `index build <dir>` / `index query <image>`
   backed by FAISS or pgvector; measure query latency + FP rate on a real batch.
4. **RivaGAN** (untested) — `invisible-watermark`'s CNN mode, claimed better crop survival;
   could add to `ic.stega` METHODS.
5. All thresholds are placeholders pending real labeled data.

## Linear tickets (parent PB-658)

- **PB-784** — POC findings: library limitations (stegano/OpenStego/invisible-watermark).
  https://linear.app/outside/issue/PB-784
- **PB-785** — Approach: index-based duplicate detection at corpus scale (pHash + CLIP ANN).
  https://linear.app/outside/issue/PB-785

## How to run

```bash
uv sync                                          # after any pyproject change

# Watermark
uv run stega embed  IN.jpg OUT.png
uv run stega detect OUT.png
uv run stega attack IN.jpg [--method dwtDct|dwtDctSvd]

# Geometry-tolerant signals
uv run phash hash    IN.jpg [--signal phash|clip]
uv run phash compare A.jpg B.jpg [--signal phash|clip]   # <-- use for FP test
uv run phash attack  IN.jpg [--signal phash|clip] [--save-dir DIR]
```

Entry points registered in `pyproject.toml`: `video`, `stega`, `phash`.
Deps added this session: `invisible-watermark`, `imagehash`.

## pHash internals & index-search design (POC prep)

Captured from the follow-up discussion, to ground the index POC.

### How a pHash is created (the DCT `imagehash.phash` variant)
1. **Grayscale** — drop color (fragile, structure-irrelevant).
2. **Downscale to 32×32** — the key robustness step; destroys fine detail, JPEG
   artifacts, resolution differences, so a resized/recompressed copy collapses to
   nearly the same small image.
3. **2D DCT** (same transform JPEG uses) — re-express as frequency components;
   top-left = low frequency (broad structure), bottom-right = high frequency (noise/edges).
4. **Keep the top-left 8×8 block** — the 64 lowest-frequency coefficients, where
   identity lives and is most stable under edits.
5. **Median threshold** — each of the 64 coefficients becomes 1 bit: `1` if above the
   median of the block, else `0`. Median (not mean/zero) is what survives brightness/
   contrast shifts — they move all coefficients together, preserving the above/below pattern.

Output: a **64-bit fingerprint**. Compare by **Hamming distance** = `popcount(a XOR b)`:
0 = identical structure, a few bits = edited copy, ~32 = unrelated.
(`ahash`/`dhash` are cheaper, less robust; `phash` is the DCT one to use.)

### Searching a corpus for Hamming neighbors (Postgres)
Store the hash as a **`BIGINT`** (not a string). Three regimes, cheapest first:
- **Exact dup:** `WHERE phash = :h` — B-tree, O(log n). Catches byte-identical re-uploads.
- **Near-dup, brute force** (viable at ≤1M): `ORDER BY bit_count(phash # :h) LIMIT k`
  (`#` = XOR; `bit_count` native in PG14+). Seq scan over 1M BIGINTs is ms-to-low-hundreds-ms.
  **Start here.**
- **Near-dup, sublinear — Multi-Index Hashing (MIH):** split the 64-bit hash into 4×16-bit
  indexed columns `b0..b3`. Pigeonhole: two hashes within ≤3 bits total must share ≥1 band
  *exactly*, so a near-match becomes 4 exact indexed lookups (UNION of the 4 legs), then
  verify true Hamming on the small candidate set. The TinEye-style scaling path; not needed
  at 1M but available.

### Important correction to the cascade idea
A strict **pHash-first → CLIP** cascade is a *lossy funnel*: pHash failed the 75% crop
(62.5%), so a cropped stolen photo never becomes a candidate and CLIP never sees it. So
pHash and CLIP should run as **two parallel candidate generators**, not a strict cascade:
- **pHash index** — exact + near-identical re-uploads (cheap, high precision, misses crops).
- **CLIP ANN index** — its *own* nearest-neighbor search (FAISS/pgvector, precomputed
  embeddings); catches resized/cropped/re-encoded. Does not depend on pHash to find anything.
- **Union** both candidate sets → run precise **ORB / watermark verification** on the union → decide.

pHash's real value is being nearly free and slashing how often the heavy CLIP+ORB
verification runs — not gating CLIP's recall. The CLIP ANN index must exist independently.

### POC to build next (the `index build` / `index query` work — item 3 above)
- **Storage:** Postgres. pHash as `BIGINT` column (+ optional 4 banded columns for MIH);
  CLIP 512-d embedding via **pgvector** (HNSW index, cosine), co-located with listing metadata.
  Lowest-ops choice, easy incremental inserts. (FAISS/hnswlib in-memory is the alternative.)
- **`index build <dir>`** — batch-embed a directory of photos: compute pHash + CLIP for each,
  upsert into the tables. (CLIP ViT-B-32 ~ hundreds/sec on GPU.)
- **`index query <image>`** — compute both descriptors; run the pHash Hamming search and the
  pgvector cosine search; union candidates; (later) verify with ORB from `ic.video`.
- **Measure:** candidate-set sizes, recall (do known edited copies surface?), query latency,
  and — the blocker — the **false-positive rate** against same-model-different-bike images.
- **Still need:** a second/different bike photo (ideally same model) to measure separation
  and calibrate thresholds (CLIP cosine cutoff, pHash Hamming radius, top-k).

## Note on this file's location

This summary lives in `src/ic/phash_v2/` — a new directory created for the handoff.
There is no `phash_v2` package yet; if the index work (item 3) becomes its own package,
this is the natural home. Otherwise the index commands can extend the existing `ic.phash`.
