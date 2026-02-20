"""
Receipt Forgery Detection — Three-Judge Gemini Ensemble
========================================================
Dataset : Find It Again Receipt Dataset
          Expected folder structure:
              data/
              ├── real/
              │   ├── receipt_001.png
              │   └── ...
              └── fake/
                  ├── receipt_042.png
                  └── ...

Judges  : gemini-2.0-flash × 3
            · Judge 1 — Visual Anomaly Inspector    (temp=0.1)
            · Judge 2 — Semantic Logic Auditor      (temp=0.5)
            · Judge 3 — Layout Forensics Analyst    (temp=0.8)

Voting  : Confidence aggregation (0–100 score)
Output  : { label, confidence, reasons }
          + CSV results log  (results/predictions.csv)
          + JSON full log    (results/predictions.json)
"""

import base64
import csv
import json
import kaggle
import logging
import os
import random
import re
import shutil
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/run.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME:     str = "gemini-2.0-flash"

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

DATASET_ROOT: Path = Path("dataset")          # root folder of the dataset
RESULTS_DIR:  Path = Path("results")          # where outputs are saved
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
#
# Google AI Studio (free tier) limits for gemini-2.0-flash:
#   · 15 requests per minute  (RPM)
#   · 1 000 000 tokens per minute  (TPM)  — not tracked here; images are small
#   · 1 500 requests per day   (RPD)
#
# We apply a generous safety margin on the RPM window so we never hit a 429.
# The limiter is thread-safe and uses a sliding-window token-bucket approach:
# it records the timestamp of every request dispatched in the last 60 s and
# blocks (with a polite sleep + log) when the bucket is full.

RATE_LIMIT_RPM:        int   = 8          # max requests per 60-second window (53 % of 15)
RATE_LIMIT_WINDOW_SEC: float = 60.0        # sliding window duration in seconds
RATE_LIMIT_MIN_GAP_SEC: float = 4.0        # minimum gap between any two consecutive calls
RATE_LIMIT_RETRY_MAX:  int   = 5           # max retries on HTTP 429 / ResourceExhausted
RATE_LIMIT_RETRY_BASE: float = 10.0        # base back-off seconds (doubles each retry)


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter for the Gemini API.

    Tracks the timestamp of every dispatched request in a deque.
    Before each call, it:
      1. Enforces a minimum inter-call gap (RATE_LIMIT_MIN_GAP_SEC).
      2. Purges timestamps older than the sliding window.
      3. Blocks with an informative sleep until the window has room.

    Usage::

        limiter = RateLimiter()
        limiter.acquire()           # blocks if needed
        response = model.generate_content(...)
        limiter.record()            # stamp the request
    """

    def __init__(
        self,
        max_rpm:    int   = RATE_LIMIT_RPM,
        window_sec: float = RATE_LIMIT_WINDOW_SEC,
        min_gap:    float = RATE_LIMIT_MIN_GAP_SEC,
    ) -> None:
        self.max_rpm    = max_rpm
        self.window_sec = window_sec
        self.min_gap    = min_gap
        self._lock      = threading.Lock()
        self._timestamps: deque[float] = deque()

    # ── public interface ──────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Block until a request slot is available, then return."""
        with self._lock:
            self._enforce_min_gap()
            self._enforce_rpm_window()

    def record(self) -> None:
        """Record that a request was just dispatched (call after acquire())."""
        with self._lock:
            self._timestamps.append(time.monotonic())

    @property
    def current_rpm(self) -> int:
        """Return the number of requests dispatched in the current window."""
        with self._lock:
            self._purge_old()
            return len(self._timestamps)

    # ── private helpers ───────────────────────────────────────────────────────

    def _purge_old(self) -> None:
        """Remove timestamps outside the sliding window (call inside lock)."""
        cutoff = time.monotonic() - self.window_sec
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def _enforce_min_gap(self) -> None:
        """Sleep until the minimum inter-call gap has elapsed (inside lock)."""
        if self._timestamps:
            elapsed = time.monotonic() - self._timestamps[-1]
            gap     = self.min_gap - elapsed
            if gap > 0:
                logger.debug(f"[RATE LIMITER] Min-gap pause: {gap:.2f}s")
                time.sleep(gap)

    def _enforce_rpm_window(self) -> None:
        """Block until the sliding window has capacity (inside lock)."""
        while True:
            self._purge_old()
            if len(self._timestamps) < self.max_rpm:
                break
            # Oldest request will age out at this time
            oldest  = self._timestamps[0]
            wait    = (oldest + self.window_sec) - time.monotonic() + 0.5   # +0.5 s buffer
            logger.warning(
                f"[RATE LIMITER] RPM window full "
                f"({len(self._timestamps)}/{self.max_rpm} requests in last {self.window_sec:.0f}s). "
                f"Sleeping {wait:.1f}s…"
            )
            time.sleep(max(wait, 1.0))


# Module-level singleton — shared across all call_judge() invocations
_rate_limiter = RateLimiter()

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

LabelType = Literal["FAKE", "REAL", "UNCERTAIN"]


class JudgeResponse(BaseModel):
    label:            LabelType
    confidence:       int = Field(..., ge=0, le=100)
    reason:           str                  # one concise sentence summarising the verdict
    evidence_regions: list[str]            # short spatial descriptions of suspicious areas

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: int) -> int:
        return max(0, min(100, v))


class FinalVerdict(BaseModel):
    label:      LabelType
    vote_tally: dict[str, int]   # e.g. {"REAL": 2, "FAKE": 1, "UNCERTAIN": 0}


class ReceiptResult(BaseModel):
    image_path:    str
    ground_truth:  Optional[LabelType]   # inferred from folder name (real/fake)
    verdict:       FinalVerdict
    judge_details: dict[str, JudgeResponse]
    timestamp:     str


# ---------------------------------------------------------------------------
# Judge System Prompts
# ---------------------------------------------------------------------------

SHARED_JSON_INSTRUCTIONS = """
Respond ONLY with a single valid JSON object — no markdown, no prose, no code fences.

Schema:
{
  "label": "FAKE" | "REAL" | "UNCERTAIN",
  "confidence": <integer 0–100>,
  "reason": "<one concise sentence summarising why you assigned this label>",
  "evidence_regions": [
    "<short spatial description of a suspicious area, e.g. 'top-right logo area', 'total field row', 'merchant address block'>",
    ...
  ]
}

Field rules:
  - "reason"           : exactly one sentence, ≤ 25 words, no bullet points
  - "evidence_regions" : 1–4 items; each is a brief region label (2–8 words);
                         leave as [] if the receipt appears genuine and no anomaly region was found

Confidence scoring guide:
  85–100 : Multiple clear, unambiguous indicators found
  65–84  : Consistent but not overwhelming evidence
  40–64  : Mixed or inconclusive evidence
  20–39  : Very few or weak signals
  0–19   : Cannot determine anything meaningful from this receipt
"""

JUDGE_PROMPTS: dict[str, tuple[str, float]] = {
    "visual": (
        f"""You are a forensic imaging specialist for document fraud detection.
Your ONLY job is to detect VISUAL anomalies in receipt images.

Focus exclusively on:
- Font rendering inconsistencies (mixed typefaces, irregular kerning, spacing)
- Copy-paste or clone-stamp artifacts anywhere on the document
- Logo or stamp quality degradation, resampling, or blurring
- Unnatural PNG compression blocks or pixel noise
- Inconsistent ink density or color profiles across the document
- Signs of digital erasure, smearing, or overdrawing
- Lighting and shadow inconsistencies across the document surface

{SHARED_JSON_INSTRUCTIONS}""",
        0.1,
    ),
    "semantic": (
        f"""You are a forensic accountant and document fraud analyst.
Your ONLY job is to verify the semantic and numerical coherence of receipts.

Focus exclusively on:
- Subtotal + tax = total arithmetic validation (flag any discrepancy, even $0.01)
- Unit price × quantity = line item total for every row
- Tax rate plausibility for the merchant's stated jurisdiction
- Date/time logical consistency (future dates, impossible times)
- Merchant name, address, and phone number cross-coherence
- Item names that are implausible for the merchant category
- Discounts or promo codes producing mathematically impossible results
- Currency symbols or decimal separators used inconsistently

{SHARED_JSON_INSTRUCTIONS}""",
        0.5,
    ),
    "layout": (
        f"""You are a document layout forensics expert specialising in receipt structure.
Your ONLY job is to detect structural and spatial anomalies in receipts.

Focus exclusively on:
- Column and field misalignment across the receipt body
- Inconsistent margin widths or padding between sections
- Irregular or mixed line spacing within uniform text blocks
- Section headers that deviate from standard POS system templates
- Mixed or inconsistent font families within the same logical section
- Unusual whitespace insertions suggesting content removal or insertion
- Receipt structure deviating from known thermal-printer or POS templates
- Bounding-box irregularities in itemised rows

{SHARED_JSON_INSTRUCTIONS}""",
        0.8,
    ),
}

def downloader(_download_path: Path):
    # Authenticate the API (it will use the kaggle.json file automatically)
    kaggle.api.authenticate()

    # Define the dataset identifier and download path
    dataset_identifier = "nikita2998/find-it-again-dataset" # Replace with your dataset's identifier

    # Download the dataset files and unzip them
    kaggle.api.dataset_download_files(dataset_identifier, path=str(_download_path), unzip=True)

    logger.info(f"Dataset '{dataset_identifier}' downloaded to '{str(_download_path)}' and unzipped.")

def format_dataset(dataset_root: Path):
    splits = ['train', 'val', 'test']
    os.makedirs(dataset_root / "real_full_sample", exist_ok=True)
    os.makedirs(dataset_root / "fake_full_sample", exist_ok=True)
    for split in splits:
        with open(dataset_root / "findit2" / f"{split}.txt", "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                origin_img_path = dataset_root / "findit2" / split / line.split(",")[0]
                if line.split(",")[3] == "0":
                    dest_img_path = dataset_root / "real_full_sample" / line.split(",")[0]
                if line.split(",")[3] == "1":
                    dest_img_path = dataset_root / "fake_full_sample" / line.split(",")[0]
                if os.path.exists(origin_img_path):
                    shutil.copy(origin_img_path, dest_img_path)

def sample_dataset(
    dataset_root: Path = DATASET_ROOT,
    sample_size: int = 10,
) -> list[str]:
    """
    Implement a sampling method to select a subset of 20 images (10 real + 10 fake) from the Find It Again Receipt Dataset.
    METHOD: Selecting multiple unique random elements

    Args:
        dataset_root : Path to the dataset root (must contain real/ and fake/).
        sample_size  : Number of images to sample from each category (default: 10).
    """
    random.seed(42)

    real_sample_dirname = dataset_root / "real_full_sample"
    real_receipts = [img for img in os.listdir(real_sample_dirname) if img.endswith(".png")]
    fake_sample_dirname = dataset_root / "fake_full_sample"
    fake_receipts = [img for img in os.listdir(fake_sample_dirname) if img.endswith(".png")]
    
    selected_real_receipts = random.sample(real_receipts, min(sample_size, len(real_receipts)))
    selected_fake_receipts = random.sample(fake_receipts, min(sample_size, len(fake_receipts)))
    
    selected_receipts = selected_real_receipts + selected_fake_receipts
    logger.info(f"Selected {selected_real_receipts} real and {selected_fake_receipts} fake receipts.")

    if not selected_receipts:
        logger.error("No images found. Check your dataset path and folder structure.")
        return []

    for img in selected_real_receipts:
        os.makedirs(dataset_root / "real", exist_ok=True)
        origin_path = dataset_root / "real_full_sample" / img
        dest_path = dataset_root / "real" / img
        shutil.copy(origin_path, dest_path)

    for img in selected_fake_receipts:
        os.makedirs(dataset_root / "fake", exist_ok=True)
        origin_path = dataset_root / "fake_full_sample" / img
        dest_path = dataset_root / "fake" / img
        shutil.copy(origin_path, dest_path)

# ---------------------------------------------------------------------------
# Image Loading
# ---------------------------------------------------------------------------

def load_image_as_base64(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, mime_type) for a local image file."""
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
        ".gif":  "image/gif",
    }
    mime_type = mime_map.get(image_path.suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


# ---------------------------------------------------------------------------
# Ground Truth Inference
# ---------------------------------------------------------------------------

def infer_ground_truth(image_path: Path) -> Optional[LabelType]:
    """
    Infer the ground-truth label from the parent folder name.
    Expects the image to live inside a 'real/' or 'fake/' directory.
    """
    parts = [p.lower() for p in image_path.parts]
    if "real" in parts:
        return "REAL"
    if "fake" in parts:
        return "FAKE"
    return None


# ---------------------------------------------------------------------------
# Dataset Discovery
# ---------------------------------------------------------------------------

def discover_dataset(dataset_root: Path) -> list[Path]:
    """
    Walk dataset_root and return all image paths found in real/ and fake/ subfolders.

    Expected structure:
        dataset_root/
        ├── real/   ← REAL receipts
        └── fake/   ← FAKE receipts
    """
    images: list[Path] = []
    for subfolder in ("real", "fake"):
        folder = dataset_root / subfolder
        if not folder.exists():
            logger.warning(f"Subfolder not found: {folder}")
            continue
        found = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        logger.info(f"Found {len(found)} images in {folder}")
        images.extend(found)

    logger.info(f"Total images discovered: {len(images)}")
    return sorted(images)


# ---------------------------------------------------------------------------
# Single Judge Call
# ---------------------------------------------------------------------------

def call_judge(
    judge_name: str,
    system_prompt: str,
    temperature: float,
    image_data: str,
    mime_type: str,
) -> JudgeResponse:
    """
    Call one Gemini judge synchronously, respecting the shared rate limiter.

    Retry policy (exponential back-off):
        Attempt 1  — immediate (after rate-limiter gate)
        Attempt 2  — wait RATE_LIMIT_RETRY_BASE   × 2^0  seconds
        Attempt 3  — wait RATE_LIMIT_RETRY_BASE   × 2^1  seconds
        …up to RATE_LIMIT_RETRY_MAX attempts total.

    Any ResourceExhausted / 429 response resets the rate-limiter window
    by injecting a full window's worth of phantom timestamps so subsequent
    calls slow down automatically.
    """
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )

    image_part  = {"mime_type": mime_type, "data": image_data}
    prompt_text = "Analyse this receipt image and return your forensic verdict as JSON."

    last_exc: Exception | None = None

    for attempt in range(1, RATE_LIMIT_RETRY_MAX + 1):

        # ── Gate: wait for a slot in the sliding window ───────────────────────
        _rate_limiter.acquire()

        try:
            logger.debug(
                f"[{judge_name.upper()}] API call attempt {attempt}/{RATE_LIMIT_RETRY_MAX} "
                f"(window RPM={_rate_limiter.current_rpm}/{RATE_LIMIT_RPM})"
            )
            response = model.generate_content([image_part, prompt_text])

            # ── Success: stamp the request and parse ──────────────────────────
            _rate_limiter.record()

            raw_text = response.text.strip()
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
            raw_text = re.sub(r"\s*```$",           "", raw_text, flags=re.MULTILINE)

            parsed = json.loads(raw_text)
            return JudgeResponse(**parsed)

        except Exception as exc:
            last_exc    = exc
            exc_str     = str(exc).lower()
            is_rate_err = any(kw in exc_str for kw in (
                "resourceexhausted", "429", "quota", "rate limit", "too many requests"
            ))

            if is_rate_err:
                # Flood the limiter's window so all subsequent calls slow down
                with _rate_limiter._lock:
                    now = time.monotonic()
                    for _ in range(RATE_LIMIT_RPM):
                        _rate_limiter._timestamps.append(now)

                backoff = RATE_LIMIT_RETRY_BASE * (2 ** (attempt - 1))
                logger.warning(
                    f"[{judge_name.upper()}] Rate-limit hit on attempt {attempt}. "
                    f"Back-off {backoff:.0f}s before retry…"
                )
                time.sleep(backoff)

            elif attempt < RATE_LIMIT_RETRY_MAX:
                # Transient error (network hiccup, timeout) — short fixed retry
                backoff = RATE_LIMIT_RETRY_BASE
                logger.warning(
                    f"[{judge_name.upper()}] Transient error on attempt {attempt}: {exc}. "
                    f"Retrying in {backoff:.0f}s…"
                )
                time.sleep(backoff)

            else:
                # Non-rate error on final attempt — fall through to fallback
                logger.error(f"[{judge_name.upper()}] All {RATE_LIMIT_RETRY_MAX} attempts failed: {exc}")
                break

    # All retries exhausted
    logger.error(f"[{judge_name.upper()}] Returning UNCERTAIN after exhausted retries. Last error: {last_exc}")
    return JudgeResponse(
        label="UNCERTAIN",
        confidence=0,
        reasons=[f"API call failed after {RATE_LIMIT_RETRY_MAX} attempts: {last_exc}"],
    )


# ---------------------------------------------------------------------------
# Voting Aggregator
# ---------------------------------------------------------------------------

def aggregate_votes(judges: dict[str, JudgeResponse]) -> FinalVerdict:
    """
    Pure majority-vote aggregator across the three judges.

    Decision rules (in priority order):
      · REAL      — at least 2 judges voted REAL
      · FAKE      — at least 2 judges voted FAKE
      · UNCERTAIN — no label reached 2 votes (split across all three labels,
                    or too many UNCERTAIN votes prevented a clear majority)

    The vote_tally records the raw per-label head count for every judge call.
    """
    tally: dict[LabelType, int] = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}

    for verdict in judges.values():
        tally[verdict.label] += 1

    # Majority rule: need ≥ 2 votes for REAL or FAKE to win
    if tally["REAL"] >= 2:
        winner: LabelType = "REAL"
    elif tally["FAKE"] >= 2:
        winner = "FAKE"
    else:
        # Covers: 1-1-1 three-way split, or 2+ UNCERTAIN votes
        winner = "UNCERTAIN"

    logger.info(
        f"  [VOTE] REAL={tally['REAL']}  FAKE={tally['FAKE']}  "
        f"UNCERTAIN={tally['UNCERTAIN']}  →  {winner}"
    )

    return FinalVerdict(label=winner, vote_tally=dict(tally))


# ---------------------------------------------------------------------------
# Single Receipt Analysis
# ---------------------------------------------------------------------------

def analyse_receipt(image_path: Path) -> ReceiptResult:
    """Call all three judges sequentially on one receipt image and aggregate their verdicts."""
    image_data, mime_type = load_image_as_base64(image_path)

    judge_results: dict[str, JudgeResponse] = {}
    for judge_name, (system_prompt, temperature) in JUDGE_PROMPTS.items():
        logger.info(f"  [{judge_name.upper()} JUDGE] Running… (temp={temperature})")
        judge_results[judge_name] = call_judge(
            judge_name=judge_name,
            system_prompt=system_prompt,
            temperature=temperature,
            image_data=image_data,
            mime_type=mime_type,
        )
        logger.info(
            f"  [{judge_name.upper()} JUDGE] "
            f"{judge_results[judge_name].label} "
            f"(confidence={judge_results[judge_name].confidence})"
        )

    verdict      = aggregate_votes(judge_results)
    ground_truth = infer_ground_truth(image_path)

    return ReceiptResult(
        image_path=str(image_path),
        ground_truth=ground_truth,
        verdict=verdict,
        judge_details=judge_results,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Results Persistence
# ---------------------------------------------------------------------------

def save_results(results: list[ReceiptResult]) -> None:
    """Save results to both CSV and JSON in the results/ directory."""

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "predictions.json"
    serialisable = []
    for r in results:
        serialisable.append({
            "image_path":   r.image_path,
            "ground_truth": r.ground_truth,
            "verdict": {
                "label":      r.verdict.label,
                "vote_tally": r.verdict.vote_tally,
            },
            "judge_details": {
                name: {
                    "label":            j.label,
                    "confidence":       j.confidence,
                    "reason":           j.reason,
                    "evidence_regions": j.evidence_regions,
                }
                for name, j in r.judge_details.items()
            },
            "timestamp": r.timestamp,
        })
    json_path.write_text(json.dumps(serialisable, indent=2))
    logger.info(f"JSON results saved → {json_path}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "ground_truth",
            "predicted_label",
            "votes_real", "votes_fake", "votes_uncertain",
            "visual_label", "visual_confidence", "visual_reason", "visual_evidence_regions",
            "semantic_label", "semantic_confidence", "semantic_reason", "semantic_evidence_regions",
            "layout_label", "layout_confidence", "layout_reason", "layout_evidence_regions",
            "correct", "timestamp",
        ])
        for r in results:
            correct = (
                r.ground_truth == r.verdict.label
                if r.ground_truth is not None else "N/A"
            )
            writer.writerow([
                r.image_path,
                r.ground_truth,
                r.verdict.label,
                r.verdict.vote_tally["REAL"],
                r.verdict.vote_tally["FAKE"],
                r.verdict.vote_tally["UNCERTAIN"],
                r.judge_details["visual"].label,
                r.judge_details["visual"].confidence,
                r.judge_details["visual"].reason,
                " | ".join(r.judge_details["visual"].evidence_regions),
                r.judge_details["semantic"].label,
                r.judge_details["semantic"].confidence,
                r.judge_details["semantic"].reason,
                " | ".join(r.judge_details["semantic"].evidence_regions),
                r.judge_details["layout"].label,
                r.judge_details["layout"].confidence,
                r.judge_details["layout"].reason,
                " | ".join(r.judge_details["layout"].evidence_regions),
                correct,
                r.timestamp,
            ])
    logger.info(f"CSV results saved  → {csv_path}")


# ---------------------------------------------------------------------------
# Batch Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_root: Path = DATASET_ROOT,
    limit: Optional[int] = None
) -> list[ReceiptResult]:
    """
    Discover all receipt images in dataset_root, run the three-judge ensemble
    on each sequentially, persist results, and return the full list of
    ReceiptResult objects.

    Args:
        dataset_root : Path to the dataset root (must contain real/ and fake/).
        limit        : Optional cap on the number of images to process (for testing).
    """
    downloader(dataset_root)
    format_dataset(dataset_root)
    sample_dataset(dataset_root, sample_size=10)
    images = discover_dataset(dataset_root)

    if not images:
        logger.error("No images found. Check your dataset path and folder structure.")
        return []

    if limit:
        images = images[:limit]
        logger.info(f"Processing limited to first {limit} images.")

    logger.info(f"\nStarting analysis of {len(images)} receipt(s)…\n{'='*60}")
    logger.info(
        f"Rate limiter: {RATE_LIMIT_RPM} RPM max · "
        f"{RATE_LIMIT_MIN_GAP_SEC}s min gap · "
        f"{RATE_LIMIT_RETRY_MAX} retries · "
        f"base back-off {RATE_LIMIT_RETRY_BASE}s"
    )

    results: list[ReceiptResult] = []
    for image_path in tqdm(images, desc="Analysing receipts"):
        try:
            result = analyse_receipt(image_path)
            results.append(result)
            logger.info(
                f"✓ {image_path.name:<40} "
                f"GT={result.ground_truth or '?':<10} "
                f"PRED={result.verdict.label:<10} "
                f"TALLY=R:{result.verdict.vote_tally['REAL']} "
                f"F:{result.verdict.vote_tally['FAKE']} "
                f"U:{result.verdict.vote_tally['UNCERTAIN']}"
            )
        except Exception as exc:
            logger.error(f"✗ Failed on {image_path}: {exc}")

    save_results(results)

    # ── Quick summary ─────────────────────────────────────────────────────────
    if results:
        total    = len(results)
        correct  = sum(
            1 for r in results
            if r.ground_truth is not None and r.ground_truth == r.verdict.label
        )
        labelled = sum(1 for r in results if r.ground_truth is not None)
        accuracy = (correct / labelled * 100) if labelled else 0.0

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Total processed : {total}")
        print(f"  Correct         : {correct}/{labelled}  ({accuracy:.1f}% accuracy)")
        print(f"  Results saved   : {RESULTS_DIR}/")
        print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Receipt Forgery Detection — Three-Judge Gemini Ensemble"
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_ROOT),
        help="Path to the dataset root folder (default: ./dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (useful for testing)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    if args.api_key:
        genai.configure(api_key=args.api_key)

    run_pipeline(
        dataset_root=Path(args.dataset),
        limit=args.limit,
    )
