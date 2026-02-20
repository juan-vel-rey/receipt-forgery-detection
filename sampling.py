"""
"20 sampling" requirement
========================================================
Dataset : Pick sample from Find It Again Receipt Dataset train folder
          Expected folder structure:
              data/
              └── train/
                  ├── receipt_042.png
                  └── ...

Output  : List of 20 sampled images (10 real + 10 fake)
          + Sampling log       (results/sampling_log.txt)
"""

import logging
import os
import random
import shutil

from pathlib import Path

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

DATASET_ROOT: Path = Path("dataset")          # root folder of the dataset

def run_pipeline(
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

    return selected_receipts


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sampling Find It Again Receipt Dataset"
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_ROOT),
        help="Path to the dataset root folder (default: ./dataset)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of images to sample from each category (default: 10)",
    )
    args = parser.parse_args()

    run_pipeline(
        dataset_root=Path(args.dataset),
        sample_size=args.sample_size,
    )