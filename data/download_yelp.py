"""
download_yelp.py — Download and extract the Yelp Open Dataset.

The Yelp dataset is ~9GB compressed. This script handles download,
extraction, and initial validation of the required JSON files.

NOTE: Yelp requires accepting their terms at https://www.yelp.com/dataset
      Download manually if automated download fails.
"""

import os
import sys
import json
import tarfile
import argparse
from pathlib import Path
from loguru import logger


# Required files from the Yelp dataset
REQUIRED_FILES = [
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_review.json",
]


def validate_dataset(raw_dir: str) -> bool:
    """Check if required Yelp dataset files exist."""
    raw_path = Path(raw_dir)
    missing = []
    for f in REQUIRED_FILES:
        if not (raw_path / f).exists():
            missing.append(f)
    
    if missing:
        logger.warning(f"Missing files: {missing}")
        return False
    
    logger.success("All required dataset files found.")
    return True


def extract_tar(tar_path: str, extract_to: str) -> None:
    """Extract a .tar file to the specified directory."""
    logger.info(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_to)
    logger.success("Extraction complete.")


def count_records(filepath: str, max_count: int = 1000) -> int:
    """Count records in a JSON-lines file (sample first N)."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if max_count and count >= max_count:
                break
    return count


def inspect_dataset(raw_dir: str) -> None:
    """Print basic stats about the downloaded dataset."""
    raw_path = Path(raw_dir)
    
    for filename in REQUIRED_FILES:
        filepath = raw_path / filename
        if filepath.exists():
            size_gb = filepath.stat().st_size / (1024**3)
            logger.info(f"{filename}: {size_gb:.2f} GB")
            
            # Show first record structure
            with open(filepath, "r", encoding="utf-8") as f:
                first_record = json.loads(f.readline())
                logger.info(f"  Fields: {list(first_record.keys())}")
                
                # Count total lines (for small files)
                if size_gb < 1:
                    f.seek(0)
                    total = sum(1 for _ in f)
                    logger.info(f"  Total records: {total:,}")
                else:
                    logger.info(f"  (Large file — skipping full count)")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Yelp dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Directory to store raw dataset files")
    parser.add_argument("--tar-path", type=str, default=None,
                        help="Path to downloaded Yelp dataset tar file")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect existing dataset files")
    args = parser.parse_args()
    
    raw_dir = args.raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    
    if args.inspect:
        inspect_dataset(raw_dir)
        return
    
    # Check if already extracted
    if validate_dataset(raw_dir):
        logger.info("Dataset already present. Use --inspect to view stats.")
        inspect_dataset(raw_dir)
        return
    
    # Extract from tar if provided
    if args.tar_path:
        if not os.path.exists(args.tar_path):
            logger.error(f"Tar file not found: {args.tar_path}")
            sys.exit(1)
        extract_tar(args.tar_path, raw_dir)
        
        if validate_dataset(raw_dir):
            inspect_dataset(raw_dir)
        else:
            logger.error("Extraction completed but required files not found.")
            sys.exit(1)
    else:
        logger.info(
            "\n"
            "=" * 60 + "\n"
            "  YELP DATASET DOWNLOAD INSTRUCTIONS\n"
            "=" * 60 + "\n"
            "\n"
            "  1. Go to: https://www.yelp.com/dataset\n"
            "  2. Accept the terms and download the dataset\n"
            "  3. Place the downloaded tar file in this directory\n"
            "  4. Run: python data/download_yelp.py --tar-path <path-to-tar>\n"
            "\n"
            "  Required files:\n"
            f"    - {REQUIRED_FILES[0]}\n"
            f"    - {REQUIRED_FILES[1]}\n"
            "\n"
            "  Alternatively, place the extracted JSON files directly in:\n"
            f"    {os.path.abspath(raw_dir)}/\n"
            "\n"
            "=" * 60
        )


if __name__ == "__main__":
    main()
