"""
preprocess.py — Filter, label, and split the Yelp dataset for ChurnLens.

Pipeline:
  1. Load raw Yelp business data
  2. Filter to fitness-related businesses only
  3. Stream large review JSON (6M+ reviews) line-by-line
  4. Filter, clean, and label reviews on-the-fly (Memory Efficient)
  5. Train/val/test split with stratification
  6. Save processed datasets as parquet files
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(filepath: str, max_lines: int = None) -> pd.DataFrame:
    """Load a JSON-lines file into a DataFrame."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def filter_fitness_businesses(business_df: pd.DataFrame, categories: list) -> set:
    """Filter businesses to fitness-related categories."""
    fitness_ids = set()
    category_pattern = "|".join([re.escape(c) for c in categories])
    
    for _, row in business_df.iterrows():
        cats = row.get("categories", "")
        if cats and isinstance(cats, str):
            if re.search(category_pattern, cats, re.IGNORECASE):
                fitness_ids.add(row["business_id"])
    
    logger.info(f"Found {len(fitness_ids):,} fitness businesses out of {len(business_df):,} total")
    return fitness_ids


def assign_sentiment_label(stars: int, mapping: dict) -> str:
    """Convert star rating to sentiment label."""
    for label, star_values in mapping.items():
        if stars in star_values:
            return label
    return "neutral"


def detect_churn_risk(text: str, churn_keywords: dict) -> bool:
    """Detect churn risk from review text using keyword signals."""
    text_lower = text.lower()
    for category, keywords in churn_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return True
    return False


def detect_themes(text: str, theme_keywords: dict) -> list:
    """Detect complaint/praise themes from review text."""
    text_lower = text.lower()
    detected = []
    for theme, keywords in theme_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(theme)
                break
    return detected


def clean_text(text: str) -> str:
    """Basic text cleaning for review content."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
    return text.strip()


def preprocess_pipeline(config: dict) -> None:
    """Run the full preprocessing pipeline."""
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    
    # ── Step 1: Load business data ──────────────────────────────
    logger.info("Loading business data...")
    business_df = load_jsonl(os.path.join(raw_dir, "yelp_academic_dataset_business.json"))
    
    # ── Step 2: Filter to fitness businesses ─────────────────────
    logger.info("Filtering to fitness businesses...")
    fitness_ids = filter_fitness_businesses(business_df, data_cfg["fitness_categories"])
    
    # ── Step 3: Stream and Filter Reviews (Memory Efficient) ─────
    logger.info("Streaming and filtering reviews (this may take 5-10 mins)...")
    review_path = os.path.join(raw_dir, "yelp_academic_dataset_review.json")
    max_samples = data_cfg.get("max_samples")
    
    fitness_reviews = []
    processed_count = 0
    theme_counts = Counter()
    
    sentiment_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            processed_count += 1
            if max_samples and processed_count > max_samples:
                break
                
            if any(fid in line for fid in fitness_ids):
                record = json.loads(line)
                if record["business_id"] in fitness_ids:
                    text = clean_text(record["text"])
                    if len(text) > 10:
                        stars = int(record["stars"])
                        sentiment = assign_sentiment_label(stars, data_cfg["sentiment_mapping"])
                        themes = detect_themes(text, data_cfg["theme_keywords"])
                        theme_counts.update(themes)
                        
                        fitness_reviews.append({
                            "review_id": record["review_id"],
                            "business_id": record["business_id"],
                            "text": text,
                            "stars": stars,
                            "sentiment": sentiment,
                            "sentiment_id": sentiment_to_id[sentiment],
                            "churn_risk": detect_churn_risk(text, data_cfg["churn_keywords"]),
                            "themes_str": ",".join(themes) if themes else "none"
                        })
            
            if processed_count % 100000 == 0:
                logger.info(f"Processed {processed_count:,} reviews... Found {len(fitness_reviews):,} matches")

    review_df = pd.DataFrame(fitness_reviews)
    if len(review_df) == 0:
        logger.error("No fitness reviews found!")
        return

    logger.info(f"Final filtered dataset size: {len(review_df):,} reviews")
    logger.info(f"Sentiment distribution:\n{review_df['sentiment'].value_counts().to_string()}")
    logger.info(f"Churn distribution:\n{review_df['churn_risk'].value_counts().to_string()}")
    
    # ── Step 4: Train/val/test split ─────────────────────────────
    logger.info("Splitting dataset...")
    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]
    test_ratio = data_cfg["test_ratio"]
    
    train_df, temp_df = train_test_split(
        review_df, 
        test_size=(val_ratio + test_ratio),
        stratify=review_df["sentiment_id"],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df["sentiment_id"],
        random_state=42
    )
    
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    # ── Step 5: Save ─────────────────────────────────────────────
    train_df.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(processed_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(processed_dir, "test.parquet"), index=False)
    
    stats = {
        "total_reviews": len(review_df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "sentiment_distribution": review_df["sentiment"].value_counts().to_dict(),
        "churn_rate": float(review_df["churn_risk"].mean()),
        "theme_counts": dict(theme_counts.most_common()),
    }
    with open(os.path.join(processed_dir, "label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.success(f"Preprocessing complete! Files saved to {processed_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Yelp data for ChurnLens")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    preprocess_pipeline(config)


if __name__ == "__main__":
    main()
