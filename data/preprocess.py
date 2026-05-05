"""
preprocess.py — Filter, label, and split the Yelp dataset for ChurnLens.

Pipeline:
  1. Load raw Yelp business + review JSONs
  2. Filter to fitness-related businesses only
  3. Auto-label sentiment from star ratings (3-class)
  4. Auto-label churn risk from keyword signals (binary)
  5. Detect complaint themes
  6. Train/val/test split with stratification
  7. Save processed datasets as parquet files

Output:
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet
  data/processed/label_stats.json
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
    with open(config_path, "r") as f:
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
    """
    Filter businesses to fitness-related categories.
    Returns set of business_ids.
    """
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
    return "neutral"  # fallback


def detect_churn_risk(text: str, churn_keywords: dict) -> bool:
    """
    Detect churn risk from review text using keyword signals.
    Returns True if churn risk is detected.
    """
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
                break  # one match per theme is enough
    
    return detected


def clean_text(text: str) -> str:
    """Basic text cleaning for review content."""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Keep alphanumeric, basic punctuation, and spaces
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
    
    return text.strip()


def preprocess_pipeline(config: dict) -> None:
    """Run the full preprocessing pipeline."""
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    
    # ── Step 1: Load raw data ────────────────────────────────────
    logger.info("Loading business data...")
    business_df = load_jsonl(os.path.join(raw_dir, "yelp_academic_dataset_business.json"))
    logger.info(f"Loaded {len(business_df):,} businesses")
    
    logger.info("Loading review data...")
    max_samples = data_cfg.get("max_samples")
    review_df = load_jsonl(
        os.path.join(raw_dir, "yelp_academic_dataset_review.json"),
        max_lines=max_samples
    )
    logger.info(f"Loaded {len(review_df):,} reviews")
    
    # ── Step 2: Filter to fitness businesses ─────────────────────
    logger.info("Filtering to fitness businesses...")
    fitness_ids = filter_fitness_businesses(business_df, data_cfg["fitness_categories"])
    
    review_df = review_df[review_df["business_id"].isin(fitness_ids)].copy()
    logger.info(f"Filtered to {len(review_df):,} fitness reviews")
    
    if len(review_df) == 0:
        logger.error("No fitness reviews found! Check category filters.")
        return
    
    # ── Step 3: Clean text ───────────────────────────────────────
    logger.info("Cleaning review text...")
    review_df["text_clean"] = review_df["text"].apply(clean_text)
    
    # Remove empty reviews
    review_df = review_df[review_df["text_clean"].str.len() > 10].copy()
    logger.info(f"After cleaning: {len(review_df):,} reviews")
    
    # ── Step 4: Label sentiment ──────────────────────────────────
    logger.info("Labeling sentiment from star ratings...")
    sentiment_map = data_cfg["sentiment_mapping"]
    review_df["sentiment"] = review_df["stars"].apply(
        lambda s: assign_sentiment_label(int(s), sentiment_map)
    )
    
    # Encode sentiment as integer
    sentiment_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    review_df["sentiment_id"] = review_df["sentiment"].map(sentiment_to_id)
    
    logger.info(f"Sentiment distribution:\n{review_df['sentiment'].value_counts().to_string()}")
    
    # ── Step 5: Label churn risk ─────────────────────────────────
    logger.info("Detecting churn risk from keywords...")
    churn_kw = data_cfg["churn_keywords"]
    review_df["churn_risk"] = review_df["text_clean"].apply(
        lambda t: detect_churn_risk(t, churn_kw)
    )
    review_df["churn_risk_id"] = review_df["churn_risk"].astype(int)
    
    churn_rate = review_df["churn_risk"].mean()
    logger.info(f"Churn risk rate: {churn_rate:.2%}")
    logger.info(f"Churn distribution:\n{review_df['churn_risk'].value_counts().to_string()}")
    
    # ── Step 6: Detect themes ────────────────────────────────────
    logger.info("Detecting review themes...")
    theme_kw = data_cfg["theme_keywords"]
    review_df["themes"] = review_df["text_clean"].apply(
        lambda t: detect_themes(t, theme_kw)
    )
    review_df["themes_str"] = review_df["themes"].apply(lambda x: ",".join(x) if x else "none")
    
    # Theme counts
    all_themes = []
    for themes in review_df["themes"]:
        all_themes.extend(themes)
    theme_counts = Counter(all_themes)
    logger.info(f"Theme distribution:\n{json.dumps(dict(theme_counts.most_common()), indent=2)}")
    
    # ── Step 7: Select columns ───────────────────────────────────
    columns_to_keep = [
        "review_id", "business_id", "user_id",
        "text_clean", "stars",
        "sentiment", "sentiment_id",
        "churn_risk", "churn_risk_id",
        "themes_str"
    ]
    # Keep only columns that exist
    columns_to_keep = [c for c in columns_to_keep if c in review_df.columns]
    review_df = review_df[columns_to_keep].copy()
    review_df = review_df.rename(columns={"text_clean": "text"})
    
    # ── Step 8: Train/val/test split ─────────────────────────────
    logger.info("Splitting dataset...")
    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]
    test_ratio = data_cfg["test_ratio"]
    
    # Stratify by sentiment to maintain class balance
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
    
    # ── Step 9: Save ─────────────────────────────────────────────
    train_df.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(processed_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(processed_dir, "test.parquet"), index=False)
    
    # Save label statistics
    stats = {
        "total_reviews": len(review_df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "sentiment_distribution": review_df["sentiment"].value_counts().to_dict(),
        "churn_rate": float(churn_rate),
        "churn_distribution": review_df["churn_risk"].value_counts().to_dict(),
        "theme_counts": dict(theme_counts.most_common()),
        "fitness_businesses": len(fitness_ids),
    }
    
    with open(os.path.join(processed_dir, "label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.success(f"Preprocessing complete! Files saved to {processed_dir}/")
    logger.info(f"Label stats saved to {processed_dir}/label_stats.json")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Yelp data for ChurnLens")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    preprocess_pipeline(config)


if __name__ == "__main__":
    main()
