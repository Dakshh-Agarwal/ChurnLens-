"""
dataset.py — PyTorch Dataset for ChurnLens review data.

Handles tokenization and batching of preprocessed review text
for the dual-head BERT model (sentiment + churn).
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from loguru import logger


class ReviewDataset(Dataset):
    """
    PyTorch Dataset for gym review sentiment + churn classification.
    
    Each sample returns:
        - input_ids: tokenized review text
        - attention_mask: attention mask for padding
        - sentiment_label: 0=negative, 1=neutral, 2=positive
        - churn_label: 0=no_risk, 1=churn_risk
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256,
        tokenizer: BertTokenizer = None,
    ):
        """
        Args:
            data_path: Path to parquet file with preprocessed data
            tokenizer_name: HuggingFace tokenizer name
            max_length: Max token sequence length
            tokenizer: Pre-initialized tokenizer (optional, avoids reloading)
        """
        self.data = pd.read_parquet(data_path)
        self.max_length = max_length
        
        # Use provided tokenizer or load new one
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        logger.info(
            f"Loaded {len(self.data):,} samples from {data_path} | "
            f"Max length: {max_length}"
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        
        text = str(row["text"])
        sentiment_label = int(row["sentiment_id"])
        # Support both column names: churn_risk_id (int) or churn_risk (bool)
        if "churn_risk_id" in self.data.columns:
            churn_label = int(row["churn_risk_id"])
        else:
            churn_label = int(row["churn_risk"])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "churn_label": torch.tensor(churn_label, dtype=torch.long),
        }


def get_class_weights(data_path: str, task: str = "sentiment") -> torch.Tensor:
    """
    Compute inverse-frequency class weights for handling imbalanced data.
    
    Args:
        data_path: Path to parquet file
        task: 'sentiment' or 'churn'
    
    Returns:
        Tensor of class weights
    """
    df = pd.read_parquet(data_path)
    
    if task == "sentiment":
        counts = df["sentiment_id"].value_counts().sort_index()
    elif task == "churn":
        # Support both column names
        col = "churn_risk_id" if "churn_risk_id" in df.columns else "churn_risk"
        counts = df[col].astype(int).value_counts().sort_index()
    else:
        raise ValueError(f"Unknown task: {task}")
    
    total = counts.sum()
    weights = total / (len(counts) * counts.values)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    logger.info(f"Class weights for {task}: {weights.tolist()}")
    return weights
