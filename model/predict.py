"""
predict.py — Inference utilities for ChurnLens.

Provides a clean interface for single and batch predictions
with theme detection. Used by the FastAPI backend.
"""

import os, sys, json, re
import yaml, torch
from transformers import BertTokenizer
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.bert_model import ChurnLensBERT


class ChurnLensPredictor:
    """High-level predictor wrapping the BERT model + theme detection."""

    SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
    CHURN_LABELS = {0: False, 1: True}

    def __init__(self, checkpoint_dir: str = "checkpoints", config_path: str = "configs/config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config for theme keywords
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.theme_keywords = self.config["data"]["theme_keywords"]

        # Load model
        model_path = os.path.join(checkpoint_dir, "best_model")
        with open(os.path.join(model_path, "model_info.json")) as f:
            info = json.load(f)

        self.max_length = info["max_length"]
        self.model = ChurnLensBERT(
            model_name=info["model_name"],
            num_sentiment_classes=info["num_sentiment_classes"],
            num_churn_classes=info["num_churn_classes"],
            dropout=info["dropout"],
        )
        self.model.load_state_dict(
            torch.load(os.path.join(model_path, "model.pt"), map_location=self.device)
        )
        self.model.to(self.device).eval()

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        logger.info(f"Predictor ready on {self.device}")

    def _detect_themes(self, text: str) -> list:
        text_lower = text.lower()
        themes = []
        for theme, keywords in self.theme_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    themes.append(theme)
                    break
        return themes

    def predict(self, text: str) -> dict:
        """Predict sentiment, churn risk, and themes for a single review."""
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        s_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)[0]
        c_probs = torch.softmax(outputs["churn_logits"], dim=-1)[0]

        s_pred = torch.argmax(s_probs).item()
        c_pred = torch.argmax(c_probs).item()

        return {
            "sentiment": self.SENTIMENT_LABELS[s_pred],
            "sentiment_confidence": round(float(s_probs[s_pred]), 4),
            "sentiment_scores": {
                self.SENTIMENT_LABELS[i]: round(float(s_probs[i]), 4) for i in range(3)
            },
            "churn_risk": self.CHURN_LABELS[c_pred],
            "churn_confidence": round(float(c_probs[c_pred]), 4),
            "themes": self._detect_themes(text),
        }

    def predict_batch(self, texts: list) -> list:
        """Predict for a batch of reviews."""
        encodings = self.tokenizer(
            texts, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        s_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
        c_probs = torch.softmax(outputs["churn_logits"], dim=-1)

        results = []
        for i in range(len(texts)):
            sp = s_probs[i]
            cp = c_probs[i]
            s_pred = torch.argmax(sp).item()
            c_pred = torch.argmax(cp).item()
            results.append({
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "sentiment": self.SENTIMENT_LABELS[s_pred],
                "sentiment_confidence": round(float(sp[s_pred]), 4),
                "churn_risk": self.CHURN_LABELS[c_pred],
                "churn_confidence": round(float(cp[c_pred]), 4),
                "themes": self._detect_themes(texts[i]),
            })
        return results
