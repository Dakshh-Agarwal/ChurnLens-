"""
bert_model.py — Dual-head BERT model for ChurnLens.

Architecture:
    BERT base → shared encoder
                ├── Sentiment Head (3 classes: positive/neutral/negative)
                └── Churn Risk Head (2 classes: risk/no_risk)

Multi-task learning: both heads share the BERT backbone,
which learns general review understanding. Each head specializes
in its own classification task.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from loguru import logger


class ChurnLensBERT(nn.Module):
    """
    Dual-head BERT for simultaneous sentiment classification 
    and churn risk detection.
    
    Architecture:
        [CLS] token embedding (768d)
            → Dropout
            ├── Sentiment classifier: Linear(768 → 256) → ReLU → Dropout → Linear(256 → 3)
            └── Churn classifier: Linear(768 → 128) → ReLU → Dropout → Linear(128 → 2)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_sentiment_classes: int = 3,
        num_churn_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert_layers: int = 0,
    ):
        """
        Args:
            model_name: Pretrained BERT model name
            num_sentiment_classes: Number of sentiment categories
            num_churn_classes: Number of churn risk categories
            dropout: Dropout probability
            freeze_bert_layers: Number of BERT encoder layers to freeze (0 = none)
        """
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # Optionally freeze lower BERT layers for faster training
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_bert_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            logger.info(f"Froze {freeze_bert_layers} BERT encoder layers")
        
        self.dropout = nn.Dropout(dropout)
        
        # ── Sentiment classification head ────────────────────────
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sentiment_classes),
        )
        
        # ── Churn risk classification head ───────────────────────
        self.churn_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_churn_classes),
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Model initialized: {total_params:,} total params | "
            f"{trainable_params:,} trainable"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Forward pass through BERT + both classification heads.
        
        Args:
            input_ids: (batch_size, seq_len) tokenized input
            attention_mask: (batch_size, seq_len) attention mask
            
        Returns:
            dict with 'sentiment_logits' and 'churn_logits'
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        cls_output = self.dropout(cls_output)
        
        # Task-specific heads
        sentiment_logits = self.sentiment_head(cls_output)  # (batch, 3)
        churn_logits = self.churn_head(cls_output)          # (batch, 2)
        
        return {
            "sentiment_logits": sentiment_logits,
            "churn_logits": churn_logits,
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Run inference and return probabilities + predicted classes.
        
        Returns:
            dict with sentiment/churn predictions, probabilities, and labels
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        
        sentiment_probs = torch.softmax(logits["sentiment_logits"], dim=-1)
        churn_probs = torch.softmax(logits["churn_logits"], dim=-1)
        
        sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
        churn_preds = torch.argmax(churn_probs, dim=-1)
        
        # Map predictions to labels
        sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
        churn_labels = {0: False, 1: True}
        
        return {
            "sentiment_pred": sentiment_preds,
            "sentiment_probs": sentiment_probs,
            "sentiment_label": [sentiment_labels[p.item()] for p in sentiment_preds],
            "churn_pred": churn_preds,
            "churn_probs": churn_probs,
            "churn_risk": [churn_labels[p.item()] for p in churn_preds],
        }


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining sentiment and churn losses.
    
    total_loss = α * sentiment_loss + β * churn_loss
    
    Uses class weights to handle imbalanced label distributions.
    """
    
    def __init__(
        self,
        sentiment_weight: float = 0.6,
        churn_weight: float = 0.4,
        sentiment_class_weights: torch.Tensor = None,
        churn_class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.sentiment_weight = sentiment_weight
        self.churn_weight = churn_weight
        
        self.sentiment_criterion = nn.CrossEntropyLoss(
            weight=sentiment_class_weights
        )
        self.churn_criterion = nn.CrossEntropyLoss(
            weight=churn_class_weights
        )
    
    def forward(
        self,
        sentiment_logits: torch.Tensor,
        churn_logits: torch.Tensor,
        sentiment_labels: torch.Tensor,
        churn_labels: torch.Tensor,
    ) -> dict:
        """
        Compute weighted multi-task loss.
        
        Returns:
            dict with total_loss, sentiment_loss, and churn_loss
        """
        sentiment_loss = self.sentiment_criterion(sentiment_logits, sentiment_labels)
        churn_loss = self.churn_criterion(churn_logits, churn_labels)
        
        total_loss = (
            self.sentiment_weight * sentiment_loss +
            self.churn_weight * churn_loss
        )
        
        return {
            "total_loss": total_loss,
            "sentiment_loss": sentiment_loss,
            "churn_loss": churn_loss,
        }
