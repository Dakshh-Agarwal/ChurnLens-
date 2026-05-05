"""
schemas.py — Pydantic models for the ChurnLens API.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000, description="Review text to analyze")

class BatchReviewRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100, description="List of review texts")

class SentimentScores(BaseModel):
    negative: float
    neutral: float
    positive: float

class PredictionResponse(BaseModel):
    sentiment: str
    sentiment_confidence: float
    sentiment_scores: SentimentScores
    churn_risk: bool
    churn_confidence: float
    themes: list[str]

class BatchPredictionItem(BaseModel):
    text: str
    sentiment: str
    sentiment_confidence: float
    churn_risk: bool
    churn_confidence: float
    themes: list[str]

class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    count: int

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    num_sentiment_classes: int
    num_churn_classes: int
    max_length: int
    device: str
    sentiment_labels: list[str]
    churn_labels: list[str]
