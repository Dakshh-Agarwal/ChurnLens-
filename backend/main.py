"""
main.py — FastAPI application for ChurnLens.

Endpoints:
    POST /predict        — Single review prediction
    POST /predict/batch  — Batch predictions  
    GET  /health         — Health check
    GET  /model/info     — Model metadata
"""

import os, sys, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.schemas import (
    ReviewRequest, BatchReviewRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse, SentimentScores,
    BatchPredictionItem,
)
from backend.model_loader import get_predictor, is_model_loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup if checkpoint exists."""
    logger.info("Starting ChurnLens API...")
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "checkpoints")
    model_info_path = os.path.join(checkpoint_dir, "best_model", "model_info.json")
    
    if os.path.exists(model_info_path):
        try:
            get_predictor()
        except Exception as e:
            logger.warning(f"Model not loaded on startup: {e}")
            logger.info("API will run in demo mode")
    else:
        logger.info("No trained model found — running in demo mode (rule-based predictions)")
        logger.info(f"Train a model first, then place checkpoint at: {checkpoint_dir}/best_model/")
    yield
    logger.info("Shutting down ChurnLens API")


app = FastAPI(
    title="ChurnLens API",
    description="NLP-powered sentiment analysis and churn prediction for gym reviews",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Demo mode fallback ──────────────────────────────────────────
def demo_predict(text: str) -> dict:
    """Rule-based fallback when model is not loaded."""
    text_lower = text.lower()

    # Simple rule-based sentiment
    pos_words = ["great", "love", "amazing", "excellent", "best", "awesome", "fantastic", "clean", "friendly"]
    neg_words = ["bad", "terrible", "worst", "dirty", "rude", "cancel", "hate", "awful", "horrible", "overpriced"]

    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)

    if pos_count > neg_count:
        sentiment, s_conf = "positive", 0.75
    elif neg_count > pos_count:
        sentiment, s_conf = "negative", 0.75
    else:
        sentiment, s_conf = "neutral", 0.50

    # Simple churn keywords
    churn_kw = ["cancel", "leaving", "quit", "switching", "not worth", "waste", "never coming back"]
    churn_risk = any(kw in text_lower for kw in churn_kw)
    c_conf = 0.70 if churn_risk else 0.65

    # Themes
    theme_map = {
        "cleanliness": ["dirty", "clean", "filthy", "smell"],
        "equipment": ["equipment", "machine", "broken", "treadmill", "weights"],
        "staff": ["staff", "trainer", "rude", "friendly", "helpful"],
        "pricing": ["price", "expensive", "cheap", "overpriced", "fee"],
        "overcrowding": ["crowded", "packed", "busy", "wait"],
    }
    themes = [t for t, kws in theme_map.items() if any(k in text_lower for k in kws)]

    scores = {"negative": 0.1, "neutral": 0.1, "positive": 0.1}
    scores[sentiment] = s_conf
    remaining = 1.0 - s_conf
    others = [k for k in scores if k != sentiment]
    for k in others:
        scores[k] = round(remaining / len(others), 4)

    return {
        "sentiment": sentiment, "sentiment_confidence": s_conf,
        "sentiment_scores": scores,
        "churn_risk": churn_risk, "churn_confidence": c_conf,
        "themes": themes,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    """Predict sentiment and churn risk for a single review."""
    start = time.time()
    try:
        if is_model_loaded():
            result = get_predictor().predict(request.text)
        else:
            result = demo_predict(request.text)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.time() - start) * 1000
    logger.info(f"Prediction: {result['sentiment']} | churn={result['churn_risk']} | {latency:.0f}ms")

    return PredictionResponse(
        sentiment=result["sentiment"],
        sentiment_confidence=result["sentiment_confidence"],
        sentiment_scores=SentimentScores(**result["sentiment_scores"]),
        churn_risk=result["churn_risk"],
        churn_confidence=result["churn_confidence"],
        themes=result["themes"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchReviewRequest):
    """Batch predict sentiment and churn risk (processes in chunks to avoid OOM)."""
    try:
        results = []
        chunk_size = 16
        if is_model_loaded():
            predictor = get_predictor()
            for i in range(0, len(request.texts), chunk_size):
                chunk = request.texts[i:i + chunk_size]
                results.extend(predictor.predict_batch(chunk))
        else:
            results = [demo_predict(t) for t in request.texts]
            for r, t in zip(results, request.texts):
                r["text"] = t[:100] + "..." if len(t) > 100 else t
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    items = [BatchPredictionItem(**r) for r in results]
    return BatchPredictionResponse(predictions=items, count=len(items))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=is_model_loaded())


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if is_model_loaded():
        p = get_predictor()
        return ModelInfoResponse(
            model_name="ChurnLens-BERT",
            num_sentiment_classes=3,
            num_churn_classes=2,
            max_length=p.max_length,
            device=str(p.device),
            sentiment_labels=["negative", "neutral", "positive"],
            churn_labels=["no_risk", "churn_risk"],
        )
    return ModelInfoResponse(
        model_name="ChurnLens-BERT (demo mode)",
        num_sentiment_classes=3, num_churn_classes=2,
        max_length=256, device="cpu",
        sentiment_labels=["negative", "neutral", "positive"],
        churn_labels=["no_risk", "churn_risk"],
    )
