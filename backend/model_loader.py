"""
model_loader.py — Singleton model loader for FastAPI.

Loads the ChurnLens predictor once on startup and reuses it
across all API requests. Import is lazy to avoid downloading
BERT weights when running in demo mode.
"""

import os
from loguru import logger

_predictor = None


def get_predictor():
    """Get or initialize the singleton predictor."""
    global _predictor
    if _predictor is None:
        # Lazy import — avoids loading torch/transformers at module level
        from model.predict import ChurnLensPredictor
        
        checkpoint_dir = os.getenv("CHECKPOINT_DIR", "checkpoints")
        config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
        logger.info(f"Loading model from {checkpoint_dir}...")
        _predictor = ChurnLensPredictor(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
        )
        logger.success("Model loaded successfully")
    return _predictor


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return _predictor is not None
