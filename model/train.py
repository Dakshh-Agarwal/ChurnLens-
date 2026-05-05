"""
train.py — Fine-tune BERT for ChurnLens dual-task classification.

Training pipeline:
  1. Load preprocessed parquet data
  2. Initialize dual-head BERT model
  3. Train with AdamW optimizer + linear warmup scheduler
  4. Evaluate on validation set each epoch
  5. Save best checkpoint based on combined metric
  6. Log metrics to console

Usage:
    python model/train.py
    python model/train.py --config configs/config.yaml
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import ReviewDataset, get_class_weights
from model.bert_model import ChurnLensBERT, MultiTaskLoss


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config_device: str) -> torch.device:
    """Determine compute device."""
    if config_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config_device)
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def evaluate(
    model: ChurnLensBERT,
    dataloader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        dict with loss, accuracy, f1 for both tasks
    """
    model.eval()
    
    total_loss = 0
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_churn_preds = []
    all_churn_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            churn_labels = batch["churn_label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            losses = loss_fn(
                outputs["sentiment_logits"],
                outputs["churn_logits"],
                sentiment_labels,
                churn_labels,
            )
            total_loss += losses["total_loss"].item()
            
            # Predictions
            sentiment_preds = torch.argmax(outputs["sentiment_logits"], dim=-1)
            churn_preds = torch.argmax(outputs["churn_logits"], dim=-1)
            
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_churn_preds.extend(churn_preds.cpu().numpy())
            all_churn_labels.extend(churn_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    sentiment_acc = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average="weighted")
    
    churn_acc = accuracy_score(all_churn_labels, all_churn_preds)
    churn_f1 = f1_score(all_churn_labels, all_churn_preds, average="binary", pos_label=1)
    
    return {
        "loss": avg_loss,
        "sentiment_accuracy": sentiment_acc,
        "sentiment_f1": sentiment_f1,
        "churn_accuracy": churn_acc,
        "churn_f1": churn_f1,
        "sentiment_report": classification_report(
            all_sentiment_labels, all_sentiment_preds,
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
        ),
        "churn_report": classification_report(
            all_churn_labels, all_churn_preds,
            target_names=["no_risk", "churn_risk"],
            output_dict=True,
        ),
    }


def train(config: dict) -> None:
    """Main training loop."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    
    # Setup
    set_seed(train_cfg["seed"])
    device = get_device(train_cfg["device"])
    checkpoint_dir = train_cfg["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ── Load tokenizer ───────────────────────────────────────────
    logger.info(f"Loading tokenizer: {model_cfg['name']}")
    tokenizer = BertTokenizer.from_pretrained(model_cfg["name"])
    
    # ── Load datasets ────────────────────────────────────────────
    processed_dir = data_cfg["processed_dir"]
    
    train_dataset = ReviewDataset(
        data_path=os.path.join(processed_dir, "train.parquet"),
        max_length=model_cfg["max_length"],
        tokenizer=tokenizer,
    )
    val_dataset = ReviewDataset(
        data_path=os.path.join(processed_dir, "val.parquet"),
        max_length=model_cfg["max_length"],
        tokenizer=tokenizer,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # ── Initialize model ─────────────────────────────────────────
    model = ChurnLensBERT(
        model_name=model_cfg["name"],
        num_sentiment_classes=model_cfg["num_sentiment_classes"],
        num_churn_classes=model_cfg["num_churn_classes"],
        dropout=model_cfg["dropout"],
    ).to(device)
    
    # ── Class weights for imbalanced data ────────────────────────
    sentiment_weights = get_class_weights(
        os.path.join(processed_dir, "train.parquet"), task="sentiment"
    ).to(device)
    churn_weights = get_class_weights(
        os.path.join(processed_dir, "train.parquet"), task="churn"
    ).to(device)
    
    # ── Loss function ────────────────────────────────────────────
    loss_fn = MultiTaskLoss(
        sentiment_weight=model_cfg["sentiment_loss_weight"],
        churn_weight=model_cfg["churn_loss_weight"],
        sentiment_class_weights=sentiment_weights,
        churn_class_weights=churn_weights,
    )
    
    # ── Optimizer ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    
    # ── Learning rate scheduler ──────────────────────────────────
    total_steps = len(train_loader) * train_cfg["num_epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    logger.info(f"Total steps: {total_steps:,} | Warmup steps: {warmup_steps:,}")
    
    # ── Mixed precision ──────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if train_cfg["fp16"] and device.type == "cuda" else None
    
    # ── Training loop ────────────────────────────────────────────
    best_metric = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0
        epoch_sentiment_loss = 0
        epoch_churn_loss = 0
        step_count = 0
        epoch_start = time.time()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            churn_labels = batch["churn_label"].to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, attention_mask)
                    losses = loss_fn(
                        outputs["sentiment_logits"],
                        outputs["churn_logits"],
                        sentiment_labels,
                        churn_labels,
                    )
                
                scaler.scale(losses["total_loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask)
                losses = loss_fn(
                    outputs["sentiment_logits"],
                    outputs["churn_logits"],
                    sentiment_labels,
                    churn_labels,
                )
                
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                optimizer.step()
            
            scheduler.step()
            
            epoch_loss += losses["total_loss"].item()
            epoch_sentiment_loss += losses["sentiment_loss"].item()
            epoch_churn_loss += losses["churn_loss"].item()
            step_count += 1
            
            # Periodic logging
            if (step + 1) % train_cfg["log_every_n_steps"] == 0:
                avg_loss = epoch_loss / step_count
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{train_cfg['num_epochs']} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )
        
        # ── Epoch metrics ────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / step_count
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{train_cfg['num_epochs']} completed in {epoch_time:.1f}s")
        logger.info(f"Avg Loss: {avg_epoch_loss:.4f}")
        
        # ── Validation ───────────────────────────────────────────
        logger.info("Running validation...")
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Sentiment Acc: {val_metrics['sentiment_accuracy']:.4f} | "
            f"Sentiment F1: {val_metrics['sentiment_f1']:.4f} | "
            f"Churn Acc: {val_metrics['churn_accuracy']:.4f} | "
            f"Churn F1: {val_metrics['churn_f1']:.4f}"
        )
        
        # Combined metric for model selection
        combined_metric = (
            0.6 * val_metrics["sentiment_f1"] +
            0.4 * val_metrics["churn_f1"]
        )
        
        # Save training history
        epoch_history = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "val_loss": val_metrics["loss"],
            "sentiment_accuracy": val_metrics["sentiment_accuracy"],
            "sentiment_f1": val_metrics["sentiment_f1"],
            "churn_accuracy": val_metrics["churn_accuracy"],
            "churn_f1": val_metrics["churn_f1"],
            "combined_metric": combined_metric,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_time": epoch_time,
        }
        training_history.append(epoch_history)
        
        # ── Checkpointing ────────────────────────────────────────
        if combined_metric > best_metric + train_cfg["min_delta"]:
            best_metric = combined_metric
            patience_counter = 0
            
            # Save best model
            best_path = os.path.join(checkpoint_dir, "best_model")
            os.makedirs(best_path, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(best_path, "model.pt"))
            tokenizer.save_pretrained(best_path)
            
            # Save model config
            model_info = {
                "model_name": model_cfg["name"],
                "num_sentiment_classes": model_cfg["num_sentiment_classes"],
                "num_churn_classes": model_cfg["num_churn_classes"],
                "dropout": model_cfg["dropout"],
                "max_length": model_cfg["max_length"],
                "best_epoch": epoch + 1,
                "best_combined_metric": float(best_metric),
                "sentiment_accuracy": float(val_metrics["sentiment_accuracy"]),
                "sentiment_f1": float(val_metrics["sentiment_f1"]),
                "churn_accuracy": float(val_metrics["churn_accuracy"]),
                "churn_f1": float(val_metrics["churn_f1"]),
            }
            with open(os.path.join(best_path, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.success(
                f"New best model saved! Combined metric: {best_metric:.4f}"
            )
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{train_cfg['patience']}"
            )
        
        # Early stopping
        if patience_counter >= train_cfg["patience"]:
            logger.warning(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # ── Save training history ────────────────────────────────────
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    
    logger.success(f"Training complete! Best combined metric: {best_metric:.4f}")
    logger.info(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model')}")
    logger.info(f"Training history saved to: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ChurnLens BERT model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    train(config)


if __name__ == "__main__":
    main()
