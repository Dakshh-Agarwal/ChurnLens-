"""
evaluate.py — Evaluate the trained ChurnLens model on the test set.
"""

import os, sys, json, argparse
import yaml, torch, numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import ReviewDataset
from model.bert_model import ChurnLensBERT


def load_model(checkpoint_dir, device):
    model_path = os.path.join(checkpoint_dir, "best_model")
    with open(os.path.join(model_path, "model_info.json")) as f:
        info = json.load(f)
    model = ChurnLensBERT(
        model_name=info["model_name"],
        num_sentiment_classes=info["num_sentiment_classes"],
        num_churn_classes=info["num_churn_classes"],
        dropout=info["dropout"],
    )
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=device))
    model.to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer, info


def evaluate_model(config):
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, _ = load_model(train_cfg["checkpoint_dir"], device)
    test_ds = ReviewDataset(
        os.path.join(data_cfg["processed_dir"], "test.parquet"),
        max_length=model_cfg["max_length"], tokenizer=tokenizer,
    )
    loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=2)

    s_preds, s_labels, c_preds, c_labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            s_preds.extend(torch.argmax(out["sentiment_logits"], -1).cpu().numpy())
            s_labels.extend(batch["sentiment_label"].numpy())
            c_preds.extend(torch.argmax(out["churn_logits"], -1).cpu().numpy())
            c_labels.extend(batch["churn_label"].numpy())

    s_names = ["negative", "neutral", "positive"]
    c_names = ["no_risk", "churn_risk"]

    results = {
        "test_size": len(s_labels),
        "sentiment": {
            "accuracy": float(accuracy_score(s_labels, s_preds)),
            "f1_weighted": float(f1_score(s_labels, s_preds, average="weighted")),
            "report": classification_report(s_labels, s_preds, target_names=s_names, output_dict=True),
            "confusion_matrix": confusion_matrix(s_labels, s_preds).tolist(),
        },
        "churn": {
            "accuracy": float(accuracy_score(c_labels, c_preds)),
            "f1": float(f1_score(c_labels, c_preds, average="binary", pos_label=1)),
            "report": classification_report(c_labels, c_preds, target_names=c_names, output_dict=True),
            "confusion_matrix": confusion_matrix(c_labels, c_preds).tolist(),
        },
    }

    logger.info(f"Sentiment Accuracy: {results['sentiment']['accuracy']:.4f}")
    logger.info(f"Churn F1: {results['churn']['f1']:.4f}")
    logger.info(classification_report(s_labels, s_preds, target_names=s_names))
    logger.info(classification_report(c_labels, c_preds, target_names=c_names))

    path = os.path.join(train_cfg["checkpoint_dir"], "evaluation_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved to {path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluate_model(config)

if __name__ == "__main__":
    main()
