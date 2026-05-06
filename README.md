<div align="center">

<h1>🏋️ ChurnLens</h1>

<p><strong>NLP-powered churn prediction & sentiment analysis for the fitness industry</strong></p>

<p>Fine-tuned BERT on 391K real Yelp fitness reviews to simultaneously predict member sentiment and churn risk — served through a production FastAPI backend and an interactive Streamlit operator dashboard.</p>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 🎯 The Problem

Gym member churn (cancellations) runs at **30–50% annually** in the fitness industry — and gyms typically find out too late, only after a member stops showing up.

Members leave signals in their reviews *before* they cancel:
> *"The equipment is always broken and I'm seriously thinking about cancelling my membership."*

**ChurnLens** reads those signals automatically and surfaces them to gym operators before churn happens.

### Who Uses It

| User | How They Use ChurnLens |
|------|------------------------|
| 🏋️ **Gym Manager** | Paste a review → instantly see sentiment + churn risk + complaint themes |
| 📊 **Operations Team** | Upload a CSV of reviews → bulk analysis with exportable risk report |
| 🏢 **Gym Chain / HQ** | Aggregate dashboard → track churn rate trends across locations |

### What It Does

1. **Reads a member review** (typed in or from a CSV export)
2. **Classifies sentiment** — Positive / Neutral / Negative (93.25% F1)
3. **Flags churn risk** — Is this member likely to cancel? (86.41% F1)
4. **Identifies themes** — Equipment, Staff, Cleanliness, Pricing, Overcrowding
5. **Displays results** on a Streamlit dashboard the gym operator can act on

---

## 📊 Model Performance

> Trained on **313,085 real Yelp fitness reviews** · Evaluated on **39,136 held-out test samples** · Tesla T4 GPU · 3 Epochs

<div align="center">

| Metric | Score |
|--------|-------|
| 🎯 Sentiment F1-Score (3-class) | **93.25%** |
| 🎯 Sentiment Accuracy | **92.96%** |
| 🚨 Churn Detection F1-Score | **86.41%** |
| 🚨 Churn Detection Accuracy | **97.00%** |
| ⚡ Inference Latency | **< 200ms** |
| 📦 Training Dataset | **391,357 reviews** |

</div>

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ChurnLens Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Yelp Open Dataset (6.9M reviews)                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────┐    Filter to 18,857 fitness businesses           │
│   │  data/preprocess │ ──────────────────────────────────────►         │
│   └─────────────────┘    391,357 labeled reviews                        │
│          │                                                              │
│          │  Labels: Stars → Sentiment | Keywords → Churn Risk           │
│          ▼                                                              │
│   ┌──────────────────────────────────────────┐                         │
│   │           Dual-Head BERT Model            │                         │
│   │                                           │                         │
│   │   [Review Text] → [BERT Encoder]          │                         │
│   │                        │                  │                         │
│   │               [CLS Token — 768d]          │                         │
│   │                    ┌───┴───┐              │                         │
│   │                    ▼       ▼              │                         │
│   │              [Sentiment] [Churn]          │                         │
│   │              pos/neu/neg  yes/no          │                         │
│   └──────────────────────────────────────────┘                         │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────┐     ┌──────────────────────┐                         │
│   │  FastAPI     │────►│  Streamlit Dashboard  │                        │
│   │  :9000       │     │  Operator Interface   │                        │
│   └─────────────┘     └──────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Multi-Task Learning** — One BERT model, two simultaneous predictions (sentiment + churn)
- **Weak Supervision** — 391K labeled samples generated automatically from star ratings and keyword heuristics — zero manual annotation
- **Production-Ready API** — FastAPI with Pydantic validation, auto OpenAPI docs, async endpoints
- **Demo Mode** — App runs with rule-based predictions even without a trained model
- **Batch Processing** — Analyze CSVs of hundreds of reviews in seconds
- **Theme Detection** — Auto-detects complaint themes: equipment, staff, cleanliness, pricing, overcrowding
- **Dockerized** — One command to run anywhere

---

## 🗂 Project Structure

```
ChurnLens/
├── data/
│   ├── download_yelp.py          # Extract Yelp tar archive + validate files
│   ├── preprocess.py             # Stream 6.9M reviews, filter, label, split
│   └── generate_sample_data.py   # Synthetic data for development/testing
├── model/
│   ├── bert_model.py             # Dual-head BERT architecture (PyTorch)
│   ├── dataset.py                # PyTorch Dataset + class weight computation
│   ├── train.py                  # Training loop (FP16, warmup, early stopping)
│   ├── evaluate.py               # Metrics: accuracy, F1, confusion matrix
│   └── predict.py                # Inference utilities
├── backend/
│   ├── main.py                   # FastAPI application + endpoints
│   ├── schemas.py                # Pydantic request/response models
│   └── model_loader.py           # Singleton model loader
├── frontend/
│   └── app.py                    # Streamlit operator dashboard
├── configs/
│   └── config.yaml               # All hyperparameters & settings
├── notebooks/
│   ├── FULL_GUIDE.md             # End-to-end setup walkthrough
│   └── training_instructions.md  # Google Colab training guide
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 4GB+ RAM

### 1 — Install Dependencies
```bash
git clone https://github.com/Dakshh-Agarwal/ChurnLens-.git
cd ChurnLens-
pip install -r requirements.txt
```

### 2 — Get Data

> **Yelp dataset requires manual download** — go to https://www.yelp.com/dataset, accept terms, download the JSON zip.

```bash
# Extract the downloaded archive
python data/download_yelp.py --tar-path "/path/to/yelp_dataset.tar"

# Or use synthetic data for instant testing (no download needed)
python data/generate_sample_data.py

# Preprocess & label (streams 6.9M reviews, takes ~5 mins)
python data/preprocess.py
```

### 3 — Train Model

```bash
python model/train.py
```

> **GPU Recommended.** For Google Colab (free T4 GPU) training, see [notebooks/FULL_GUIDE.md](notebooks/FULL_GUIDE.md).

### 4 — Start API Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 9000
```

> Place trained model in `checkpoints/best_model/` for BERT predictions.
> Without a checkpoint, the API runs in **Demo Mode** (rule-based fallback).

### 5 — Launch Dashboard

```bash
streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501**

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict sentiment + churn for a single review |
| `POST` | `/predict/batch` | Batch predictions for multiple reviews |
| `GET` | `/health` | Health check + model mode (BERT vs Demo) |
| `GET` | `/model/info` | Model metadata + accuracy metrics |

### Example

```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The gym is dirty and overcrowded. Thinking about cancelling."}'
```

```json
{
  "sentiment": "negative",
  "sentiment_confidence": 0.94,
  "churn_risk": true,
  "churn_confidence": 0.89,
  "themes": ["cleanliness", "overcrowding"],
  "model_mode": "bert"
}
```

Auto-generated interactive docs available at **http://localhost:9000/docs**

---

## 🏗 Technical Highlights

### Multi-Task BERT Architecture
A single `bert-base-uncased` backbone (109M parameters) shared across two classification heads. Sentiment and churn predictions are generated in one forward pass — 2× more efficient than two separate models, with better generalization through shared representations.

### Weak Supervision at Scale
391K labeled training samples generated automatically:
- **Sentiment** — Yelp star ratings (4-5★ → positive, 3★ → neutral, 1-2★ → negative)
- **Churn Risk** — Domain-specific keyword lexicon (cancel, leaving, switching, worst gym, etc.)

No manual annotation required.

### Training Setup
| Hyperparameter | Value |
|----------------|-------|
| Base Model | `bert-base-uncased` |
| Max Sequence Length | 256 tokens |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| LR Schedule | Linear warmup (10%) + linear decay |
| Optimizer | AdamW (weight decay = 0.01) |
| Precision | FP16 mixed precision |
| Epochs | 4 (early stopping at 3) |
| GPU | Tesla T4 (Google Colab) |
| Training Time | ~3.5 hours |

### Class Imbalance Handling
Inverse-frequency class weights applied to cross-entropy loss:
- Neutral sentiment (6% of data) gets 5.57× higher penalty
- Churn-positive (10% of data) gets 4.94× higher penalty

---

## 🐳 Docker

```bash
docker build -t churnlens .
docker run -p 9000:9000 -v $(pwd)/checkpoints:/app/checkpoints churnlens
```

---

## 📈 Training Results

```
Epoch 1/4 | Train Loss: 0.8211 → Val Sentiment Acc: 0.9008 | Churn F1: 0.8121
Epoch 2/4 | Train Loss: 0.2714 → Val Sentiment Acc: 0.9203 | Churn F1: 0.8479  ← Best
Epoch 3/4 | Train Loss: 0.1853 → Val Sentiment Acc: 0.9296 | Churn F1: 0.8641  ← Best
Early stopping triggered (patience: 2)

Best Combined Metric: 0.9072
Best model saved to: checkpoints/best_model/
```

---

## 📄 License

MIT © 2024 Daksh Agarwal
