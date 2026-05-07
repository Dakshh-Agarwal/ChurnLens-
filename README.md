<div align="center">

<h1>🏋️ ChurnLens</h1>

<p><strong>Multi-signal churn prediction & sentiment analysis for the fitness industry</strong></p>

<p>Fine-tuned BERT on 391K real Yelp fitness reviews for sentiment + churn detection, fused with a behavioral risk engine (visit frequency, class bookings, membership tenure) to produce hybrid churn scores — served through a production FastAPI backend and an interactive Streamlit operator dashboard.</p>

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

1. **Reads member reviews** (typed in, pasted CSV, or uploaded CSV file)
2. **Classifies sentiment** — Positive / Neutral / Negative (93.25% F1)
3. **Flags text-based churn risk** — Is this member saying they'll leave? (86.41% F1)
4. **Scores behavioral risk** — Visit frequency, class bookings, membership tenure
5. **Produces hybrid churn score** — 50% text signal + 50% behavioral signal
6. **Identifies complaint themes** — Equipment, Staff, Cleanliness, Pricing, Overcrowding, Classes, Facilities, Hours
7. **Displays per-location analytics** — multi-gym churn comparison dashboard

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
┌──────────────────────────────────────────────────────────────────────────┐
│                        ChurnLens Pipeline                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Yelp Open Dataset (6.9M reviews)                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────────┐    Filter to 18,857 fitness businesses            │
│   │  data/preprocess │ ─────────────────────────────────────►           │
│   └─────────────────┘    391,357 labeled reviews                         │
│          │                                                               │
│          │  Labels: Stars → Sentiment | Keywords → Churn Risk            │
│          ▼                                                               │
│   ┌──────────────────────────────────────────┐                          │
│   │           Dual-Head BERT Model            │                          │
│   │   [Review Text] → [BERT Encoder]          │                          │
│   │               [CLS Token — 768d]          │                          │
│   │                    ┌───┴───┐              │                          │
│   │                    ▼       ▼              │                          │
│   │              [Sentiment] [Churn]          │                          │
│   │              pos/neu/neg  yes/no          │                          │
│   └──────────────────────────────────────────┘                          │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────────────────────────────────┐                          │
│   │        Multi-Signal Churn Scoring         │                          │
│   │                                           │                          │
│   │   ┌────────────┐    ┌─────────────────┐  │                          │
│   │   │ BERT Text  │    │  Behavioral     │  │                          │
│   │   │ Score 50%  │    │  Risk Score 50% │  │                          │
│   │   └─────┬──────┘    └───────┬─────────┘  │                          │
│   │         └───────┬───────────┘             │                          │
│   │                 ▼                         │                          │
│   │        Hybrid Churn Score                 │                          │
│   │        (threshold > 0.35)                 │                          │
│   └──────────────────────────────────────────┘                          │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐     ┌──────────────────────┐                          │
│   │  FastAPI     │────►│  Streamlit Dashboard  │                         │
│   │  :9000       │     │  Multi-Location View  │                         │
│   └─────────────┘     └──────────────────────┘                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Multi-Task Learning** — One BERT model, two simultaneous predictions (sentiment + churn)
- **Multi-Signal Churn Scoring** — Blends BERT text analysis (50%) with behavioral risk scoring (50%) for hybrid churn detection
- **Weak Supervision** — 391K labeled samples generated automatically from star ratings and keyword heuristics — zero manual annotation
- **Production-Ready API** — FastAPI with Pydantic validation, auto OpenAPI docs, async endpoints
- **Multi-Location Analytics** — Upload a CSV with a `location` column to compare churn risk across gym branches
- **Behavioral Risk Engine** — Rule-based scoring on visit frequency, class bookings, and membership tenure
- **CSV Upload + Paste** — Upload a `.csv` file or paste CSV-formatted text directly — auto-detected
- **Downloadable Results** — Export all predictions (including hybrid scores) as CSV
- **Theme Heatmap** — Visual heatmap showing which complaint themes dominate at each location
- **Demo Mode** — App runs with rule-based predictions even without a trained model
- **Dockerized** — One command to run anywhere

---

## 🖥 Dashboard — How It Works

The Streamlit dashboard (`frontend/app.py`) connects to the FastAPI backend and has **two tabs**:

### 🔍 Review Analyzer
> *"A member just left a Google review — is this person about to churn?"*

Paste a single review → hit **Analyze** → instantly see:
- **Sentiment** with confidence bar (e.g., Negative — 99.5%)
- **Churn Risk** flag (e.g., ⚠️ HIGH RISK — 99.9%)
- **Detected Themes** — equipment, staff, cleanliness, pricing, overcrowding
- **Full confidence breakdown** chart across all sentiment classes

### 📊 Batch Analysis + Dashboard
> *"Analyze 500 reviews across 6 gym locations with behavioral data in one click"*

Upload a CSV (or paste CSV text) → BERT analyzes every review → behavioral risk scored → one unified page shows:

- **KPI cards** — total reviews, positive %, negative %, **hybrid churn %**, behavioral risk %
- **Churn Risk by Location** — bar chart with 20% risk threshold line
- **Sentiment by Location** — stacked bar chart (positive/neutral/negative)
- **Theme Heatmap** — which complaints dominate at each location
- **Location Summary Cards** — per-gym risk level, sentiment, behavioral risk, top complaint
- **📥 Download Results CSV** — export all predictions with hybrid scores
- **Collapsible Detailed Results** — every review with its prediction

### 🧠 Multi-Signal Churn Architecture

ChurnLens uses a **hybrid scoring approach** that combines two independent churn signals:

| Signal | Source | Weight | What It Catches |
|--------|--------|--------|-----------------|
| **Text Signal** | BERT model | 50% | Members who *say* they're leaving |
| **Behavioral Signal** | Rule-based engine | 50% | Members who *act* like they're leaving |

**Behavioral risk factors:**
- `visits_last_month ≤ 1` → 45% risk (stopped coming)
- `classes_booked = 0` → 25% risk (disengaged)
- `membership_months > 12` + low visits → 20% risk (veteran burnout)
- `membership_months ≤ 2` → 10% risk (new member undecided)

**CSV Format (full):**
```csv
location,text,visits_last_month,membership_months,classes_booked
Downtown Fitness,"Great gym!",12,8,4
Westside Gym,"Thinking about cancelling",2,14,0
```


## 🗂 Project Structure

```
ChurnLens/
├── data/
│   ├── download_yelp.py          # Extract Yelp tar archive + validate files
│   ├── preprocess.py             # Stream 6.9M reviews, filter, label, split
│   ├── generate_sample_data.py   # Synthetic data for development/testing
│   └── sample_100_reviews.csv    # 100-review demo CSV with behavioral data
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
│   └── app.py                    # Streamlit dashboard (multi-signal scoring)
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

## ⚠️ Known Limitations

| Limitation | Impact | Potential Solution |
|-----------|--------|-------------------|
| **Churn labels are keyword-based** | ~20% label noise — keywords like "cancel" in *"I'd never cancel, this place is perfect"* cause false positives | Use actual CRM cancellation data as ground truth |
| **Behavioral data is simulated** | Rule-based scoring uses synthetic visit/class data, not real CRM exports | Integrate with gym management APIs (Mindbody, ClubReady) |
| **50/50 blend is a heuristic** | Text/behavioral weights aren't optimized — equal weighting may not be ideal for all gyms | Cross-validate blend weights on real cancellation data |
| **No temporal modeling** | Reviews are treated independently — no tracking of a member's sentiment trend over time | Add user-level aggregation (LSTM/time-series on review history) |
| **English only** | Non-English reviews are ignored during preprocessing | Add multilingual BERT (`bert-base-multilingual-cased`) |
| **CPU inference** | ~200ms/review on CPU — sufficient for single reviews, but batch processing of 10K+ reviews is slow | Distill to DistilBERT (2× faster) or add GPU inference |

---

## 🔮 Future Improvements

**Short-term:**
- [ ] Add confusion matrix visualization to evaluation pipeline
- [ ] Deploy API to cloud (Railway / GCP Cloud Run) for live demo access
- [ ] Add dashboard screenshots to README
- [ ] Export per-class precision/recall/F1 breakdown in training logs

**Medium-term:**
- [x] ~~**Multi-signal churn scoring** — Blend BERT text analysis with behavioral risk engine~~ ✅ Done
- [x] ~~**Multi-location analytics** — Per-gym churn comparison with theme heatmaps~~ ✅ Done
- [ ] **Model distillation** — Distill BERT into DistilBERT (66M → 40M params) for 2× faster inference with <2% accuracy loss
- [ ] **Active learning** — Flag low-confidence predictions for human review to iteratively improve label quality
- [ ] **A/B testing framework** — Measure whether acting on churn predictions actually reduces cancellations

**Long-term:**
- [ ] **Real CRM integration** — Connect to Mindbody / ClubReady APIs for real visit data, payment history, and cancellation ground truth
- [ ] **Train behavioral model** — Replace rule-based scoring with XGBoost trained on actual cancellation records
- [ ] **User-level churn trajectory** — Aggregate a member's reviews over time to detect sentiment trends before churn
- [ ] **Configurable risk thresholds** — Let operators adjust churn sensitivity per location based on business tolerance

---

## 📄 License

MIT © 2025 Daksh Agarwal
