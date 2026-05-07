# ChurnLens — Complete Interview Prep Guide
## Everything you need to explain this project end-to-end

---

## 1. The 30-Second Pitch

**ChurnLens** is a multi-signal churn prediction system for the fitness industry. It combines **BERT-based NLP** (analyzing what members *say* in reviews) with a **behavioral risk engine** (analyzing what members *do* — visit frequency, class bookings, membership tenure) to produce a **hybrid churn score**.

A gym operator uploads a CSV of member reviews with optional behavioral data → the system runs each review through a fine-tuned BERT model for sentiment and text-based churn detection → a rule-based engine scores behavioral risk → both signals are blended 50/50 → the dashboard shows per-location breakdowns, theme heatmaps, and actionable risk cards.

**One sentence:** *"I fine-tuned BERT on 391K real gym reviews to predict churn and sentiment, then built a hybrid scoring system that fuses NLP predictions with behavioral risk signals, served through a production API and interactive dashboard."*

---

## 2. The Business Problem

**Churn** = members cancelling gym memberships. Industry average: **50% per year**.

**Why it's hard:** Most churning members don't write reviews — they just stop coming. The ones who DO write reviews often don't say "I'm cancelling" — they say "equipment is dirty" or "too crowded."

**The two types of churners:**

| Type | Signal | Who catches them |
|------|--------|-----------------|
| **Vocal churners** | Write angry reviews: "I'm cancelling" | BERT text analysis |
| **Silent churners** | Stop visiting, stop booking classes | Behavioral risk engine |

**ChurnLens catches both** — that's the key differentiator vs. a pure NLP approach.

**Who are the users?**
- **Gym managers** → paste individual reviews for instant risk assessment
- **Gym chains (multi-location)** → upload CSV of reviews + behavioral data → see which branches are bleeding members

---

## 3. The Dataset — Yelp Open Dataset

**Source:** [yelp.com/dataset](https://yelp.com/dataset) — publicly available for academic use.

**Raw data:**
- `yelp_academic_dataset_review.json` — **6.9 million reviews** (5.34 GB)
- `yelp_academic_dataset_business.json` — **150,346 businesses** (120 MB)

**Our filtering pipeline:**
```
6.9M reviews → filter to 18,857 fitness businesses → 391,357 gym reviews
```

**Why Yelp?**
- Free and legal (no scraping, no ToS violations)
- Real-world messy language (not clean academic text)
- Star ratings = **free sentiment labels** (no manual annotation)
- Massive scale = noise averages out

---

## 4. The Labeling Strategy — Weak Supervision

> **Interview gold:** *"We used weak supervision to auto-generate 391K labeled samples without a single human annotator."*

### Sentiment Labels (from star ratings)
```
4-5 stars → positive (68% of data)
3 stars   → neutral  (6% of data)
1-2 stars → negative (26% of data)
```

**Why it works:** Star ratings are a reliable proxy. A 5-star review is almost always positive. At 391K samples, the noise from edge cases (sarcastic 5-star reviews) averages out.

### Churn Labels (from keyword matching)
Keywords: `"cancelling"`, `"leaving"`, `"switching gym"`, `"waste of money"`, `"never coming back"`

**Result:** 10.1% churn rate (39,598 / 391,357) — realistic vs. real-world gym churn (5-15%).

### The Honest Limitation
> *"Our churn labels come from keywords in reviews — but most unhappy members don't write reviews at all. The real churn signal is silent. That's exactly why we built the behavioral risk engine — it catches what text analysis misses."*

This limitation is a **strength** in interviews. It shows you understand real-world ML challenges and you actually did something about it.

---

## 5. The Model — Dual-Head Fine-tuned BERT

### What is BERT? (Explain it simply)

**BERT** = Bidirectional Encoder Representations from Transformers (Google, 2018).

- Pre-trained on BookCorpus (800M words) + Wikipedia (2.5B words)
- Reads text **bidirectionally** — understands each word in context of ALL surrounding words
- Two pre-training tasks: Masked Language Modeling (predict hidden words) + Next Sentence Prediction

**The analogy:** BERT is like hiring someone who speaks fluent English. Fine-tuning is teaching them gym industry terminology. You don't re-teach grammar.

### Our Architecture — Dual-Head BERT

```
[Review Text]
     ↓
[Tokenizer — WordPiece, max 256 tokens]
     ↓
[BERT Encoder — 12 transformer layers, 768 hidden dims]
     ↓
[CLS Token Embedding — 768 dims]
     ↓
  ┌──────────────┬──────────────┐
  ↓              ↓
[Dropout(0.3)]  [Dropout(0.3)]
  ↓              ↓
[Dense → 3]    [Dense → 2]
  ↓              ↓
[Sentiment]    [Churn Risk]
 positive       True/False
 neutral
 negative
```

**Why one model with two heads?**
1. **Shared representations** — sentiment and churn are correlated (negative reviews → higher churn risk). Shared BERT layers learn features useful for both.
2. **Efficiency** — one forward pass = both predictions. 2x faster than two models.
3. **Regularization** — multi-task learning prevents overfitting on either task alone.

**The `[CLS]` token:** BERT adds a special token at position 0. After processing, its 768-dim hidden state summarizes the entire input. We feed this into both classification heads.

### The Loss Function

```
Total Loss = 0.6 × Sentiment_CrossEntropyLoss + 0.4 × Churn_BCELoss
```

- Sentiment is primary (0.6) because labels are more reliable (from star ratings)
- Churn is secondary (0.4) because labels are noisier (from keywords)

### Handling Class Imbalance

**Problem:** Positive reviews are 68% — model could predict "positive" for everything and get 68% accuracy.

**Solution:** Inverse-frequency class weights:
```
weight = total_samples / (num_classes × samples_in_class)
```

| Class | Data % | Weight | Effect |
|-------|--------|--------|--------|
| Positive | 68% | 0.49 | Penalize less for errors |
| Neutral | 6% | 5.57 | Penalize heavily for errors |
| Negative | 26% | 1.29 | Moderate penalty |
| Churn True | 10% | High | Force model to catch churn |
| Churn False | 90% | Low | Don't let majority dominate |

---

## 6. Training Details

| Setting | Value | Why |
|---------|-------|-----|
| **GPU** | Google Colab T4 (16GB VRAM) | Free, sufficient for BERT |
| **Optimizer** | AdamW | Adam + weight decay (L2 regularization) |
| **Learning Rate** | 2e-5 | Standard for BERT fine-tuning — avoids catastrophic forgetting |
| **Weight Decay** | 0.01 | Prevents overfitting |
| **Batch Size** | 32 | Largest that fits in T4 memory with FP16 |
| **Max Length** | 256 tokens | Covers 95%+ of reviews without excessive padding |
| **Epochs** | 4 | Early stopping if validation doesn't improve for 2 epochs |
| **Precision** | FP16 (mixed precision) | Half memory, 2x speed, negligible accuracy loss |
| **Training Time** | ~3.5 hours | On 313K training samples |

### Learning Rate Schedule: Linear Warmup + Decay

```
LR
^
|      /\
|     /  \
|    /    \____________
|   /
|  /  warmup (10%)
+-------------------------> steps
```

**Why warmup?** At training start, the new classification heads have random weights. A high LR would make wild updates that corrupt BERT's pre-trained knowledge. Warmup lets the heads stabilize first.

### Results

| Metric | Score |
|--------|-------|
| Sentiment Accuracy | ~89% |
| Churn F1 Score | ~85% |
| Test Set Size | 39K samples (stratified hold-out) |

**Why F1 for churn?** Accuracy is misleading with imbalanced data — a model predicting "no churn" for everything gets 90% accuracy but catches 0% of churners. F1 balances precision (don't flag happy members) and recall (don't miss at-risk members).

---

## 7. Multi-Signal Churn Architecture

This is the **most interview-impressive** part of the project.

### The Problem with Text-Only Churn Detection

> *"BERT catches members who SAY they're leaving. But most churners are silent — they just stop coming."*

### The Solution: Hybrid Scoring

```
                    CSV Input
                       ↓
        ┌──────────────┼──────────────┐
        ↓                             ↓
   [BERT Model]              [Behavioral Engine]
   Analyzes review text       Scores visit patterns
        ↓                             ↓
   Text Score (0-1)          Behavioral Score (0-1)
        ↓                             ↓
        └──────── Weighted Blend ─────┘
                  (50% + 50%)
                       ↓
              Hybrid Churn Score
                       ↓
              Threshold > 0.35?
                  ↓         ↓
                YES         NO
              AT RISK     SAFE
```

### Behavioral Risk Scoring Rules

```python
def compute_behavioral_risk(visits, classes, months):
    score = 0.0
    
    # Visit frequency (biggest signal)
    if visits <= 1:   score += 0.45   # stopped coming entirely
    elif visits <= 3: score += 0.30   # dramatically reduced
    elif visits <= 6: score += 0.10   # declining
    
    # Class engagement
    if classes == 0:  score += 0.25   # zero engagement
    elif classes <= 1: score += 0.10  # minimal
    
    # Tenure patterns
    if months > 12 and visits <= 3:
        score += 0.20   # veteran burnout (long member, low activity)
    elif months <= 2:
        score += 0.10   # new member still deciding
    
    return min(score, 1.0)
```

### How The Blend Works

| Scenario | Text Score | Behavioral Score | Hybrid | Result |
|----------|-----------|-----------------|--------|--------|
| Angry review, stopped visiting | 0.95 | 0.90 | 0.93 | 🔴 HIGH RISK |
| Angry review, still visiting regularly | 0.95 | 0.10 | 0.53 | 🟡 MEDIUM RISK |
| Happy review, stopped visiting | 0.05 | 0.70 | 0.38 | 🟡 MEDIUM RISK |
| Happy review, visits regularly | 0.05 | 0.05 | 0.05 | 🟢 SAFE |

**The key insight:** Row 3 — a member who writes "great gym!" but hasn't visited in 3 weeks. Pure NLP says safe. Behavioral engine catches the silent churn.

### Why Rules Instead of ML for Behavioral Scoring?

> *"We use rule-based scoring for behavioral data because: (1) We don't have real cancellation labels to train on. (2) The risk factors are well-understood — low visits = high risk is a business heuristic, not a pattern that needs ML to discover. (3) Rules are interpretable — a gym manager can understand 'visits dropped below 3' better than a neural network confidence score. In production with real CRM data, you'd train a gradient boosted model on actual cancellation records."*

---

## 8. The Data Pipeline

```
Yelp TAR (4.35 GB)
       ↓
download_yelp.py (extract)
       ↓
data/raw/ (8.65 GB JSON files)
       ↓
preprocess.py (stream 6.9M reviews line-by-line)
       ↓
Filter to 18,857 fitness businesses → 391,357 reviews
       ↓
Label: sentiment (from stars) + churn_risk (from keywords)
       ↓
Train/Val/Test split: 80/10/10 stratified by sentiment
       ↓
data/processed/ (163 MB parquet files)
       ↓
model/train.py (BERT fine-tuning on Colab T4 GPU)
       ↓
checkpoints/best_model/ (saved weights)
```

### Memory-Efficient Streaming
The review file is **5.34 GB** — too large for RAM.

**Our approach:** Read one JSON line at a time, use `set` lookup for O(1) business ID filtering:
```python
fitness_ids = set(...)  # 18,857 IDs
with open("reviews.json") as f:
    for line in f:
        record = json.loads(line)
        if record["business_id"] in fitness_ids:  # O(1) lookup
            process(record)
```

**Performance fix we made:** Originally used `any(id in line for id in fitness_ids)` — O(n×m) = ~132 billion operations. Fixed to `set` lookup → processing dropped from estimated 23 hours to 3 minutes.

### Why Parquet?
- **5-10x smaller** than CSV (column compression)
- **Typed columns** — preserves int/bool/string (no parsing bugs)
- **Column-oriented** — faster reads for specific columns
- Native pandas support

---

## 9. The API — FastAPI

### Why FastAPI over Flask?
| FastAPI | Flask |
|---------|-------|
| Async by default | Sync by default |
| Auto OpenAPI docs (/docs) | Manual docs |
| Pydantic validation built-in | Manual validation |
| 2-3x faster throughput | Slower |
| Type hints enforced | Optional |

### Key API Design Decisions

**Singleton model loading:** Load BERT once on first request, reuse forever.
```python
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ChurnLensPredictor()  # ~5 sec, 1.5GB
    return _predictor
```

**Chunked batch processing:** Process reviews in chunks of 16 to prevent OOM:
```python
for i in range(0, len(texts), 16):
    chunk = texts[i:i + 16]
    results.extend(predictor.predict_batch(chunk))
```

**Demo mode fallback:** API runs with rule-based predictions when no trained model exists — prevents crashes during development.

### API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Single review → sentiment + churn |
| `/predict/batch` | POST | Up to 500 reviews at once |
| `/health` | GET | API status + model loaded check |
| `/model/info` | GET | Model metadata |

---

## 10. The Dashboard — Streamlit

### Two tabs:

**🔍 Review Analyzer** — Single review analysis
- Paste one review → instant prediction
- Shows: sentiment badge, churn risk flag, confidence bars, detected themes, score distribution chart

**📊 Batch Analysis + Dashboard** — Multi-location intelligence
- Upload CSV or paste CSV text (auto-detected)
- Supports 5 columns: `location`, `text`, `visits_last_month`, `membership_months`, `classes_booked`
- Shows (in one unified page):
  - **5 KPI cards** — total reviews, positive %, negative %, hybrid churn %, behavioral risk %
  - **Churn Risk by Location** — bar chart with 20% threshold line
  - **Sentiment by Location** — stacked bar chart
  - **Theme Heatmap** — complaints per location (8 themes × N locations)
  - **Location Summary Cards** — per-gym risk level with all metrics
  - **📥 Download Results CSV** — exports all predictions including hybrid scores
  - **Collapsible Detailed Results** — every review with its prediction

### CSV Auto-Detection
When users paste CSV text into the text area, the app automatically:
1. Detects if the first line is a CSV header (contains "text" and has commas)
2. Parses location + behavioral columns if present
3. Falls back to plain text mode if no CSV format detected

---

## 11. Infrastructure

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why Docker?** Reproducible environment, no "works on my machine" issues, easy cloud deployment.

### Tech Stack Summary
| Layer | Technology |
|-------|-----------|
| NLP Model | BERT (bert-base-uncased), PyTorch |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data | Pandas, Parquet |
| Tokenizer | HuggingFace Transformers |
| Container | Docker |
| Training | Google Colab (T4 GPU, FP16) |

---

## 12. Interview Questions & Answers

### On the Model

**Q: Why BERT over simpler models (TF-IDF + Logistic Regression)?**
> "Bag-of-words models lose word order and context. 'Not bad' would be misclassified. BERT understands context bidirectionally, handles negation, sarcasm, and domain-specific language. The tradeoff is compute cost, which we solved with Colab's free T4 GPU."

**Q: Why not GPT-4 / a larger LLM?**
> "Cost and latency. GPT-4 at $0.03/1K tokens × 391K reviews × ~150 tokens = $1,800 just for inference, plus ongoing API costs. BERT runs locally in <200ms per review, costs nothing after training, and for focused classification tasks, fine-tuned BERT on domain data often matches larger models."

**Q: What is fine-tuning vs training from scratch?**
> "Training from scratch = random weights, massive data and compute needed. Fine-tuning = start from pre-trained weights (BERT already knows English), continue training on our specific domain. It's like hiring an English speaker and teaching them gym terminology vs. teaching someone both English AND gym terminology from scratch."

**Q: Why two output heads instead of two models?**
> "Multi-task learning. Sentiment and churn share features — negative reviews correlate with churn. Shared BERT layers learn representations useful for both tasks. One forward pass gives both predictions (2x efficient). Multi-task training also acts as regularization."

**Q: How did you handle class imbalance?**
> "Inverse-frequency class weights in the loss function. Rare classes (neutral at 6%, churn-true at 10%) get higher weights — misclassifying them costs more. This prevents the model from gaming accuracy by always predicting the majority class."

### On the Data

**Q: How did you create labels without manual annotation?**
> "Weak supervision. Star ratings → sentiment labels (5 stars = positive, reliable signal). Keyword matching → churn labels (reviews containing 'cancelling', 'switching gym'). It's not perfect — but at 391K samples, noise averages out. We handle residual imbalance with class weights."

**Q: What about the silent churners — members who don't write reviews?**
> "That's exactly why we built the behavioral risk engine. BERT catches vocal churners. The behavioral engine catches silent ones — members whose visit frequency drops, who stop booking classes, who've been members for 14 months but only visited once last month. The hybrid score combines both signals at 50/50 weight."

### On Engineering

**Q: How does the multi-signal scoring work?**
> "BERT produces a text churn confidence (0-1). The behavioral engine produces a behavioral risk score (0-1) based on visit frequency, class bookings, and membership tenure using business rule thresholds. We blend them 50/50 into a hybrid score. If hybrid > 0.35, we flag as at-risk. This catches both types of churners — vocal and silent."

**Q: Why rules instead of ML for behavioral scoring?**
> "Three reasons: (1) No real cancellation labels to train on. (2) The risk factors are well-understood business heuristics — 'low visits = risk' doesn't need ML to discover. (3) Rules are interpretable for gym managers. In production with real CRM data, you'd train a gradient boosted model on actual cancellations."

**Q: How does the API handle large batches without crashing?**
> "Chunked processing. We split the batch into chunks of 16 reviews, process each through BERT, and concatenate results. This prevents GPU/CPU OOM errors on large uploads (up to 500 reviews). The API also has a 300-second timeout for batch requests."

**Q: Why prioritize pasted text over uploaded files?**
> "UX decision. If a user pastes CSV text AND has an old file in the upload widget, the paste is their active intent. The uploaded file might be stale. Pasted text takes priority to match user expectations."

### Tough Questions

**Q: What are the weaknesses of this system?**
> "Three honest limitations: (1) Churn labels from keywords have false positives ('never coming back, it's too perfect here'). (2) Behavioral data is simulated — in production you'd need real CRM integration. (3) The 50/50 blend weight is a heuristic — with real data, you'd optimize the weights via cross-validation."

**Q: How would you improve this in production?**
> "Four upgrades: (1) Real CRM integration — actual cancellation records as ground truth labels. (2) Retrain BERT on real churn labels instead of keyword proxies. (3) Replace behavioral rules with a trained XGBoost on behavioral features + historical cancellations. (4) A/B test interventions — does reaching out to high-risk members actually reduce churn?"

**Q: What's the biggest technical decision you made?**
> "The dual-signal architecture. Pure NLP misses silent churners. Pure behavioral analysis misses the 'why' behind churn. Combining them gives gym operators both WHO is at risk AND WHY — the review text explains the cause while the behavioral data identifies members who wouldn't otherwise be flagged."

---

## 13. Quick Reference Table

| Component | One Sentence |
|-----------|-------------|
| BERT | Pre-trained transformer fine-tuned on 391K gym reviews to predict sentiment + churn simultaneously |
| Dual-head | One shared BERT encoder + two classification heads = one forward pass for both tasks |
| Weak supervision | Auto-generating labels from star ratings (sentiment) and keywords (churn) without human annotation |
| Behavioral engine | Rule-based scoring on visit frequency, class bookings, and membership tenure to catch silent churners |
| Hybrid scoring | 50% BERT text signal + 50% behavioral signal → catches both vocal and silent churners |
| Class weights | Inverse-frequency weighting to prevent majority class dominance during training |
| FP16 training | 16-bit precision = half memory, 2x speed, negligible accuracy loss |
| Chunked inference | Processing batches in chunks of 16 to prevent OOM on large uploads |
| Singleton pattern | Loading BERT once into memory and reusing for all API requests |
| CSV auto-detect | Frontend automatically parses CSV format when pasted into text area |
| Theme detection | Keyword-based extraction of 8 complaint themes (equipment, staff, cleanliness, etc.) |
| Demo mode | Rule-based fallback when no trained model exists — app never crashes |

---

## 14. The "Walk Me Through Your Code" Answer

> "Starting from the bottom up: We downloaded 6.9M Yelp reviews, streamed them line-by-line to filter 391K fitness reviews, and labeled them using weak supervision — star ratings for sentiment, keywords for churn. We stored the processed data in Parquet files for efficiency.
>
> For the model, we fine-tuned bert-base-uncased with two classification heads — one for 3-class sentiment, one for binary churn — sharing the same BERT encoder. We trained on Colab T4 with mixed precision, AdamW optimizer at 2e-5 learning rate, and inverse-frequency class weights to handle imbalance. Got ~89% sentiment accuracy and ~85% churn F1.
>
> The trained model is served through a FastAPI backend with a singleton pattern — model loads once, serves forever. Batch requests are processed in chunks of 16 to prevent OOM.
>
> On top of that, we built a behavioral risk engine — a rule-based scorer that evaluates visit frequency, class bookings, and membership tenure. The final churn prediction blends BERT's text analysis (50%) with behavioral risk (50%) into a hybrid score.
>
> The Streamlit dashboard ties it all together — upload a CSV with reviews and behavioral data, get per-location churn analysis, theme heatmaps, and actionable risk cards. It's designed for gym operators managing multiple branches."
