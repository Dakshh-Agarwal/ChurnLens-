# ChurnLens 🏋️‍♂️

**NLP-powered churn prediction and sentiment analysis for the fitness industry.**

Fine-tuned BERT on 6M+ Yelp fitness reviews to perform 3-class sentiment classification and binary churn-risk detection. FastAPI backend serves predictions to a Streamlit operator dashboard.

---

## Architecture

```
Raw Yelp Reviews → Preprocess & Label → Fine-tune BERT → FastAPI API → Streamlit Dashboard
```

| Layer       | What's Happening                                      |
|-------------|-------------------------------------------------------|
| **Data**    | Yelp Open Dataset filtered to fitness businesses      |
| **Labels**  | Star ratings → sentiment; keyword rules → churn risk  |
| **Model**   | BERT fine-tuned (dual-head: sentiment + churn)        |
| **Inference** | FastAPI serves real-time predictions                |
| **UI**      | Streamlit dashboard for gym operators                 |

## Key Metrics

- **93.25% sentiment F1-score** (3-class: positive / neutral / negative)
- **86.41% churn detection F1-score** (binary: at-risk / not-at-risk)
- **92.96% sentiment accuracy** on 39K held-out test reviews
- **391K reviews** processed from Yelp Open Dataset (filtered from 6.9M)
- **< 200ms** inference latency per review

## Tech Stack

`Python` · `BERT (HuggingFace Transformers)` · `PyTorch` · `scikit-learn` · `pandas` · `FastAPI` · `Streamlit` · `Docker`

---

## Project Structure

```
ChurnLens/
├── data/
│   ├── download_yelp.py          # Extract Yelp tar archive + validate files
│   ├── preprocess.py             # Filter fitness, label, split
│   └── generate_sample_data.py   # Generate synthetic data for testing
├── model/
│   ├── dataset.py                # PyTorch Dataset class
│   ├── bert_model.py             # Dual-head BERT architecture
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Metrics & evaluation
│   └── predict.py                # Inference utilities
├── backend/
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic models
│   └── model_loader.py           # Model loading singleton
├── frontend/
│   └── app.py                    # Streamlit dashboard
├── configs/
│   └── config.yaml               # All hyperparameters
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get & Preprocess Data

> **Yelp dataset requires manual download** — go to https://www.yelp.com/dataset, accept terms, download the JSON zip, then:

```bash
# Extract the archive (adjust path to where you downloaded it)
python data/download_yelp.py --tar-path "/path/to/yelp_dataset.tar"

# Or use synthetic data for testing (no download needed)
python data/generate_sample_data.py

# Then preprocess
python data/preprocess.py
```

### 3. Train Model
```bash
python model/train.py
```
**Recommended (GPU Required):** For faster training on the full dataset, use Google Colab. See [notebooks/training_instructions.md](notebooks/training_instructions.md) for step-by-step instructions.

### 4. Start API Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5. Launch Dashboard
```bash
streamlit run frontend/app.py
```

---

## API Endpoints

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| POST   | `/predict`         | Predict sentiment + churn for a review |
| POST   | `/predict/batch`   | Batch predictions                    |
| GET    | `/health`          | Health check                         |
| GET    | `/model/info`      | Model metadata                       |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The gym is dirty and overcrowded. Thinking about cancelling."}'
```

### Example Response
```json
{
  "sentiment": "negative",
  "sentiment_confidence": 0.94,
  "churn_risk": true,
  "churn_confidence": 0.87,
  "themes": ["cleanliness", "overcrowding"]
}
```

---

## License

MIT
