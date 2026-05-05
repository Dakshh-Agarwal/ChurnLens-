# ChurnLens — Complete Step-by-Step Guide
# From Yelp Download → Preprocessing → Colab Training → Local Dashboard

---

## PHASE 1 — Download & Extract Yelp Data (Your PC)

### Step 1.1 — Download the Dataset
1. Go to https://www.yelp.com/dataset
2. Click **"Download JSON"** (NOT the photos one)
3. Fill in the form → accept terms
4. File saves to Downloads: `yelp_dataset.tar` (~4.35 GB)

### Step 1.2 — Extract the Archive
Open PowerShell inside your ChurnAi folder and run:

```powershell
python data/download_yelp.py --tar-path "C:\Users\YOUR_USERNAME\Downloads\yelp_dataset.tar"
```

This creates `data/raw/` and extracts:
- `yelp_academic_dataset_business.json` (~120 MB)
- `yelp_academic_dataset_review.json`   (~5 GB, 6M+ reviews)

Expected output:
```
SUCCESS - All required dataset files found.
yelp_academic_dataset_business.json: 0.12 GB
yelp_academic_dataset_review.json: 5.34 GB
```

---

## PHASE 2 — Preprocess Data (Your PC)

### Step 2.1 — Run Preprocessing
```powershell
python data/preprocess.py
```

Takes **5–15 minutes**. What it does:
- Reads 6M+ reviews ONE LINE AT A TIME (no RAM crash)
- Keeps ONLY gym/fitness reviews (~50,000 reviews)
- Auto-labels sentiment:
  - ⭐⭐⭐⭐⭐ 4–5 stars → positive
  - ⭐⭐⭐ 3 stars     → neutral
  - ⭐⭐ 1–2 stars    → negative
- Flags churn risk (keywords: "cancelling", "leaving", "switching")
- Detects themes: cleanliness, equipment, staff, overcrowding
- Splits 80% train / 10% val / 10% test
- Saves to `data/processed/` (only ~40 MB total)

Expected output:
```
Processed 1,000,000 reviews... Found 3,421 matches
Processed 2,000,000 reviews... Found 7,891 matches
...
Final filtered dataset: 52,000 reviews
Train: 41,600 | Val: 5,200 | Test: 5,200
SUCCESS - Preprocessing complete!
```

### Step 2.2 — Upload Processed Data to Google Drive
- Open https://drive.google.com
- Create a folder called `ChurnLens`
- Upload the `data/processed/` folder (only ~40 MB — do NOT upload `data/raw/`)

---

## PHASE 3 — Push Code to GitHub (Your PC)

### Step 3.1 — Create a GitHub Repo
1. Go to https://github.com/new
2. Name it `ChurnAi` → Private → Create (do NOT add README/gitignore)

### Step 3.2 — Push
```powershell
git remote add origin https://github.com/YOUR_USERNAME/ChurnAi.git
git push -u origin master
```

What gets pushed ✅ — all Python files, configs, Dockerfile, README  
What stays off GitHub ❌ — data/raw/, data/processed/, checkpoints/ (all in .gitignore)

---

## PHASE 4 — Train on Google Colab (FREE T4 GPU)

### Step 4.1 — Open Colab with GPU
1. Go to https://colab.research.google.com
2. File → New Notebook
3. Runtime → Change runtime type → **T4 GPU** → Save

### Step 4.2 — Clone & Install
New cell, paste and run:

```python
!git clone https://github.com/YOUR_USERNAME/ChurnAi.git
%cd ChurnAi
!pip install -r requirements.txt
```

### Step 4.3 — Get Your Data from Google Drive
New cell:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil, os

# Copy your preprocessed data from Drive into Colab
shutil.copytree('/content/drive/MyDrive/ChurnLens/', 'data/processed/')

# Verify
for f in os.listdir('data/processed/'):
    size = os.path.getsize(f'data/processed/{f}')
    print(f"{f}: {size/1024/1024:.1f} MB")
```

Expected output:
```
train.parquet: 28.4 MB
val.parquet: 3.6 MB
test.parquet: 3.6 MB
label_stats.json: 0.0 MB
```

### Step 4.4 — Train the Model
New cell:

```python
!python model/train.py
```

What happens:
- Downloads bert-base-uncased weights (~440 MB, ~2 mins)
- Trains 3 epochs on T4 GPU (~30–60 mins total)
- Saves best checkpoint to `checkpoints/best_model/`

Expected training output:
```
Using device: cuda
GPU: Tesla T4
Model initialized: 109,778,565 total params

Epoch 1/3 | Step 50/1300 | Loss: 1.2341 | LR: 2.00e-05
Epoch 1/3 | Step 100/1300 | Loss: 0.9821 | LR: 2.00e-05
...
Val Loss: 0.4231 | Sentiment Acc: 0.8812 | Churn F1: 0.8243
SUCCESS - New best model saved! Combined metric: 0.8612
...
Training complete! Best combined metric: 0.8612
```

### Step 4.5 — Download the Trained Model
New cell — zips everything and downloads:

```python
!zip -r best_model.zip checkpoints/best_model/
from google.colab import files
files.download('best_model.zip')
```

Files inside the zip:
- `model.pt`               (~420 MB — the actual trained weights)
- `model_info.json`        (~1 KB  — accuracy/metric info)
- `config.json`            (~1 KB  — BERT architecture config)
- `vocab.txt`              (~200 KB — tokenizer vocabulary)
- `tokenizer_config.json`  (~1 KB)

---

## PHASE 5 — Switch Dashboard to Real BERT (Your PC)

### Step 5.1 — Place the Model Files
Extract `best_model.zip` into your project:

```
ChurnAi/
└── checkpoints/
    └── best_model/         ← create this folder
        ├── model.pt
        ├── model_info.json
        ├── config.json
        ├── vocab.txt
        └── tokenizer_config.json
```

### Step 5.2 — Restart the API
```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 9000
```

You'll see this (no more Demo Mode!):
```
INFO - Checkpoint found! Loading BERT model...
INFO - ChurnLens BERT model loaded successfully
INFO - Sentiment accuracy: 0.881 | Churn F1: 0.824
Application startup complete.
```

### Step 5.3 — Open the Dashboard
```powershell
streamlit run frontend/app.py --server.port 8501
```

Open http://localhost:8501 — the sidebar will say **"BERT Model"** ✅

---

## ✅ Checklist

| # | Action | Done? |
|---|--------|-------|
| 1 | Download Yelp JSON from yelp.com/dataset | ⬜ |
| 2 | `python data/download_yelp.py --tar-path "..."` | ⬜ |
| 3 | `python data/preprocess.py` (5–15 mins) | ⬜ |
| 4 | Upload `data/processed/` to Google Drive | ⬜ |
| 5 | Push code to GitHub | ⬜ |
| 6 | Open Colab → set T4 GPU runtime | ⬜ |
| 7 | Clone repo + `pip install -r requirements.txt` | ⬜ |
| 8 | Mount Drive + copy data + `python model/train.py` | ⬜ |
| 9 | Download `best_model.zip` from Colab | ⬜ |
| 10 | Extract zip into `checkpoints/best_model/` | ⬜ |
| 11 | Restart API — should say "BERT Model loaded" | ⬜ |
| 12 | Open dashboard at http://localhost:8501 | ⬜ |

---

## Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| Extraction takes long | Normal — 4 GB tar takes 5–10 mins |
| Preprocess crashes | Already fixed to stream line-by-line |
| Colab disconnects mid-training | Re-run cell — checkpoint saved each epoch |
| `model.pt` too big to download | Use the zip method in Step 4.5 |
| API still shows "Demo Mode" | Check path is exactly `checkpoints/best_model/` |
| Out of memory on Colab | In `configs/config.yaml` reduce `batch_size: 16` → `batch_size: 8` |
| Colab training too slow | Normal — T4 does ~3–5 steps/sec, 1300 steps ≈ 7 mins/epoch |
