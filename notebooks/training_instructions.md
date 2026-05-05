# Training ChurnLens on Google Colab 🚀

Since BERT training is computationally intensive, using Google Colab's free GPU (T4) is highly recommended. Follow these steps:

### 1. Prepare your Repository
- Push your current `ChurnAi` project to a new private or public GitHub repository.
- Ensure your `requirements.txt` is up to date.

### 2. Open Google Colab
- Go to [colab.research.google.com](https://colab.research.google.com).
- Create a new notebook.

### 3. Switch to GPU
- Go to **Runtime** > **Change runtime type**.
- Select **T4 GPU** as the hardware accelerator.

### 4. Run these Commands in Colab
Create a cell and paste the following:

```python
# 1. Clone your repo (replace with your URL)
!git clone https://github.com/YOUR_USERNAME/ChurnAi.git
%cd ChurnAi

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Preprocess Data (if you haven't uploaded preprocessed files)
# Make sure your raw data is in data/raw/
!python data/preprocess.py

# 4. Start Training
!python model/train.py
```

### 5. Download the Model
- Once training is finished, look in the `checkpoints/best_model` folder in the Colab file explorer.
- Download the following files back to your local `ChurnAi/checkpoints/best_model/` folder:
    - `model.pt`
    - `tokenizer/` (all files inside)
    - `model_info.json`

### 6. Run Locally
- After placing the files, your local FastAPI backend will automatically detect the model and switch from "Demo Mode" to "BERT Mode".
