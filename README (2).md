# DermAI — Skin & Nail Disease Detection App

> B.Tech Final Year Project | AI-powered skin & nail disease detection using MobileNetV2 + Flask

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stack)
4. [Quick Start (Demo Mode)](#quick-start-demo-mode)
5. [Full Setup with Real Dataset](#full-setup-with-real-dataset)
6. [Training the Model](#training-the-model)
7. [Running the Backend API](#running-the-backend-api)
8. [API Reference](#api-reference)
9. [Frontend Guide](#frontend-guide)
10. [Diseases Covered](#diseases-covered)
11. [How the AI Works](#how-the-ai-works)
12. [Project Architecture](#project-architecture)
13. [Extending the Project](#extending-the-project)
14. [Disclaimer](#disclaimer)

---

## Project Overview

DermAI is a web application that:
- Accepts an uploaded skin or nail photo from the user
- Runs it through a MobileNetV2 deep-learning model (transfer learning)
- Returns the predicted disease name, confidence percentage, estimated duration,
  precautions, and care tips
- Displays results in a clean, modern UI

Detected conditions: **Acne**, **Eczema**, **Psoriasis**, **Nail Fungus**

---

## Project Structure

```
skin_nail_app/
├── app.py                  ← Flask backend (main server)
├── train_model.py          ← Model training script
├── disease_info.json       ← Disease database (duration, precautions, tips)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── frontend/
│   └── index.html          ← Complete single-file frontend
│
├── dataset/                ← (you create this with real images)
│   ├── train/
│   │   ├── acne/
│   │   ├── eczema/
│   │   ├── nail_fungus/
│   │   └── psoriasis/
│   └── val/
│       ├── acne/
│       ├── eczema/
│       ├── nail_fungus/
│       └── psoriasis/
│
└── model/
    └── skin_nail_model.h5  ← Saved model (created after training)
```

---

## Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Frontend  | HTML5, CSS3, Vanilla JavaScript     |
| Backend   | Python 3.10+, Flask, Flask-CORS     |
| AI Model  | TensorFlow 2.x, Keras, MobileNetV2 |
| Images    | Pillow (PIL)                        |
| Database  | JSON file (disease_info.json)       |

---

## Quick Start (Demo Mode)

No dataset or GPU required. The app runs with simulated predictions.

```bash
# 1. Clone / unzip the project
cd skin_nail_app

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies (TensorFlow is optional for demo)
pip install flask flask-cors pillow numpy

# 4. Run the server
python app.py

# 5. Open browser
#    http://localhost:5000
```

In demo mode a banner "⚡ Demo Mode" appears in the top-right corner.
All confidence values are simulated but the full UI and API are functional.

---

## Full Setup with Real Dataset

### Step 1 — Get Images

Recommended free sources:
- **Kaggle Skin Disease Dataset** — search "skin disease classification"
- **DermNet NZ** — https://dermnetnz.org (free for educational use)
- **ISIC Archive** — https://www.isic-archive.com

You need at least **100–200 images per class** for reasonable accuracy;
500+ per class is better.

### Step 2 — Organise Images

```
dataset/
    train/
        acne/          ← ~80% of acne images
        eczema/        ← ~80% of eczema images
        nail_fungus/   ← ~80% of nail fungus images
        psoriasis/     ← ~80% of psoriasis images
    val/
        acne/          ← ~20% of acne images
        eczema/
        nail_fungus/
        psoriasis/
```

### Step 3 — Install Full Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

### Option A — Demo Training (no real images)
```bash
python train_model.py --demo
```
This creates tiny synthetic images just to verify the pipeline. The resulting
model will NOT give accurate predictions on real photos.

### Option B — Real Training
```bash
python train_model.py
```

Training has two stages automatically:

**Stage 1 — Feature Extraction** (20 epochs)
- MobileNetV2 base frozen
- Only the custom classification head is trained
- Fast; sets up good initial weights

**Stage 2 — Fine-Tuning** (10 epochs)
- Top 30 layers of MobileNetV2 unfrozen
- Lower learning rate (LR/10)
- Improves accuracy on your specific dataset

Callbacks used:
- `ModelCheckpoint` — saves best model by `val_accuracy`
- `EarlyStopping` — stops training if `val_loss` plateaus for 5 epochs
- `ReduceLROnPlateau` — halves LR when `val_loss` stagnates

After training, `model/skin_nail_model.h5` is created and a plot is saved to
`model/training_history.png`.

### Expected Training Time
| Hardware         | Approx. Time     |
|------------------|------------------|
| CPU only         | 2–6 hours        |
| GPU (NVIDIA)     | 20–40 minutes    |
| Google Colab GPU | 30–60 minutes    |

---

## Running the Backend API

```bash
python app.py
```

The server starts at `http://localhost:5000`.

---

## API Reference

### `POST /predict`

Upload an image and get a prediction.

**Request:**
```
Content-Type: multipart/form-data
Field: image   (PNG / JPG / JPEG / WEBP)
```

**Response (200 OK):**
```json
{
  "disease":       "Eczema (Atopic Dermatitis)",
  "disease_key":   "eczema",
  "confidence":    87.3,
  "confidence_map": {
    "acne":        3.1,
    "eczema":      87.3,
    "nail_fungus": 2.4,
    "psoriasis":   7.2
  },
  "duration":      "Can develop within days of trigger exposure…",
  "precautions":   ["Identify and avoid triggers…", "…"],
  "care_tips":     ["Apply a thick moisturiser…", "…"],
  "severity":      "Moderate to Severe",
  "description":   "Eczema is a chronic inflammatory…",
  "demo_mode":     false
}
```

**Error Response (400 / 500):**
```json
{ "error": "No image file provided." }
```

### `GET /diseases`
Returns the full disease info JSON database.

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "demo_mode": false }
```

---

## Frontend Guide

`frontend/index.html` is a single self-contained file with no dependencies.

Features:
- Drag-and-drop or click-to-browse image upload
- Live image preview with remove button
- Animated spinner with cycling messages during API call
- Results panel: disease name, description, severity chip, confidence bar,
  estimated duration, precautions list, care tips list
- Per-class confidence breakdown with animated mini-bars
- Medical disclaimer banner
- "Demo Mode" badge when running without a trained model
- Full error handling with user-friendly messages

---

## Diseases Covered

| Key          | Display Name              | Severity            |
|--------------|---------------------------|---------------------|
| `acne`       | Acne Vulgaris             | Mild to Moderate    |
| `eczema`     | Eczema (Atopic Dermatitis)| Moderate to Severe  |
| `psoriasis`  | Psoriasis                 | Moderate to Severe  |
| `nail_fungus`| Nail Fungus (Onychomycosis)| Mild to Moderate   |

All disease info (duration, precautions, care tips) is stored in
`disease_info.json` and can be edited without touching Python code.

---

## How the AI Works

```
Input Image (any size)
        ↓
   PIL resize → 224×224 RGB
        ↓
   Normalise pixels to [0, 1]
        ↓
   MobileNetV2 (frozen base)
   — 154 convolutional layers
   — Trained on 1.2M ImageNet images
        ↓
   Global Average Pooling
        ↓
   Dense(256, ReLU) → Dropout(0.4)
   Dense(128, ReLU) → Dropout(0.3)
        ↓
   Dense(4, Softmax)
        ↓
   [acne, eczema, nail_fungus, psoriasis] probabilities
        ↓
   argmax → predicted class + confidence
```

### Data Augmentation (training only)

| Technique           | Value         | Purpose                        |
|---------------------|---------------|--------------------------------|
| Rotation            | ±30°          | Handle tilted photos           |
| Width/height shift  | ±20%          | Handle off-centre subjects     |
| Shear               | ±15%          | Handle perspective distortion  |
| Zoom                | ±20%          | Handle distance variation      |
| Horizontal flip     | Yes           | Double effective dataset size  |
| Brightness          | [0.7, 1.3]    | Handle lighting conditions     |

---

## Project Architecture

```
Browser
   │  HTTP POST /predict (multipart image)
   ▼
Flask (app.py)
   ├── Validate file type & size
   ├── PIL open & resize to 224×224
   ├── Normalise to [0,1]
   ├── model.predict(input_tensor)    ← TensorFlow/Keras
   ├── argmax → predicted class
   ├── Lookup disease_info.json
   └── Return JSON response
          │
          ▼
Browser renders results card
```

---

## Extending the Project

### Add a new disease class
1. Add images to `dataset/train/<new_class>/` and `dataset/val/<new_class>/`
2. Add the class key to `MODEL_CLASSES` in `train_model.py` and `app.py`
3. Add disease info to `disease_info.json`
4. Retrain the model

### Use ResNet50 instead of MobileNetV2
Replace in `train_model.py`:
```python
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
```

### Deploy to production
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Disclaimer

> **This application is NOT a medical diagnostic tool.**
> Predictions are AI-generated and for educational/informational purposes only.
> Always consult a qualified dermatologist or healthcare professional for proper
> diagnosis and treatment.
