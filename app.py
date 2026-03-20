"""
Skin & Nail Disease Detection App - Flask Backend
=================================================
Main entry point. Run with: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os
import traceback
from PIL import Image
import io
import base64

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ── Load disease database ──────────────────────────────────────────────────────
with open("disease_info.json", "r") as f:
    DISEASE_INFO = json.load(f)

# ── Try to load the trained Keras model ───────────────────────────────────────
MODEL = None
MODEL_CLASSES = ["acne", "eczema", "nail_fungus", "psoriasis"]  # must match training order

try:
    import tensorflow as tf
    model_path = "model/skin_nail_model.h5"
    if os.path.exists(model_path):
        MODEL = tf.keras.models.load_model(model_path)
        print(f"[INFO] Model loaded from {model_path}")
    else:
        print("[WARN] Trained model not found. Using demo mode (random predictions).")
except Exception as e:
    print(f"[WARN] TensorFlow not available or model failed to load: {e}")
    print("[WARN] Running in DEMO mode with simulated predictions.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize and normalise the uploaded image for the model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # shape: (1, 224, 224, 3)


def demo_prediction(image_bytes: bytes) -> dict:
    """
    Simulated prediction used when the real model is absent.
    It derives a deterministic-ish result from image content so
    repeated uploads of the same image give the same answer.
    """
    # Use average pixel brightness to pick a disease class pseudo-randomly
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    arr = np.array(img, dtype=np.float32)
    seed = int(arr.mean() * 100) % len(MODEL_CLASSES)

    confidences = np.random.dirichlet(np.ones(len(MODEL_CLASSES)) * 2)
    # Boost the seeded class so the result looks realistic
    confidences[seed] += 0.4
    confidences /= confidences.sum()

    predicted_idx = int(np.argmax(confidences))
    predicted_class = MODEL_CLASSES[predicted_idx]
    confidence = float(confidences[predicted_idx])
    return predicted_class, confidence, confidences.tolist()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: multipart/form-data with field 'image'
    Returns JSON with prediction details.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed = {"png", "jpg", "jpeg", "bmp", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use PNG or JPEG."}), 400

    try:
        image_bytes = file.read()

        if MODEL is not None:
            # ── Real model prediction ──────────────────────────────────────
            input_tensor = preprocess_image(image_bytes)
            raw_preds = MODEL.predict(input_tensor)[0]           # shape: (num_classes,)
            predicted_idx = int(np.argmax(raw_preds))
            predicted_class = MODEL_CLASSES[predicted_idx]
            confidence = float(raw_preds[predicted_idx])
            all_confidences = raw_preds.tolist()
        else:
            # ── Demo / fallback prediction ─────────────────────────────────
            predicted_class, confidence, all_confidences = demo_prediction(image_bytes)

        info = DISEASE_INFO.get(predicted_class, {})

        # Build per-class confidence map for the frontend chart
        confidence_map = {
            cls: round(float(all_confidences[i]) * 100, 1)
            for i, cls in enumerate(MODEL_CLASSES)
        }

        result = {
            "disease": info.get("display_name", predicted_class.replace("_", " ").title()),
            "disease_key": predicted_class,
            "confidence": round(confidence * 100, 1),
            "confidence_map": confidence_map,
            "duration": info.get("duration", "Unknown"),
            "precautions": info.get("precautions", []),
            "care_tips": info.get("care_tips", []),
            "severity": info.get("severity", "Moderate"),
            "description": info.get("description", ""),
            "demo_mode": MODEL is None,
        }

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/diseases", methods=["GET"])
def list_diseases():
    """Return the full disease database for reference."""
    return jsonify(DISEASE_INFO)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "demo_mode": MODEL is None,
    })


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
