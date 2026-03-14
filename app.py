"""
Face Mask Detection System — Flask Web Application
===================================================
Serves the frontend and provides prediction API.
Uses the trained Keras CNN (mask_detector.h5).

Model path : D:/python/Facemaskdetect/files/mask_detector.h5
CSV log    : D:/python/Facemaskdetect/files/results/predictions_log.csv

Run:
  python app.py
  Open: http://localhost:5000
"""

import os
import io
import cv2
import csv
import base64
import numpy as np
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.models import load_model as keras_load

# ─── HARDCODED PATHS ──────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\python\Facemaskdetect\files\mask_detector.h5"
CSV_DIR     = r"D:\python\Facemaskdetect\files\results"
CSV_LOG     = os.path.join(CSV_DIR, "predictions_log.csv")

# ─── CONFIG (must match train.py) ─────────────────────────────────────────────
IMG_SIZE    = 100
CLASSES     = ["with_mask", "without_mask"]
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}

# ─── FLASK ────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16MB max upload
os.makedirs(CSV_DIR, exist_ok=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found: {MODEL_PATH}")
        print("[WARN] Train first: python train.py")
        return None
    print(f"[INFO] Loading model: {MODEL_PATH}")
    _model = keras_load(MODEL_PATH)
    print(f"[INFO] Model ready. Input shape: {_model.input_shape}")
    return _model

# ─── PREPROCESSING (same as train.py) ────────────────────────────────────────
def preprocess_pil(pil_image):
    """
    Preprocess a PIL image exactly as done in training:
      1. Resize to 100x100
      2. Convert to RGB numpy array
      3. Normalize [0,255] → [0.0, 1.0]
      4. Add batch dimension
    """
    img = pil_image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)    # (1, 100, 100, 3)
    return arr

def run_prediction(pil_image):
    """Returns (label, raw_sigmoid, prob_with, prob_without)"""
    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded. Run python train.py first.")

    tensor       = preprocess_pil(pil_image)
    raw_prob     = float(model.predict(tensor, verbose=0)[0][0])
    label        = CLASSES[int(raw_prob >= 0.5)]
    prob_without = raw_prob
    prob_with    = 1.0 - raw_prob
    return label, raw_prob, prob_with, prob_without

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def log_prediction(filename, label, prob_with, prob_without):
    """Append one prediction row to the CSV log."""
    fields     = ["filename", "prediction",
                  "prob_with_mask_pct", "prob_without_mask_pct", "timestamp"]
    new_file   = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "filename"              : filename,
            "prediction"            : label,
            "prob_with_mask_pct"    : round(prob_with    * 100, 2),
            "prob_without_mask_pct" : round(prob_without * 100, 2),
            "timestamp"             : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"),
              encoding="utf-8") as f:
        return f.read()

@app.route("/api/status")
def status():
    model = get_model()
    return jsonify({
        "model_loaded" : model is not None,
        "model_path"   : MODEL_PATH,
        "input_shape"  : str(model.input_shape) if model else None,
        "img_size"     : IMG_SIZE,
        "classes"      : CLASSES,
        "status"       : "ready" if model else "no_model",
    })

@app.route("/api/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not allowed(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, BMP, WEBP."}), 400

    try:
        pil_img = Image.open(file.stream).convert("RGB")
        label, raw_prob, prob_with, prob_without = run_prediction(pil_img)

        # Encode thumbnail as base64 for UI preview
        buf   = io.BytesIO()
        thumb = pil_img.copy()
        thumb.thumbnail((400, 400))
        thumb.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Log to CSV
        log_prediction(file.filename, label, prob_with, prob_without)

        return jsonify({
            "filename"               : file.filename,
            "prediction"             : label,
            "label_display"          : "✅ With Mask" if label == "with_mask" else "❌ Without Mask",
            "confidence"             : round(max(prob_with, prob_without) * 100, 2),
            "raw_sigmoid"            : round(raw_prob, 6),
            "confidence_with_mask"   : round(prob_with    * 100, 2),
            "confidence_without_mask": round(prob_without * 100, 2),
            "image_preview"          : f"data:image/jpeg;base64,{img_b64}",
            "timestamp"              : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route("/api/download_csv")
def download_csv():
    if not os.path.exists(CSV_LOG):
        return jsonify({"error": "No predictions yet. Upload an image first."}), 404
    return send_file(CSV_LOG, mimetype="text/csv",
                     as_attachment=True, download_name="mask_predictions.csv")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Face Mask Detection — Flask Web App")
    print("  http://localhost:5000")
    print("=" * 55)
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
