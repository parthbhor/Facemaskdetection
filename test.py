"""
Face Mask Detection System — Testing Script
============================================
Single image inference using the trained CNN model.
Applies the SAME preprocessing as training:
  - Resize to 100x100
  - Normalize to [0, 1]
  - Sigmoid threshold: >= 0.5 → without_mask, < 0.5 → with_mask

Hardcoded model: D:/python/Facemaskdetect/files/mask_detector.h5

Usage:
  python test.py --input "D:/path/to/image.jpg"
  python test.py --input photo.jpg --output my_results.csv
"""

import os
import cv2
import csv
import argparse
import numpy as np
from datetime import datetime

from tensorflow.keras.models import load_model

# ─── HARDCODED PATHS ──────────────────────────────────────────────────────────
DEFAULT_MODEL  = r"D:\python\Facemaskdetect\files\mask_detector.h5"
DEFAULT_OUTPUT = r"D:\python\Facemaskdetect\files\results.csv"

# ─── ARGS ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",  required=True,          help="Path to input image")
ap.add_argument("-m", "--model",  default=DEFAULT_MODEL,  help="Path to trained .h5 model")
ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output CSV file path")
args = vars(ap.parse_args())

# ─── CONFIG (must match train.py exactly) ────────────────────────────────────
IMG_SIZE = 100
CLASSES  = ["with_mask", "without_mask"]

print("=" * 55)
print("  Face Mask Detection — Single Image Test")
print("=" * 55)
print(f"  Model  : {args['model']}")
print(f"  Input  : {args['input']}")
print("=" * 55)

# ─── VALIDATE ─────────────────────────────────────────────────────────────────
if not os.path.exists(args["model"]):
    raise FileNotFoundError(
        f"\n[ERROR] Model not found: {args['model']}\n"
        "Train first: python train.py"
    )
if not os.path.exists(args["input"]):
    raise FileNotFoundError(f"\n[ERROR] Image not found: {args['input']}")

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
print("\n[INFO] Loading model...")
model = load_model(args["model"])
print("[INFO] Model loaded.")
print(f"[INFO] Input shape expected: {model.input_shape}")

# ─── PREPROCESS IMAGE (same as training) ─────────────────────────────────────
def preprocess(img_path):
    """
    Applies the exact same preprocessing used during training:
      1. Read with OpenCV
      2. Resize to 100x100
      3. Convert BGR → RGB
      4. Normalize [0, 255] → [0.0, 1.0]
      5. Expand dims for batch: (1, 100, 100, 3)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # shape: (1, 100, 100, 3)
    return img

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict(img_path):
    """
    Returns:
      label        : 'with_mask' or 'without_mask'
      probability  : raw sigmoid output (0.0 – 1.0)
      prob_with    : probability of with_mask
      prob_without : probability of without_mask
    """
    tensor = preprocess(img_path)
    prob   = float(model.predict(tensor, verbose=0)[0][0])   # sigmoid output

    # sigmoid >= 0.5 → class index 1 → 'without_mask'
    # sigmoid  < 0.5 → class index 0 → 'with_mask'
    label        = CLASSES[int(prob >= 0.5)]
    prob_without = prob
    prob_with    = 1.0 - prob

    return label, prob, prob_with, prob_without

# ─── RUN ──────────────────────────────────────────────────────────────────────
try:
    label, raw_prob, prob_with, prob_without = predict(args["input"])
    filename  = os.path.basename(args["input"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status    = "OK"
    confidence = max(prob_with, prob_without)
except Exception as e:
    label, raw_prob, prob_with, prob_without = "ERROR", 0.0, 0.0, 0.0
    filename  = os.path.basename(args["input"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status    = str(e)
    confidence = 0.0
    print(f"\n[ERROR] {e}")

# ─── DISPLAY RESULT ───────────────────────────────────────────────────────────
icon    = "✅" if label == "with_mask" else "❌"
bar_w   = int(confidence * 30)
bar_e   = 30 - bar_w
conf_pct = confidence * 100

print("\n┌─────────────────────────────────────────────────────┐")
print(f"│  File        : {filename[:48]:<48}│")
print(f"│  Prediction  : {icon}  {label:<47}│")
print(f"│  Confidence  : {'█' * bar_w}{'░' * bar_e}  {conf_pct:5.1f}%   │")
print(f"│  Raw sigmoid : {raw_prob:.6f}                                  │")
print("├─────────────────────────────────────────────────────┤")
print(f"│  ✅ With Mask    prob : {prob_with*100:6.2f}%                      │")
print(f"│  ❌ Without Mask prob : {prob_without*100:6.2f}%                      │")
print("└─────────────────────────────────────────────────────┘")

# ─── SAVE TO CSV ──────────────────────────────────────────────────────────────
csv_path = args["output"]
os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

file_exists = os.path.exists(csv_path)
fieldnames  = [
    "filename", "filepath", "prediction",
    "confidence_pct", "raw_sigmoid_output",
    "prob_with_mask_pct", "prob_without_mask_pct",
    "status", "model_used", "timestamp"
]

with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow({
        "filename"               : filename,
        "filepath"               : args["input"],
        "prediction"             : label,
        "confidence_pct"         : round(confidence     * 100, 2),
        "raw_sigmoid_output"     : round(raw_prob,              6),
        "prob_with_mask_pct"     : round(prob_with       * 100, 2),
        "prob_without_mask_pct"  : round(prob_without    * 100, 2),
        "status"                 : status,
        "model_used"             : args["model"],
        "timestamp"              : timestamp,
    })

print(f"\n[INFO] Result saved → {csv_path}")
print("[INFO] Done!")
