"""
Face Mask Detection System — Training Script
=============================================
Custom CNN Binary Image Classifier
Trains on dataset built from:
  - Scraped images (via scrape_data.py)
  - Kaggle dataset (or any manually placed images)
  - Or a mix of both

Spec:
  - Input     : 100x100 RGB images
  - Normalize : [0, 1]
  - Split     : 80% train / 20% test
  - Conv+ReLU → MaxPool (x4 blocks)
  - Flatten → Dense → Dropout → Dense
  - Output    : sigmoid (binary classification)
  - Loss      : binary_crossentropy
  - Optimizer : Adam
  - Saved as  : mask_detector.h5

Dataset path (hardcoded):
  D:/python/Facemaskdetect/files/dataset/
    ├── with_mask/
    └── without_mask/

Usage:
  python train.py
  python train.py --epochs 30 --batch 64
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── HARDCODED PATHS ──────────────────────────────────────────────────────────
DATASET_PATH = r"D:\python\Facemaskdetect\files\dataset"
MODEL_PATH   = r"D:\python\Facemaskdetect\files\mask_detector.h5"
PLOT_PATH    = r"D:\python\Facemaskdetect\files\training_plot.png"
CM_PATH      = r"D:\python\Facemaskdetect\files\confusion_matrix.png"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--epochs",  type=int,   default=25)
ap.add_argument("--batch",   type=int,   default=32)
ap.add_argument("--lr",      type=float, default=0.001)
ap.add_argument("--imgsize", type=int,   default=100)
args = vars(ap.parse_args())

IMG_SIZE   = args["imgsize"]
BATCH_SIZE = args["batch"]
EPOCHS     = args["epochs"]
LR         = args["lr"]
TEST_SPLIT = 0.20
CLASSES    = ["with_mask", "without_mask"]
LABELS     = {cls: idx for idx, cls in enumerate(CLASSES)}

print("=" * 62)
print("  Face Mask Detection — CNN Training")
print("=" * 62)
print(f"  Dataset   : {DATASET_PATH}")
print(f"  Model out : {MODEL_PATH}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}  |  Normalize: [0,1]")
print(f"  Epochs    : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
print("=" * 62)

# ─── VALIDATE DATASET ─────────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found: {DATASET_PATH}\n"
        "Run scrape_data.py first to build the dataset, or place your\n"
        "Kaggle images in with_mask/ and without_mask/ subfolders."
    )

print("\n[INFO] Dataset validation:")
for cls in CLASSES:
    p = os.path.join(DATASET_PATH, cls)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"[ERROR] Missing folder: {p}\n"
            "Run: python scrape_data.py"
        )
    imgs = [f for f in os.listdir(p)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    print(f"  {cls:20s} → {len(imgs):,} images  (label={LABELS[cls]})")

    if len(imgs) < 50:
        print(f"  [WARN] Very few images in {cls}. "
              "Run scrape_data.py to collect more.")
print()

# ─── LOAD & PREPROCESS ────────────────────────────────────────────────────────
print("[INFO] Loading and preprocessing images...")
data, labels = [], []
skipped      = 0

for cls in CLASSES:
    cls_path = os.path.join(DATASET_PATH, cls)
    label    = LABELS[cls]
    files    = [f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

    for fname in files:
        img_path = os.path.join(cls_path, fname)
        img      = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))       # → 100x100
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # BGR → RGB
        img = img.astype("float32") / 255.0               # normalize [0,1]
        data.append(img)
        labels.append(label)

if skipped:
    print(f"  [WARN] Skipped {skipped} unreadable files.")

data   = np.array(data,   dtype="float32")
labels = np.array(labels, dtype="float32")

print(f"[INFO] Dataset ready:")
print(f"  Total   : {len(data):,}  |  shape: {data.shape}")
print(f"  With mask    : {int(np.sum(labels==0)):,}")
print(f"  Without mask : {int(np.sum(labels==1)):,}")

# ─── TRAIN / TEST SPLIT (80/20) ───────────────────────────────────────────────
print(f"\n[INFO] Splitting: 80% train / 20% test ...")
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
)
print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# ─── DATA AUGMENTATION ────────────────────────────────────────────────────────
# Augment TRAINING data only — improves generalisation on scraped images
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(X_train)

# ─── CNN ARCHITECTURE ─────────────────────────────────────────────────────────
print("\n[INFO] Building CNN model...")

model = Sequential([

    # Block 1 — Conv + ReLU + MaxPool: 100x100 → 50x50
    Conv2D(32, (3, 3), activation="relu", padding="same",
           input_shape=(IMG_SIZE, IMG_SIZE, 3), name="conv1"),
    BatchNormalization(),
    MaxPooling2D((2, 2), name="pool1"),

    # Block 2 — Conv + ReLU + MaxPool: 50x50 → 25x25
    Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
    BatchNormalization(),
    MaxPooling2D((2, 2), name="pool2"),

    # Block 3 — Conv + ReLU + MaxPool: 25x25 → 12x12
    Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
    BatchNormalization(),
    MaxPooling2D((2, 2), name="pool3"),

    # Block 4 — Conv + ReLU + MaxPool: 12x12 → 6x6
    Conv2D(128, (3, 3), activation="relu", padding="same", name="conv4"),
    BatchNormalization(),
    MaxPooling2D((2, 2), name="pool4"),

    # Flatten
    Flatten(name="flatten"),

    # Fully Connected Layers
    Dense(256, activation="relu", name="dense1"),
    Dropout(0.5, name="drop1"),
    Dense(128, activation="relu", name="dense2"),
    Dropout(0.3, name="drop2"),

    # Output — sigmoid for binary classification
    Dense(1, activation="sigmoid", name="output"),

], name="FaceMaskCNN")

model.summary()

# ─── COMPILE ──────────────────────────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ─── CALLBACKS ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Training for up to {EPOCHS} epochs...")
H = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
print("\n[INFO] Evaluating on test set...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {acc*100:.2f}%")

y_pred = (model.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
print("\n[INFO] Classification Report:")
print(classification_report(y_test.astype(int), y_pred, target_names=CLASSES))

# ─── PLOTS ────────────────────────────────────────────────────────────────────
N = len(H.history["loss"])
e = range(1, N+1)

plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Face Mask Detection CNN — Training History", fontsize=13, fontweight="bold")

ax1.plot(e, H.history["loss"],     label="Train Loss", lw=2)
ax1.plot(e, H.history["val_loss"], label="Val Loss",   lw=2)
ax1.set_title("Binary Crossentropy Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(e, [a*100 for a in H.history["accuracy"]],     label="Train Acc", lw=2)
ax2.plot(e, [a*100 for a in H.history["val_accuracy"]], label="Val Acc",   lw=2)
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("%")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)

cm = confusion_matrix(y_test.astype(int), y_pred)
fig2, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)

print(f"\n[INFO] Model   → {MODEL_PATH}")
print(f"[INFO] Plot    → {PLOT_PATH}")
print(f"[INFO] CM      → {CM_PATH}")
print("\n[INFO] Training complete! ✓  Run: python app.py")
