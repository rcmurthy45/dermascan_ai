"""
Skin & Nail Disease Detection — Model Training Script
======================================================
Run this script ONCE to train and save the model.

    python train_model.py

Prerequisites
-------------
pip install tensorflow pillow numpy scikit-learn matplotlib

Dataset structure expected
--------------------------
dataset/
    train/
        acne/          ← training images
        eczema/
        nail_fungus/
        psoriasis/
    val/
        acne/          ← validation images
        eczema/
        nail_fungus/
        psoriasis/

If you don't have images yet, use the --demo flag to train on tiny synthetic
data just to verify the pipeline works:

    python train_model.py --demo
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # MobileNetV2 expects 224×224
BATCH_SIZE  = 32
EPOCHS      = 20           # increase for better accuracy
FINE_TUNE_EPOCHS = 10      # additional epochs after unfreezing top layers
LEARNING_RATE   = 1e-4
CLASSES     = ["acne", "eczema", "nail_fungus", "psoriasis"]
NUM_CLASSES = len(CLASSES)
MODEL_SAVE_PATH = "model/skin_nail_model.h5"

os.makedirs("model", exist_ok=True)


# ── 1. Data Augmentation ───────────────────────────────────────────────────────
def get_data_generators(train_dir: str, val_dir: str):
    """
    Returns (train_gen, val_gen) ImageDataGenerators with augmentation.
    Augmentation techniques used:
      • Rotation      — handles tilted/rotated photos
      • Width/height shift — handles off-centre subjects
      • Shear         — handles perspective distortion
      • Zoom          — handles close-up vs distant shots
      • Horizontal flip — doubles effective dataset size
      • Brightness    — handles varying lighting conditions
      • Fill mode     — nearest-neighbour fill for empty corners
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,          # normalise pixel values to [0, 1]
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,          # NO augmentation on validation data
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, val_gen


# ── 2. Build the Model (Transfer Learning) ────────────────────────────────────
def build_model() -> keras.Model:
    """
    Two-stage architecture:
      Stage 1 — Feature extraction: MobileNetV2 base (frozen) + custom head
      Stage 2 — Fine-tuning:        unfreeze top 30 base layers, lower LR
    """
    # Load MobileNetV2 pre-trained on ImageNet; remove top classification head
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # freeze all base layers during stage 1

    # Custom classification head
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)                 # regularisation
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model, base_model


# ── 3. Callbacks ──────────────────────────────────────────────────────────────
def get_callbacks():
    return [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ── 4. Plot Training History ──────────────────────────────────────────────────
def plot_history(history, fine_history=None, save_path="model/training_history.png"):
    acc  = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if fine_history:
        acc      += fine_history.history["accuracy"]
        val_acc  += fine_history.history["val_accuracy"]
        loss     += fine_history.history["loss"]
        val_loss += fine_history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc,     label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss,     label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Training plot saved to {save_path}")
    plt.show()


# ── 5. Demo Mode: generate tiny synthetic dataset ─────────────────────────────
def create_demo_dataset():
    """Create 20 random-colour images per class so training code can run."""
    import shutil
    for split in ("train", "val"):
        for cls in CLASSES:
            path = os.path.join("dataset", split, cls)
            os.makedirs(path, exist_ok=True)
            n = 20 if split == "train" else 6
            for i in range(n):
                from PIL import Image as PILImage
                colour = np.random.randint(0, 256, 3).tolist()
                img = PILImage.new("RGB", (224, 224), tuple(colour))
                img.save(os.path.join(path, f"{i}.jpg"))
    print("[DEMO] Synthetic dataset created at ./dataset/")


# ── 6. Main Training Pipeline ─────────────────────────────────────────────────
def main(demo=False):
    if demo:
        print("[DEMO] Generating synthetic dataset …")
        create_demo_dataset()

    train_dir = "dataset/train"
    val_dir   = "dataset/val"

    if not os.path.exists(train_dir):
        print("[ERROR] dataset/train not found. Use --demo or add real images.")
        sys.exit(1)

    # ── Stage 1: Feature Extraction ──
    print("\n========== STAGE 1: Feature Extraction ==========")
    train_gen, val_gen = get_data_generators(train_dir, val_dir)
    model, base_model  = build_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
    )

    # ── Stage 2: Fine-Tuning ──
    print("\n========== STAGE 2: Fine-Tuning Top Layers ==========")
    # Unfreeze the top 30 layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(),
    )

    # ── Evaluate ──
    print("\n========== Evaluation ==========")
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"Final Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"Final Validation Loss     : {val_loss:.4f}")

    # ── Save & Plot ──
    model.save(MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
    plot_history(history, fine_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the skin/nail disease model.")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data (no real dataset needed)")
    args = parser.parse_args()
    main(demo=args.demo)
