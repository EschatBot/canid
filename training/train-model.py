#!/usr/bin/env python3
"""
Canid CNN Model Training
=========================
Loads the processed dataset from data-pipeline.py, trains a compact CNN
for dog vocalization classification, evaluates it, and exports to both
SavedModel and TensorFlow.js formats.

Architecture is deliberately small (<2MB) to work on mobile browsers.

Usage:
    python train-model.py [--dataset-dir dataset/] [--output-dir model/]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_LABELS = [
    "play_bark",
    "warning_bark",
    "alert_bark",
    "demand_bark",
    "whine",
    "growl",
    "howl",
    "yip",
    "silence",
]
N_CLASSES = len(CLASS_LABELS)
N_FEATURES = 30     # must match data-pipeline.py feature extraction
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 20


# ---------------------------------------------------------------------------
# Feature Reshaping
# ---------------------------------------------------------------------------

def features_to_2d(X: np.ndarray) -> np.ndarray:
    """
    Reshape flat feature vectors into 2D tensors for Conv1D processing.

    Layout: (batch, timesteps=6, channels=5)
      Row 0: MFCC means [0:5]
      Row 1: MFCC means [5:10]
      Row 2: MFCC means [10:13] + MFCC stds [0:2]
      Row 3: MFCC stds [2:7]
      Row 4: MFCC stds [7:12]
      Row 5: MFCC std[12] + pitch + centroid + flatness + energy (0-padded)

    This gives the Conv1D layers temporal structure to exploit.
    Actually simpler: we treat the 30 features as a 1D sequence of length 30
    with 1 channel, which Conv1D can process effectively.
    """
    # Shape: (batch, 30, 1) — treat each feature as a timestep
    return X.reshape(-1, N_FEATURES, 1)


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def build_model(input_shape: tuple) -> keras.Model:
    """
    Compact Conv1D model for 30-feature dog vocalization classification.

    Architecture:
      Input → Conv1D(32) → BN → ReLU → MaxPool
            → Conv1D(64) → BN → ReLU → MaxPool
            → Conv1D(64) → BN → ReLU → GlobalAvgPool
            → Dense(64)  → Dropout(0.4)
            → Dense(9, softmax)

    ~120K parameters — well under 2MB after TF.js conversion.
    """
    inp = keras.Input(shape=input_shape, name="features")

    # Block 1
    x = layers.Conv1D(32, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(0.2, name="drop2")(x)

    # Block 3
    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # Dense head
    x = layers.Dense(64, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="dense1")(x)
    x = layers.Dropout(0.4, name="drop4")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="canid_cnn")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
) -> keras.Model:

    # Reshape to (batch, 30, 1)
    X_train_2d = features_to_2d(X_train)
    X_val_2d   = features_to_2d(X_val)

    # One-hot encode
    y_train_oh = keras.utils.to_categorical(y_train, N_CLASSES)
    y_val_oh   = keras.utils.to_categorical(y_val,   N_CLASSES)

    # Build model
    model = build_model(input_shape=(N_FEATURES, 1))
    model.summary()

    # Compute class weights to handle any remaining imbalance
    class_counts = np.bincount(y_train, minlength=N_CLASSES)
    total = class_counts.sum()
    class_weight = {
        i: total / (N_CLASSES * max(c, 1))
        for i, c in enumerate(class_counts)
    }
    print("\nClass weights:", {CLASS_LABELS[i]: f"{w:.2f}" for i, w in class_weight.items()})

    # Optimizer with learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=30,
        t_mul=2.0,
        m_mul=0.9,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    ckpt_path = output_dir / "checkpoints" / "best_weights.weights.h5"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=0,
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={EARLY_STOP_PATIENCE})...")
    history = model.fit(
        X_train_2d, y_train_oh,
        validation_data=(X_val_2d, y_val_oh),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training curves
    _plot_history(history, output_dir / "training_curves.png")

    return model


def _plot_history(history, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"[Train] Saved training curves → {save_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path):
    X_val_2d = features_to_2d(X_val)
    y_pred_prob = model.predict(X_val_2d, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Per-class report
    print("\n[Eval] Per-class metrics:")
    report = classification_report(y_val, y_pred, target_names=CLASS_LABELS, digits=3)
    print(report)

    report_path = output_dir / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Canid CNN Confusion Matrix")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix → {cm_path}")

    # Overall accuracy
    accuracy = float((y_pred == y_val).mean())
    print(f"\n[Eval] Validation accuracy: {accuracy:.3f}")

    return accuracy


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_savedmodel(model: keras.Model, output_dir: Path) -> Path:
    """Export to SavedModel format."""
    saved_model_path = output_dir / "saved_model"
    model.export(str(saved_model_path))
    print(f"[Export] SavedModel → {saved_model_path}")
    return saved_model_path


def export_tfjs(saved_model_path: Path, output_dir: Path) -> Path:
    """Convert SavedModel to TF.js graph model using tensorflowjs_converter."""
    tfjs_path = output_dir / "tfjs_model"
    tfjs_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        str(saved_model_path),
        str(tfjs_path),
    ]

    print(f"[Export] Converting to TF.js format...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[WARN] tensorflowjs_converter stderr:\n{result.stderr}")
            # Try tensorflowjs_converter CLI directly
            cmd2 = [
                "tensorflowjs_converter",
                "--input_format=tf_saved_model",
                "--output_format=tfjs_graph_model",
                str(saved_model_path),
                str(tfjs_path),
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
            if result2.returncode != 0:
                print(f"[ERROR] TF.js conversion failed:\n{result2.stderr}")
                print("Run manually: tensorflowjs_converter --input_format=tf_saved_model "
                      f"--output_format=tfjs_graph_model {saved_model_path} {tfjs_path}")
                return tfjs_path
    except FileNotFoundError:
        print("[WARN] tensorflowjs_converter not found. Install with: pip install tensorflowjs")
        print(f"Manual command: tensorflowjs_converter --input_format=tf_saved_model "
              f"--output_format=tfjs_graph_model {saved_model_path} {tfjs_path}")
        return tfjs_path

    print(f"[Export] TF.js model → {tfjs_path}")

    # Report file sizes
    total_size = 0
    for p in tfjs_path.glob("**/*"):
        if p.is_file():
            size = p.stat().st_size
            total_size += size
            print(f"  {p.name}: {size / 1024:.1f} KB")
    print(f"  Total: {total_size / (1024*1024):.2f} MB")

    if total_size > 2 * 1024 * 1024:
        print("[WARN] Model exceeds 2MB target! Consider reducing architecture.")
    else:
        print(f"[Export] ✓ Model is {total_size / (1024*1024):.2f} MB — within 2MB budget")

    return tfjs_path


def export_class_labels(output_dir: Path, normalization_stats: dict):
    """Export class labels + normalization stats for the JS module."""
    labels_path = output_dir / "class_labels.json"
    data = {
        "classes": CLASS_LABELS,
        "index_to_class": {str(i): lbl for i, lbl in enumerate(CLASS_LABELS)},
        "class_to_index": {lbl: i for i, lbl in enumerate(CLASS_LABELS)},
        "normalization": normalization_stats,
        "model_info": {
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "input_shape": [N_FEATURES, 1],
        },
        "version": "1.0.0",
    }
    with open(labels_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Export] class_labels.json → {labels_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Canid CNN classifier")
    parser.add_argument("--dataset-dir", default="dataset",  help="Processed dataset directory")
    parser.add_argument("--output-dir",  default="model",    help="Where to save model artifacts")
    parser.add_argument("--epochs",      type=int,           default=EPOCHS)
    parser.add_argument("--batch-size",  type=int,           default=BATCH_SIZE)
    parser.add_argument("--seed",        type=int,           default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    print("[Train] Loading dataset...")
    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    for fname in required_files:
        if not (dataset_dir / fname).exists():
            print(f"[ERROR] Missing {dataset_dir / fname}. Run data-pipeline.py first.")
            sys.exit(1)

    X_train = np.load(dataset_dir / "X_train.npy")
    y_train = np.load(dataset_dir / "y_train.npy")
    X_val   = np.load(dataset_dir / "X_val.npy")
    y_val   = np.load(dataset_dir / "y_val.npy")

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")

    # Load normalization stats
    norm_path = dataset_dir / "normalization.json"
    norm_stats = {}
    if norm_path.exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
    else:
        print("[WARN] normalization.json not found — JS module won't normalize inputs")

    # Train
    model = train(X_train, y_train, X_val, y_val, output_dir)

    # Evaluate
    accuracy = evaluate(model, X_val, y_val, output_dir)

    # Export
    saved_model_path = export_savedmodel(model, output_dir)
    tfjs_path = export_tfjs(saved_model_path, output_dir)
    export_class_labels(output_dir, norm_stats)

    # Copy class_labels.json to tfjs dir for easy serving
    import shutil
    tfjs_labels = tfjs_path / "class_labels.json"
    shutil.copy(output_dir / "class_labels.json", tfjs_labels)

    print(f"\n{'='*60}")
    print("Canid CNN Training Complete!")
    print(f"  Validation accuracy: {accuracy:.1%}")
    print(f"  SavedModel:   {saved_model_path}/")
    print(f"  TF.js model:  {tfjs_path}/")
    print(f"  Serve the tfjs_model/ directory and update MODEL_URL in canid-cnn-classifier.js")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
