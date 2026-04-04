#!/usr/bin/env python3
"""
Canid Data Pipeline
====================
Downloads ESC-50 and AudioSet dog vocalization clips, extracts audio features
matching the browser's Web Audio API implementation, handles class imbalance,
and outputs labeled numpy arrays ready for CNN training.

Feature extraction is carefully calibrated to match canid-cnn-classifier.js.

Usage:
    python data-pipeline.py [--output-dir dataset/] [--sr 22050] [--no-download]
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants — MUST match canid-cnn-classifier.js
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050          # Hz — matches Web Audio API default
HOP_LENGTH = 512             # ~23ms hop
N_FFT = 2048                 # FFT window
N_MFCC = 13                  # Coefficients — matches app
FRAME_DURATION = 1.0         # Seconds per analysis window
N_FRAMES = int(SAMPLE_RATE * FRAME_DURATION / HOP_LENGTH)  # ~43 frames

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

# ESC-50 category ID for dogs
ESC50_DOG_CATEGORY = 5

# AudioSet ontology IDs for dog vocalizations
AUDIOSET_LABELS = {
    "bark":    "/m/05tny_",
    "growl":   "/m/07qf0zm",
    "howl":    "/m/07qrkrw",
    "whimper": "/m/07qn5dc",
}

# Rough mapping from raw audio source label → Canid class
# ESC-50 dog clips are mostly barks; we use heuristics at feature time
SOURCE_TO_CLASS = {
    "bark":    ["play_bark", "warning_bark", "alert_bark", "demand_bark"],  # further split by features
    "growl":   ["growl"],
    "howl":    ["howl"],
    "whimper": ["whine"],
}

# Target samples per class for balanced training
TARGET_SAMPLES_PER_CLASS = 300

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Extract a fixed-size feature vector from a 1-second audio segment.

    Feature layout (matches JS):
      [0:13]  — mean MFCCs (13 coefficients)
      [13:26] — std MFCCs
      [26]    — fundamental frequency (Hz, 0 if unvoiced), normalized
      [27]    — spectral centroid (Hz), normalized
      [28]    — spectral flatness
      [29]    — RMS energy

    Returns shape (30,) float32, or None if audio is too short.
    """
    if len(y) < N_FFT:
        return None

    # Resample if needed
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Ensure exactly FRAME_DURATION seconds (pad or trim)
    target_len = int(SAMPLE_RATE * FRAME_DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # MFCCs — 13 coefficients, mean + std over time frames
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)   # (13,)
    mfcc_std  = np.std(mfcc, axis=1)    # (13,)

    # Fundamental frequency (pitch) via pyin
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=4000,
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    # Mean pitch over voiced frames only; 0 if unvoiced
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
    pitch_hz = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_norm = pitch_hz / 4000.0  # normalize to [0,1]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    centroid_mean = float(np.mean(centroid)) / (SAMPLE_RATE / 2.0)  # normalize

    # Spectral flatness (Wiener entropy) — already in [0,1]
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    flatness_mean = float(np.mean(flatness))

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    rms_mean = float(np.mean(rms))

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [pitch_norm, centroid_mean, flatness_mean, rms_mean],
    ]).astype(np.float32)

    assert features.shape == (30,), f"Expected 30 features, got {features.shape}"
    return features


def classify_bark_heuristic(features: np.ndarray) -> str:
    """
    Heuristically sub-classify a bark clip into one of the four bark subtypes
    using extracted features. This is intentionally simple — the CNN will learn
    better boundaries from user feedback data over time.

      play_bark   — high pitch, high energy, low spectral flatness
      warning_bark — mid pitch, sustained energy
      alert_bark  — short, sharp, high centroid
      demand_bark — repetitive pattern, mid-high energy
    """
    pitch    = features[26] * 4000   # denormalize Hz
    centroid = features[27]          # normalized
    flatness = features[28]
    rms      = features[29]
    mfcc_mean_energy = np.mean(np.abs(features[:13]))

    if pitch > 1200 and rms > 0.05:
        return "play_bark"
    elif centroid > 0.5 and rms > 0.04:
        return "alert_bark"
    elif rms > 0.06 and flatness < 0.1:
        return "warning_bark"
    else:
        return "demand_bark"


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_audio(y: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Generate augmented variants of an audio clip.
    Returns list of (potentially) augmented clips.
    """
    augmented = [y]  # always include original

    # Pitch shift ±2 semitones
    for n_steps in [-2, -1, 1, 2]:
        try:
            shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            augmented.append(shifted)
        except Exception:
            pass

    # Time stretch 0.85× and 1.15×
    for rate in [0.85, 1.15]:
        try:
            stretched = librosa.effects.time_stretch(y, rate=rate)
            augmented.append(stretched)
        except Exception:
            pass

    # Add Gaussian noise
    for noise_factor in [0.003, 0.008]:
        noise = np.random.randn(len(y)) * noise_factor
        augmented.append(y + noise)

    return augmented


def oversample_class(
    features_list: list[np.ndarray],
    labels: list[str],
    target_class: str,
    target_count: int,
    audio_cache: Optional[list] = None,
) -> tuple[list, list]:
    """
    Oversample a class to reach target_count by duplicating + augmenting.
    If audio_cache is provided (list of (y, sr) tuples), augments from audio.
    Falls back to feature-level jitter if no audio available.
    """
    class_indices = [i for i, l in enumerate(labels) if l == target_class]
    current_count = len(class_indices)

    if current_count == 0:
        return features_list, labels

    new_features = list(features_list)
    new_labels = list(labels)

    needed = target_count - current_count
    if needed <= 0:
        return new_features, new_labels

    print(f"  Oversampling {target_class}: {current_count} → {target_count}")

    while needed > 0:
        idx = random.choice(class_indices)
        base_features = features_list[idx]

        if audio_cache and idx < len(audio_cache):
            y, sr = audio_cache[idx]
            aug_clips = augment_audio(y, sr)
            for aug_y in aug_clips:
                if needed <= 0:
                    break
                feat = extract_features(aug_y, sr)
                if feat is not None:
                    new_features.append(feat)
                    new_labels.append(target_class)
                    needed -= 1
        else:
            # Feature-level jitter: add small noise to existing features
            jitter = np.random.randn(*base_features.shape) * 0.02
            new_features.append((base_features + jitter).astype(np.float32))
            new_labels.append(target_class)
            needed -= 1

    return new_features, new_labels


# ---------------------------------------------------------------------------
# Dataset Downloaders
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=desc
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        return True
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        return False


def load_esc50(data_dir: Path) -> tuple[list, list, list]:
    """
    Download and load ESC-50 dataset, extracting dog vocalization clips.

    Returns (features, labels, audio_cache) where audio_cache is (y, sr) tuples.
    """
    esc50_dir = data_dir / "esc50"
    esc50_dir.mkdir(parents=True, exist_ok=True)

    zip_path = esc50_dir / "ESC-50-master.zip"
    extract_dir = esc50_dir / "ESC-50-master"

    if not extract_dir.exists():
        print("\n[ESC-50] Downloading dataset...")
        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        if not download_file(url, zip_path, "ESC-50"):
            print("[ESC-50] Download failed — skipping ESC-50")
            return [], [], []

        print("[ESC-50] Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(esc50_dir)
        zip_path.unlink()

    # Load metadata
    meta_path = extract_dir / "meta" / "esc50.csv"
    if not meta_path.exists():
        print("[ESC-50] Metadata not found — skipping")
        return [], [], []

    features_list = []
    labels_list = []
    audio_cache = []

    import csv
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if int(r["target"]) == ESC50_DOG_CATEGORY]

    print(f"[ESC-50] Found {len(rows)} dog clips")

    audio_dir = extract_dir / "audio"
    for row in tqdm(rows, desc="ESC-50 dog clips"):
        audio_path = audio_dir / row["filename"]
        if not audio_path.exists():
            continue
        try:
            y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            # ESC-50 clips are 5s — split into 1s windows with 50% overlap
            step = int(SAMPLE_RATE * 0.5)
            window = int(SAMPLE_RATE * FRAME_DURATION)
            for start in range(0, len(y) - window, step):
                segment = y[start:start + window]
                feat = extract_features(segment, SAMPLE_RATE)
                if feat is None:
                    continue
                # Sub-classify the bark
                label = classify_bark_heuristic(feat)
                features_list.append(feat)
                labels_list.append(label)
                audio_cache.append((segment, SAMPLE_RATE))
        except Exception as e:
            print(f"  [WARN] Failed to process {row['filename']}: {e}")

    print(f"[ESC-50] Extracted {len(features_list)} feature vectors")
    return features_list, labels_list, audio_cache


def load_audioset_balanced(data_dir: Path) -> tuple[list, list, list]:
    """
    Download balanced AudioSet clips for dog vocalization labels.

    AudioSet provides pre-extracted 128-dimensional embeddings via TFRecord files,
    but for our purposes we need raw audio. We use the balanced train set CSV
    to identify relevant YouTube IDs, then download with yt-dlp.

    Note: AudioSet download is slow and requires yt-dlp + ffmpeg.
    We download up to MAX_PER_CLASS clips per label.
    """
    MAX_PER_CLASS = 50  # cap to keep download time reasonable

    audioset_dir = data_dir / "audioset"
    audioset_dir.mkdir(parents=True, exist_ok=True)

    # Download balanced train set labels CSV
    csv_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
    csv_path = audioset_dir / "balanced_train_segments.csv"

    if not csv_path.exists():
        print("\n[AudioSet] Downloading balanced train CSV...")
        if not download_file(csv_url, csv_path, "AudioSet CSV"):
            print("[AudioSet] CSV download failed — skipping AudioSet")
            return [], [], []

    # Parse CSV and find dog vocalization entries
    target_ids = set(AUDIOSET_LABELS.values())
    clips_by_label: dict[str, list] = {k: [] for k in AUDIOSET_LABELS}

    print("[AudioSet] Parsing CSV for dog vocalization clips...")
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(", ")
            if len(parts) < 4:
                continue
            ytid, start_s, end_s = parts[0], float(parts[1]), float(parts[2])
            labels_str = ", ".join(parts[3:])
            for label_name, label_id in AUDIOSET_LABELS.items():
                if label_id in labels_str:
                    clips_by_label[label_name].append((ytid, start_s, end_s))

    features_list = []
    labels_list = []
    audio_cache = []

    try:
        import yt_dlp  # noqa
        yt_dlp_available = True
    except ImportError:
        print("[AudioSet] yt-dlp not available — skipping YouTube download")
        yt_dlp_available = False

    if not yt_dlp_available:
        return [], [], []

    import yt_dlp

    for label_name, clips in clips_by_label.items():
        canid_classes = SOURCE_TO_CLASS.get(label_name, ["silence"])
        clips = clips[:MAX_PER_CLASS]
        print(f"[AudioSet] Downloading {len(clips)} clips for '{label_name}'...")

        for ytid, start_s, end_s in tqdm(clips, desc=f"AudioSet {label_name}"):
            url = f"https://www.youtube.com/watch?v={ytid}"
            tmp_path = audioset_dir / f"tmp_{ytid}_{int(start_s)}.%(ext)s"

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(tmp_path),
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }],
                "download_ranges": yt_dlp.utils.download_range_func(
                    None, [(start_s, end_s)]
                ),
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                wav_path = audioset_dir / f"tmp_{ytid}_{int(start_s)}.wav"
                if not wav_path.exists():
                    continue

                y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
                wav_path.unlink()

                # Extract 1s windows
                window = int(SAMPLE_RATE * FRAME_DURATION)
                for start in range(0, max(1, len(y) - window), int(window * 0.5)):
                    segment = y[start:start + window]
                    feat = extract_features(segment, SAMPLE_RATE)
                    if feat is None:
                        continue

                    # Map to Canid class
                    if label_name == "bark":
                        canid_class = classify_bark_heuristic(feat)
                    else:
                        canid_class = canid_classes[0]

                    features_list.append(feat)
                    labels_list.append(canid_class)
                    audio_cache.append((segment, SAMPLE_RATE))

            except Exception as e:
                # Many clips will fail (age-gated, deleted, etc.) — that's fine
                pass

    print(f"[AudioSet] Extracted {len(features_list)} feature vectors")
    return features_list, labels_list, audio_cache


def generate_silence_samples(n: int) -> tuple[list, list, list]:
    """Generate synthetic silence samples (near-zero RMS)."""
    features_list = []
    labels_list = []
    audio_cache = []

    for _ in range(n):
        # Very low amplitude noise
        noise_level = np.random.uniform(1e-5, 1e-3)
        y = np.random.randn(SAMPLE_RATE) * noise_level
        feat = extract_features(y, SAMPLE_RATE)
        if feat is not None:
            features_list.append(feat)
            labels_list.append("silence")
            audio_cache.append((y, SAMPLE_RATE))

    return features_list, labels_list, audio_cache


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(output_dir: Path, raw_data_dir: Path, no_download: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    all_audio: list = []

    # 1. ESC-50
    if not no_download:
        f, l, a = load_esc50(raw_data_dir)
        all_features.extend(f)
        all_labels.extend(l)
        all_audio.extend(a)

    # 2. AudioSet (optional — requires yt-dlp and time)
    if not no_download:
        f, l, a = load_audioset_balanced(raw_data_dir)
        all_features.extend(f)
        all_labels.extend(l)
        all_audio.extend(a)

    # 3. Generate silence samples
    silence_count = max(20, TARGET_SAMPLES_PER_CLASS // 4)
    f, l, a = generate_silence_samples(silence_count)
    all_features.extend(f)
    all_labels.extend(l)
    all_audio.extend(a)

    # 4. Report class distribution
    print("\n[Pipeline] Class distribution (before oversampling):")
    label_counts: dict[str, int] = {}
    for lbl in CLASS_LABELS:
        count = all_labels.count(lbl)
        label_counts[lbl] = count
        bar = "█" * min(count, 50)
        print(f"  {lbl:15s} {count:4d}  {bar}")

    # Warn about missing classes
    missing = [lbl for lbl in CLASS_LABELS if label_counts.get(lbl, 0) == 0]
    if missing:
        print(f"\n  [WARN] Classes with 0 samples: {missing}")
        print("  Add synthetic or placeholder features to avoid training failures.")
        # Add minimal placeholder samples for missing classes
        for lbl in missing:
            placeholder = np.zeros(30, dtype=np.float32)
            placeholder[29] = 0.01  # minimal RMS
            all_features.append(placeholder)
            all_labels.append(lbl)
            print(f"  Added 1 placeholder for '{lbl}' — replace with real data!")

    # 5. Oversample to balance
    print("\n[Pipeline] Oversampling to balance classes...")
    for lbl in CLASS_LABELS:
        all_features, all_labels = oversample_class(
            all_features, all_labels, lbl,
            TARGET_SAMPLES_PER_CLASS, all_audio
        )

    # 6. Shuffle
    combined = list(zip(all_features, all_labels))
    random.shuffle(combined)
    all_features, all_labels = zip(*combined) if combined else ([], [])

    # 7. Convert to numpy
    X = np.array(all_features, dtype=np.float32)
    y_int = np.array([CLASS_LABELS.index(l) for l in all_labels], dtype=np.int32)

    # 8. Normalize features (per-feature mean/std)
    feature_mean = X.mean(axis=0)
    feature_std  = X.std(axis=0) + 1e-8  # avoid division by zero
    X_norm = (X - feature_mean) / feature_std

    # 9. Train/val split (80/20)
    n = len(X_norm)
    n_val = int(n * 0.2)
    X_train, X_val = X_norm[n_val:], X_norm[:n_val]
    y_train, y_val = y_int[n_val:], y_int[:n_val]

    # 10. Save
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy",   X_val)
    np.save(output_dir / "y_val.npy",   y_val)
    np.save(output_dir / "feature_mean.npy", feature_mean)
    np.save(output_dir / "feature_std.npy",  feature_std)

    # Save normalization stats as JSON (needed by JS module)
    norm_stats = {
        "feature_mean": feature_mean.tolist(),
        "feature_std":  feature_std.tolist(),
        "n_features":   int(X.shape[1]),
        "sample_rate":  SAMPLE_RATE,
        "n_mfcc":       N_MFCC,
        "n_fft":        N_FFT,
        "hop_length":   HOP_LENGTH,
        "frame_duration_s": FRAME_DURATION,
    }
    with open(output_dir / "normalization.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n[Pipeline] ✓ Dataset saved to {output_dir}/")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Features:   {X.shape[1]} per sample")
    print("\n[Pipeline] Final class distribution:")
    for lbl in CLASS_LABELS:
        idx = CLASS_LABELS.index(lbl)
        count = int((y_int == idx).sum())
        print(f"  {lbl:15s} {count:4d}")


def main():
    parser = argparse.ArgumentParser(description="Canid audio data pipeline")
    parser.add_argument("--output-dir",  default="dataset",     help="Where to save processed data")
    parser.add_argument("--raw-dir",     default="dataset/raw", help="Where to cache raw downloads")
    parser.add_argument("--no-download", action="store_true",   help="Skip downloading (use cached raw data)")
    parser.add_argument("--seed",        type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_pipeline(
        output_dir=Path(args.output_dir),
        raw_data_dir=Path(args.raw_dir),
        no_download=args.no_download,
    )


if __name__ == "__main__":
    main()
