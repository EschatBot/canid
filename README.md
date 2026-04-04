# Canid 🐕

**A Dog Behavioral Translator** — real-time behavioral state analysis using your phone's camera and microphone.

**[Try it live →](https://eschatbot.github.io/canid/)**

## What it does

Canid fuses three signal channels to assess your dog's behavioral state:
- **Audio** — MFCC, pitch, spectral features classify vocalizations (bark types, growl, whine, howl, yip)
- **Video** — TensorFlow.js COCO-SSD detects dogs, infers posture and motion from bounding box geometry
- **Context** — You tell it the situation (just got home, stranger at door, etc.)

Outputs one of 8 behavioral states with confidence and evidence.

## Privacy

All processing runs on your device. No audio, video, or behavioral data is transmitted anywhere. No backend, no account, no tracking.

## Training Pipeline

The `training/` directory contains Python scripts to train a CNN vocalization classifier:
- `data-pipeline.py` — Download and prepare ESC-50 + AudioSet dog audio
- `train-model.py` — Train a small Conv1D model, export to TensorFlow.js
- `requirements.txt` — Python dependencies

## Built with

- TensorFlow.js (COCO-SSD + custom CNN)
- Web Audio API
- Pure HTML/CSS/JS — no framework, no build step

v0.3 · April 2026
