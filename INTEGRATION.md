# Canid CNN Integration Guide

This guide walks you through replacing Canid's rule-based heuristic classifier
with the trained CNN module, wiring in the feedback system, and retraining with
user-collected data.

---

## Prerequisites

- Canid web app (single-file HTML) loaded in a browser supporting Web Audio API and IndexedDB
- TensorFlow.js 4.x already included (`<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4/dist/tf.min.js">`)
- A trained model in `tfjs_model/` (see [Training a Model](#training-a-model))

---

## Step 1 — Train a Model

### 1a. Install Python dependencies

```bash
cd canid-cnn/
pip install -r requirements.txt
```

> Python 3.9+, CUDA optional (CPU training is fine for this model size).

### 1b. Run the data pipeline

```bash
python data-pipeline.py --output-dir dataset/
```

This downloads ESC-50, optionally AudioSet clips (requires `yt-dlp` + `ffmpeg`),
extracts features, balances classes via augmentation, and saves:

```
dataset/
  X_train.npy       # (N, 30) float32 feature vectors
  y_train.npy       # (N,) int32 class indices
  X_val.npy
  y_val.npy
  feature_mean.npy  # Per-feature normalization mean
  feature_std.npy   # Per-feature normalization std
  normalization.json  # Same stats as JSON (used by JS module)
```

**Skip downloading and use only generated silence samples:**
```bash
python data-pipeline.py --no-download
```

This gives you a minimal dataset to verify the pipeline works. Classification
quality will be poor without real dog audio — you'll bootstrap from user feedback.

### 1c. Train the model

```bash
python train-model.py --dataset-dir dataset/ --output-dir model/
```

Outputs:
```
model/
  saved_model/          # TF SavedModel
  tfjs_model/           # TF.js graph model (serve this)
    model.json
    group1-shard1of1.bin
    class_labels.json
  eval_report.txt
  confusion_matrix.png
  training_curves.png
```

**Expected training time:** 2–5 minutes on CPU, <1 minute on GPU.

**Expected validation accuracy:** 55–70% on ESC-50 alone (limited data).
Accuracy improves significantly with AudioSet clips and user feedback retraining.

### 1d. Serve the model

The browser needs to fetch `model.json` and the `.bin` shard(s). Serve them over HTTP:

```bash
# Simple dev server
python -m http.server 8080 --directory model/tfjs_model/

# Or copy to your web app's static directory
cp -r model/tfjs_model/ /path/to/your/app/static/tfjs_model/
```

**CORS:** The model files must be served from the same origin as your app,
or your server must send `Access-Control-Allow-Origin: *`.

---

## Step 2 — Replace the Rule-Based Classifier

### 2a. Add script tags to Canid's HTML

```html
<!-- After your existing TF.js include -->
<script type="module">
  import { CanidCNNClassifier } from './canid-cnn-classifier.js';
  import { CanidFeedback }      from './canid-feedback.js';

  window.canidCNN = new CanidCNNClassifier({
    modelUrl:   './tfjs_model/model.json',
    onProgress: (pct) => {
      document.getElementById('model-load-status').textContent =
        pct < 100 ? `Loading model ${pct}%...` : 'Model ready';
    },
    fallback:   window.existingRuleBasedClassify, // your existing function
    debug:      false,
  });

  window.canidFeedback = new CanidFeedback({
    promptDelay: 1000,  // ms before prompt appears
  });

  // Load in parallel
  Promise.all([
    window.canidCNN.load(),
    window.canidFeedback.init(),
  ]).then(([modelLoaded]) => {
    console.log('[Canid] CNN loaded:', modelLoaded);
  });
</script>
<link rel="stylesheet" href="./canid-feedback.css">
```

### 2b. Replace your classify() call

**Before (rule-based):**
```javascript
function onAudioFrame(features) {
  const result = existingRuleBasedClassify(features);
  displayResult(result);
}
```

**After (CNN with fallback):**
```javascript
async function onAudioFrame(features) {
  // CNN classifier with automatic fallback if model not loaded
  const result = window.canidCNN.classify(features);
  displayResult(result);

  // Show feedback prompt
  window.canidFeedback.show(result, features);
}
```

The `classify()` call is synchronous and safe to call before the model finishes
loading — it falls back to your rule-based classifier automatically.

### 2c. Result format

Both the CNN and the fallback return the same shape:

```javascript
{
  type:        'play_bark',   // Primary class
  subtype:     'play',        // Behavioral subtype
  description: 'Playful excitement',
  confidence:  0.87,          // 0.0-1.0
  allScores: {
    play_bark:    0.87,
    warning_bark: 0.04,
    alert_bark:   0.03,
    demand_bark:  0.02,
    whine:        0.01,
    growl:        0.01,
    howl:         0.01,
    yip:          0.01,
    silence:      0.00,
  },
  source: 'cnn'  // or 'fallback'
}
```

---

## Step 3 — Wire Up the Feedback UI

The feedback module injects its own DOM (a sliding bottom bar) and manages all
state internally. You only need to call two methods:

```javascript
// After each classification:
canidFeedback.show(result, features);

// In your Settings / About UI:
document.getElementById('feedback-stats-btn').addEventListener('click', () => {
  canidFeedback.showStats();
});

// Export button (optional — also available from stats modal):
document.getElementById('export-btn').addEventListener('click', () => {
  canidFeedback.exportData();
});
```

### Customizing the UI

Override CSS variables in your app's stylesheet:

```css
:root {
  --canid-fb-accent:  #your-brand-color;
  --canid-fb-radius:  8px;
  --canid-fb-font:    'Your App Font', sans-serif;
}
```

Or override specific classes:

```css
.canid-feedback-prompt {
  /* match your app's bottom nav height */
  bottom: 60px;
}
```

---

## Step 4 — Retrain with User Feedback

### 4a. Export feedback data

In the stats modal (tap "Export Training Data") or programmatically:

```javascript
await canidFeedback.exportData();
// Downloads: canid-feedback-2024-01-15.json
```

### 4b. Convert exported JSON to numpy arrays

```python
# In your retrain script or notebook:
import json
import numpy as np

with open('canid-feedback-2024-01-15.json') as f:
    data = json.load(f)

X_user = np.array([s['features'] for s in data['samples']], dtype=np.float32)
y_user = np.array([
    ['play_bark','warning_bark','alert_bark','demand_bark',
     'whine','growl','howl','yip','silence'].index(s['label'])
    for s in data['samples']
], dtype=np.int32)

print(f"User samples: {X_user.shape}")
```

### 4c. Merge with training data and retrain

```bash
# The user data is already normalized by the JS module before storage.
# You need to un-normalize before merging with the original dataset:
python -c "
import numpy as np, json

with open('dataset/normalization.json') as f:
    norm = json.load(f)

mean = np.array(norm['feature_mean'])
std  = np.array(norm['feature_std'])

# Un-normalize user features (they were stored raw from JS, not normalized)
# NOTE: canid-feedback.js stores RAW features before normalization.
# The JS classifyAsync() normalizes internally — raw features are stored.

X_user = np.load('/tmp/X_user.npy')  # from step 4b
X_merged = np.vstack([np.load('dataset/X_train.npy'), X_user])
y_merged = np.hstack([np.load('dataset/y_train.npy'), np.load('/tmp/y_user.npy')])

np.save('dataset/X_train.npy', X_merged)
np.save('dataset/y_train.npy', y_merged)
print(f'Merged dataset: {X_merged.shape}')
"

python train-model.py --dataset-dir dataset/ --output-dir model/
```

### 4d. Deploy updated model

Replace `tfjs_model/` on your server. The browser will pick up the new model
on next page load.

**Tip:** Version your model URLs to bust CDN cache:
```javascript
modelUrl: './tfjs_model/model.json?v=20240115'
```

---

## Feature Extraction Alignment

**Critical:** The 30-element feature vector must be computed identically in Python
(training) and JavaScript (inference). Any mismatch degrades accuracy silently.

| Feature | Index | Python (`librosa`) | JavaScript (`canid-cnn-classifier.js`) |
|---------|-------|-------------------|----------------------------------------|
| MFCC means | 0–12 | `librosa.feature.mfcc(n_mfcc=13)`, mean over frames | `computeSimplifiedMFCC()` |
| MFCC stds | 13–25 | `librosa.feature.mfcc(n_mfcc=13)`, std over frames | Approximated (see note) |
| Pitch (norm) | 26 | `librosa.pyin()`, voiced mean / 4000 | Autocorrelation / 4000 |
| Spectral centroid (norm) | 27 | `librosa.feature.spectral_centroid()` / (sr/2) | Weighted mag sum / (sr/2) |
| Spectral flatness | 28 | `librosa.feature.spectral_flatness()` | Geo/arith power ratio |
| RMS energy | 29 | `librosa.feature.rms()` | `sqrt(sum(x²)/n)` |

> **MFCC std note:** The JavaScript MFCC implementation is a simplified
> single-frame computation. The MFCC std features (indices 13–25) are
> approximated. For highest accuracy, use [Meyda.js](https://meyda.js.org/)
> in the browser, which computes proper multi-frame MFCC statistics.
> Update `extractFeaturesFromBuffer()` to use Meyda's `mfcc` extractor
> and compute std across frames.

---

## Model Size and Performance

| Metric | Value |
|--------|-------|
| Architecture | Conv1D × 3 + BN + GlobalAvgPool + Dense |
| Parameters | ~120,000 |
| SavedModel size | ~1.5 MB |
| TF.js model size | ~0.5–1.0 MB |
| Inference time (mobile) | <5 ms per frame |
| Memory footprint | ~10 MB (TF.js runtime + model) |
| Validation accuracy (ESC-50) | 55–70% |
| Expected accuracy (with feedback) | 75–85%+ |

The model loads in <2 seconds on 4G and <5 seconds on 3G.

---

## Troubleshooting

**Model fails to load (CORS error)**
→ Serve `tfjs_model/` from the same origin as your app, or add CORS headers.

**Low confidence on all classes**
→ Normalization mismatch. Check that `class_labels.json` `normalization` field
is being loaded by the JS module (see `labelsUrl` option).

**Falls back to rule-based every time**
→ Check browser console for TF.js errors. Ensure `tf.js` is loaded before
`canid-cnn-classifier.js`.

**Feedback prompts don't appear**
→ Call `await canidFeedback.init()` before `canidFeedback.show()`.
Check for console errors.

**IndexedDB quota exceeded**
→ Use the stats modal to export and clear data periodically. Each sample
stores 30 floats (~240 bytes); 10,000 samples ≈ 2.4 MB.

---

## Roadmap / Next Steps

1. **Better MFCC alignment** — Integrate Meyda.js for accurate in-browser MFCC extraction
2. **Breed-specific models** — Fine-tune on per-breed data for higher accuracy
3. **Continuous learning** — Stream feedback samples directly to a lightweight
   federated training endpoint (optional, privacy-preserving)
4. **Spectrogram input** — Replace 1D feature vectors with log-mel spectrograms
   for higher-capacity models (requires larger model budget)
5. **Confidence calibration** — Add temperature scaling post-training to better
   calibrate softmax probabilities
