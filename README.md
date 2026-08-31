# Reelistic: Robust AI-Generated Image Detection

Reelistic is an end-to-end system for detecting AI-generated images after the kinds of transformations that happen on real platforms: JPEG recompression, blur, resizing, noise, color adjustment, and cropping. It combines a browser extension, a cloud-hosted inference API, a human-review dashboard, and a multi-scale DINOv3 forensic classifier.

> **Submission status:** the application, DINOv3 inference path, cloud deployment, human-in-the-loop review flow, and ONNX export are implemented. Final training, validation, test, and robustness results will be inserted into the empty tables in [Results](#results).

## What the app does

The browser extension lets a user drag over an image on a web page. It captures the visible tab, crops the selected region, validates the capture, and sends the image to the backend. The backend returns `ai-generated` or `human-made` with a confidence score.

The result stays lightweight for the user, while the full event is available to an administrator:

- The extension records the selected image, prediction, confidence, source page, and user feedback.
- If the user marks a prediction as wrong, the app infers the opposite actual label.
- If the user moves on without responding, the result is auto-confirmed after 10 seconds.
- Images that cannot be processed are retained as `UNREVIEWED` and excluded from model-quality metrics.
- The admin dashboard shows recent predictions, image previews, filters, accuracy over time, a confusion matrix, and review status.
- An admin can open a prediction, inspect the image and metadata, correct its status/actual label, and write the review back to the database.
- Admin-approved reviews can be curated into new training material for a later model version.

![Reelistic application workflow](docs/app-workflow.svg)

### Capture and inference flow

1. A user activates the extension and drags a rectangle over an image.
2. The extension captures the visible browser tab and crops the selected area to PNG.
3. `POST /detect` decodes and preprocesses the image to a normalized `3 x 224 x 224` tensor.
4. Reelistic DINOv3 returns one logit; `sigmoid(logit)` is the AI-generated probability.
5. The API stores the prediction and image reference, then returns a stable response to the extension.
6. User feedback or an admin correction is linked to the original prediction ID.

```json
{
  "id": 42,
  "verdict": "ai-generated",
  "confidence": 91,
  "note": "Reelistic DINOv3 probability"
}
```

### Dashboard, database, and hosting

The FastAPI service is hosted on Google Compute Engine. Captured images and event JSON are archived in Google Cloud Storage; structured prediction and feedback records are stored through SQLAlchemy. SQLite is used for the current prototype, while the same data layer can be pointed to PostgreSQL using `DATABASE_URL`.

The database keeps:

| Entity | Stored fields |
|---|---|
| Prediction | ID, predicted label, confidence, AI probability, Cloud Storage URI, timestamp |
| Feedback | Prediction ID, review status, actual label, original prediction, confidence, source page, timestamp |
| Image archive | Captured PNG and its matching prediction/feedback event JSON in Cloud Storage |

Only reviewed `CORRECT` and `WRONG` records contribute to accuracy, trend charts, and the confusion matrix. `UNREVIEWED` and processing-error records remain visible for investigation but do not change the reported metrics.

### Human-in-the-loop retraining

Feedback is not used for immediate online learning. A user review or admin correction is first written back to the prediction database and linked to the captured image. The administrator can then inspect the evidence and approve a reliable actual label.

Approved records form a candidate training pool. Before they can influence the model, the pool is:

1. audited for incorrect or ambiguous labels;
2. checked for supported and decodable image formats;
3. deduplicated against existing development and evaluation data;
4. assigned to a versioned training manifest without contaminating validation or test splits—the final test set remains permanently excluded;
5. used in an offline retraining run with the full clean and transformation evaluation suite.

A retrained checkpoint is deployed only if it improves the agreed validation criteria without unacceptable regression under transformations or at low false-positive operating points. The deployment remains behind the same `/detect` API, so the extension and dashboard continue to work without model-specific changes. This creates a controlled loop:

```text
Prediction → human/admin review → database → approved training pool
    → audit and deduplication → offline retraining and evaluation
    → versioned checkpoint/ONNX → cloud deployment → new predictions
```

Validation data may guide model selection, but final test data never enters this loop. After retraining, a new model must be selected using development/validation evidence before any one-time final-test evaluation.

## The model: multi-scale DINOv3 forensics

The detector fine-tunes `facebook/dinov3-vitb16-pretrain-lvd1689m` and taps three depths of the ViT-B/16 backbone. A single final layer can over-specialize in semantics; the multi-scale design preserves evidence from lower-level image formation through higher-level scene consistency.

![Reelistic DINOv3 architecture](docs/model-architecture.svg)

### Why layers 4, 8, and 12?

| Tap | Role in the detector | Examples of evidence |
|---|---|---|
| Layer 4 | Low-level forensic evidence | High-frequency noise, edge anomalies, resampling traces, pixel-grid irregularities |
| Layer 8 | Mid-level appearance evidence | Local texture consistency, repeated patterns, material coherence, regional structure |
| Layer 12 | High-level semantic evidence | Global composition, lighting consistency, object relations, anatomical and physical plausibility |

At every tapped layer, Reelistic uses two complementary views:

- **Patch branch:** patch tokens are reshaped to a `14 x 14` feature map, processed by a `3 x 3` convolution, GELU, and global pooling.
- **CLS branch:** the global CLS token passes through a two-layer MLP.

The six resulting 512-dimensional features are concatenated into a 3,072-dimensional representation, normalized, and passed to the binary classifier. A projection head is used by the supervised contrastive objective during training and is not required for deployment.

### Model specification

| Property | Value |
|---|---|
| Backbone | DINOv3 ViT-B/16 |
| Input | RGB, `224 x 224`, ImageNet normalization |
| Patch size | `16 x 16` |
| Layer taps | 4, 8, 12 |
| Fused feature size | 3,072 |
| Positive class | `ai_generated` |
| Checkpoint parameters | 101,532,673 |
| Competition limit | Less than 2 billion parameters |
| Selection metric | Internal-validation TPR at 1% FPR |
| Deployment formats | PyTorch checkpoint and ONNX opset 18 |

### Training objective and robustness curriculum

The training objective combines binary classification with prediction consistency and supervised contrastive learning:

```text
L = BCEWithLogits + 0.5 x Binary-JS-Consistency + 0.1 x Supervised-Contrastive
```

Training uses source- and class-balanced batches from SID, CIFAKE, and approved WildFake data. Each augmented view receives one sampled transformation rather than a chain of transformations. The curriculum moves from easier to harder parameters while clean images remain the classification anchor.

| Transformation | Parameters evaluated |
|---|---|
| JPEG compression | Quality 90, 70, 50, 30 |
| Gaussian blur | Sigma 0.5, 1.0, 2.0 |
| Resize and upscale | Scale 0.5, 0.25 |
| Gaussian noise | Sigma 0.02, 0.05, 0.10 |
| Color jitter | Strength 0.10, 0.20 |
| Center crop | Keep 80% |

The organizer demonstration set is evaluation-only and is never added to training.

> **Strict test isolation:** the final test set is never used for initial training, feedback-based retraining, augmentation design, checkpoint selection, hyperparameter tuning, calibration, or threshold selection. It is evaluated only after the model and decision policy are frozen. No dashboard feedback record can be assigned to the final test manifest.

## Results

These tables are intentionally empty. They are the single source of truth to fill after the final training, validation, and test artifacts are supplied.

### Training result

| Checkpoint / epoch | Samples | Loss | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TPR @ 1% FPR | TPR @ 5% FPR | FPR @ 95% TPR | FPR @ 99% TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _Pending_ |  |  |  |  |  |  |  |  |  |  |  |  |  |

#### Training robustness

| Condition | Parameter | Samples | Accuracy | Balanced accuracy | F1 | ROC-AUC | TPR @ 1% FPR | Delta ROC-AUC vs clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | None |  |  |  |  |  |  |  |
| JPEG | Quality 90 |  |  |  |  |  |  |  |
| JPEG | Quality 70 |  |  |  |  |  |  |  |
| JPEG | Quality 50 |  |  |  |  |  |  |  |
| JPEG | Quality 30 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 0.5 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 1.0 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 2.0 |  |  |  |  |  |  |  |
| Resize | Scale 0.5 |  |  |  |  |  |  |  |
| Resize | Scale 0.25 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.02 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.05 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.10 |  |  |  |  |  |  |  |
| Color jitter | Strength 0.10 |  |  |  |  |  |  |  |
| Color jitter | Strength 0.20 |  |  |  |  |  |  |  |
| Center crop | 0.80 |  |  |  |  |  |  |  |

### Validation result

| Checkpoint / epoch | Samples | Loss | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TPR @ 1% FPR | TPR @ 5% FPR | FPR @ 95% TPR | FPR @ 99% TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _Pending_ |  |  |  |  |  |  |  |  |  |  |  |  |  |

#### Validation robustness

| Condition | Parameter | Samples | Accuracy | Balanced accuracy | F1 | ROC-AUC | TPR @ 1% FPR | Delta ROC-AUC vs clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | None |  |  |  |  |  |  |  |
| JPEG | Quality 90 |  |  |  |  |  |  |  |
| JPEG | Quality 70 |  |  |  |  |  |  |  |
| JPEG | Quality 50 |  |  |  |  |  |  |  |
| JPEG | Quality 30 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 0.5 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 1.0 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 2.0 |  |  |  |  |  |  |  |
| Resize | Scale 0.5 |  |  |  |  |  |  |  |
| Resize | Scale 0.25 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.02 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.05 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.10 |  |  |  |  |  |  |  |
| Color jitter | Strength 0.10 |  |  |  |  |  |  |  |
| Color jitter | Strength 0.20 |  |  |  |  |  |  |  |
| Center crop | 0.80 |  |  |  |  |  |  |  |

### Test result

| Dataset / split | Samples | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TPR @ 1% FPR | TPR @ 5% FPR | FPR @ 95% TPR | FPR @ 99% TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _Pending_ |  |  |  |  |  |  |  |  |  |  |  |  |

#### Test robustness

| Condition | Parameter | Samples | Accuracy | Balanced accuracy | F1 | ROC-AUC | TPR @ 1% FPR | Delta ROC-AUC vs clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | None |  |  |  |  |  |  |  |
| JPEG | Quality 90 |  |  |  |  |  |  |  |
| JPEG | Quality 70 |  |  |  |  |  |  |  |
| JPEG | Quality 50 |  |  |  |  |  |  |  |
| JPEG | Quality 30 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 0.5 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 1.0 |  |  |  |  |  |  |  |
| Gaussian blur | Sigma 2.0 |  |  |  |  |  |  |  |
| Resize | Scale 0.5 |  |  |  |  |  |  |  |
| Resize | Scale 0.25 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.02 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.05 |  |  |  |  |  |  |  |
| Gaussian noise | Sigma 0.10 |  |  |  |  |  |  |  |
| Color jitter | Strength 0.10 |  |  |  |  |  |  |  |
| Color jitter | Strength 0.20 |  |  |  |  |  |  |  |
| Center crop | 0.80 |  |  |  |  |  |  |  |

### Efficiency and deployment result

| Format | Model size | Device | Batch size | Mean latency | P95 latency | Throughput | Peak memory |
|---|---:|---|---:|---:|---:|---:|---:|
| PyTorch |  |  |  |  |  |  |  |
| ONNX Runtime | 382 MB |  |  |  |  |  |  |

## Try the application

### Live prototype

- API health: [http://34.124.152.42:8000/health](http://34.124.152.42:8000/health)
- Admin dashboard: [http://34.124.152.42:8000/dashboard](http://34.124.152.42:8000/dashboard) (review credentials are provided separately)

The hackathon prototype is hosted on Google Cloud and may be stopped outside the judging window to control cost.

### Browser extension

1. Clone this repository.
2. Open `chrome://extensions` or `edge://extensions`.
3. Enable **Developer mode**.
4. Select **Load unpacked** and choose the `extension/` directory.
5. Pin **AI Image Check**, open a web page, and select an image region.

The extension currently points to the deployed Google Cloud API configured in `extension/background.js`. For another backend, update `API` there and in `extension/dashboard.js`, then update `host_permissions` in `extension/manifest.json`.

### Run the PyTorch API locally

The DINOv3 backbone is gated on Hugging Face. Request access to `facebook/dinov3-vitb16-pretrain-lvd1689m`, authenticate with `hf auth login`, and place the competition checkpoint at:

```text
models/reelistic_dino/checkpoints/best_competition_tpr_at_1_fpr.pt
```

Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r detector/requirements.txt
uvicorn detector.detector:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

### Export ONNX

```bash
python scripts/export_onnx.py \
  --checkpoint models/reelistic_dino/checkpoints/best_competition_tpr_at_1_fpr.pt \
  --config models/reelistic_dino/configs/dinov3_multiscale_full_mixed.toml \
  --output models/reelistic_dino/checkpoints/reelistic_dinov3.onnx
```

The exported opset-18 graph accepts an ImageNet-normalized FP32 tensor named `image` with shape `N x 3 x 224 x 224`. It returns `logit`; use `sigmoid(logit)` as `pred`, the probability that an image is AI-generated.

### Evaluate the ONNX model on your own directory

Install `onnxruntime`, `Pillow`, and `numpy`, then use the following inference core in a directory loop. The submission JSON should contain `image_path` and `pred` for every image.

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
session = ort.InferenceSession("reelistic_dinov3.onnx")

def predict(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = 224 / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)))
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = ((tensor - MEAN) / STD)[None]
    logit = session.run(["logit"], {"image": tensor})[0][0]
    return float(1.0 / (1.0 + np.exp(-logit)))
```

> **Before final submission:** publish the ONNX artifact in the project release or model repository and replace this note with its permanent download URL and SHA-256 checksum.

## Repository structure

```text
.
├── detector/                    # FastAPI API, persistence layer, backend dependencies
├── extension/                   # Browser extension and administrator dashboard
├── models/reelistic_dino/       # DINOv3 forensic source and experiment configs
│   ├── configs/
│   ├── src/
│   └── checkpoints/             # Local/cloud artifacts; ignored by Git
├── scripts/export_onnx.py       # PyTorch-to-ONNX conversion
├── docs/
│   ├── app-workflow.svg
│   └── model-architecture.svg
├── JOURNAL.md                   # Experiment decisions and engineering history
└── README.md
```

## Reproducibility and experiment history

The reasoning behind architecture changes, data controls, training choices, deployment decisions, and unresolved questions is recorded in [JOURNAL.md](JOURNAL.md). The earlier branch-ensemble investigations remain available in the [legacy Reelistic journal](reelistic/JOURNAL.md); those results are not presented as results of the current DINOv3 model.

## Limitations and next steps

- A binary image detector cannot establish authorship or intent; its output is a risk signal, not proof.
- Confidence can shift under unseen generators, screenshots, heavy editing, or source-domain changes.
- Human feedback may contain label noise, so dashboard accuracy reflects reviewed operational samples rather than a controlled benchmark.
- The current prototype uses a public HTTP endpoint; a production release should add HTTPS, API authentication, restricted CORS, rate limiting, retention controls, and service monitoring.
- Final clean/robust benchmark tables, ONNX latency, model download URL, checksum, representative false positives, and representative false negatives are pending.

## Acknowledgements

The system uses PyTorch, Hugging Face Transformers, DINOv3, FastAPI, SQLAlchemy, Google Cloud, ONNX, and ONNX Runtime. The current model implementation is derived from the companion [AIGC-detection repository](https://github.com/omgacai/AIGC-detection), with artifacts maintained in the [Reelistic DINO model repository](https://huggingface.co/omgacai/reelistic-dino). Development datasets and their licenses must be documented in the final data card before submission.
