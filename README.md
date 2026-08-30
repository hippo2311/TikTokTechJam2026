# AI Image Check

AI Image Check is a Safari/Chrome WebExtension that captures an image area, sends it to a FastAPI backend, and displays the model prediction in the browser. Human feedback is stored in a database and displayed in the monitoring dashboard.

## Architecture

```text
Browser Extension → HTTPS → FastAPI Backend → Trained Model
       ↑                         ↓
       └──── Dashboard ← /stats ┴── Database
```

The extension never connects directly to the database. FastAPI owns model inference, feedback storage, and dashboard data.

## Folder structure

```text
TechJam/
├── extension/                  # Browser/Safari WebExtension and dashboard
│   ├── manifest.json
│   ├── background.js            # Capture and API calls
│   ├── content.js               # Floating UI and result display
│   ├── popup.*                  # Extension popup
│   └── dashboard.*              # Monitoring dashboard
├── detector/                   # FastAPI backend
│   ├── detector.py              # API routes and model inference
│   ├── database.py              # Database connection and feedback model
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── data/                       # Local SQLite database and legacy files
└── README.md
```

## Detection flow

1. The user selects an image area or captures the current viewport.
2. `extension/background.js` creates a PNG data URL.
3. The extension sends it to `POST /detect`.
4. FastAPI passes the image to the trained model.
5. The backend converts the model output to the common response below.
6. `extension/content.js` displays the verdict and confidence.

```json
{
  "verdict": "ai-generated",
  "confidence": 91,
  "note": "Prediction from the configured model"
}
```

`verdict` must be `ai-generated` or `not-ai`. `confidence` is a percentage from 0 to 100.

## Reelistic model

The backend uses the Reelistic seed-43 ensemble from the companion repository:

```text
reelistic/cluster_results/seed43/best_ensemble_calibrated.pt
```

The checkpoint is downloaded from Hugging Face and is intentionally excluded
from Git because it is approximately 1.2 GB. From the project root:

```bash
git clone https://github.com/Bryanngu03/TiktokTechjam26Reelistic.git reelistic
pip install -r reelistic/requirements.txt
mkdir -p reelistic/cluster_results/seed43
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('bryan3112/reelistic-seed43', 'best_ensemble_calibrated.pt', local_dir='reelistic/cluster_results/seed43')"
```

Keep the `/detect` API unchanged so the browser extension does not depend on
the model implementation.

For a Hugging Face-compatible model:

```bash
HF_MODEL=your-model-id uvicorn detector.detector:app --host 127.0.0.1 --port 8000
```

For a local or non-Hugging Face model, place it in `detector/models/`, load it using its framework, run inference in `/detect`, and convert the result to the response format above. This works with PyTorch, ONNX Runtime, TensorFlow, or scikit-learn.

## Feedback and database

`POST /feedback` stores the user's prediction and correction. If the user does nothing for five seconds, or starts a new capture, the previous prediction is automatically recorded as `correct`.

The backend uses SQLite locally at `data/feedback.db`. For production, set:

```bash
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE'
```

The API and dashboard do not need to change when switching to PostgreSQL.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r detector/requirements.txt
uvicorn detector.detector:app --host 127.0.0.1 --port 8000
```

Check the backend:

```bash
curl http://127.0.0.1:8000/health
```

Then open `chrome://extensions`, enable Developer mode, select **Load unpacked**, and choose the `extension/` folder.

## Dashboard connection

The dashboard uses these FastAPI endpoints:

- `GET /stats` — accuracy, reviewed samples, and wrong detections.
- `POST /seed-demo` — add demo feedback.
- `POST /retrain` — create a retraining report.

For local development, `extension/dashboard.js` uses:

```javascript
const API = "http://127.0.0.1:8000";
```

## Cloud deployment

```text
Extension → https://api.example.com → FastAPI → PostgreSQL
                                      └───────→ trained model
```

On the cloud server:

```bash
git clone <your-repository-url>
cd TechJam
python3 -m venv .venv
source .venv/bin/activate
pip install -r detector/requirements.txt
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE'
uvicorn detector.detector:app --host 0.0.0.0 --port 8000
```

Put a domain and HTTPS reverse proxy in front of Uvicorn. `localhost` only works on the developer's computer. Use a GPU-enabled server if the model requires GPU inference.

Update `extension/background.js`:

```javascript
https://api.example.com/detect
https://api.example.com/feedback
```

Update `extension/dashboard.js`:

```javascript
const API = "https://api.example.com";
```

Update `host_permissions` in `extension/manifest.json`:

```json
"host_permissions": ["<all_urls>", "https://api.example.com/*"]
```

Before public release, add authentication, HTTPS, request limits, rate limiting, restricted CORS, monitoring, and an image-retention policy.

## Docker

```bash
docker build -f detector/Dockerfile -t ai-image-check-backend .
docker run --rm -p 8000:8000 \
  -e DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE' \
  ai-image-check-backend
```

For production, run the container behind an HTTPS gateway and use a persistent PostgreSQL database.
