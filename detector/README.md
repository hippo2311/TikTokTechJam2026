# AI Image Check Backend

This folder contains the local AI detection API. It loads the trained model, analyzes images, and returns a stable response format for the browser extension.

Feedback is stored through the database layer in `database.py`. SQLite is used by default for local development; set `DATABASE_URL` to a PostgreSQL connection string in production. The API and dashboard do not need to change when the database changes.

## API contract

The extension sends an image to:

```http
POST /detect
Content-Type: application/json
```

Request:

```json
{
  "image": "data:image/png;base64,..."
}
```

Response:

```json
{
  "verdict": "ai-generated",
  "confidence": 91,
  "note": "Prediction from the configured model"
}
```

The `verdict` must be `ai-generated` or `not-ai`. The UI does not need to know which model produced the result.

The deployed Reelistic DINOv3 decision cutoff is `0.5` for the calibrated AI
probability. Set `DINO_DECISION_THRESHOLD` only when a new, documented
operating-point evaluation selects a replacement.

## Run locally

From the project root:

```bash
pip install -r detector/requirements.txt
uvicorn detector.detector:app --host 127.0.0.1 --port 8000
```

The extension will call `http://localhost:8000` during local development.

The default local database is created automatically at `data/feedback.db`.

For a cloud PostgreSQL database:

```bash
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE'
uvicorn detector.detector:app --host 0.0.0.0 --port 8000
```

The backend creates the `feedback` table on startup. For a larger production system, add database migrations before changing the schema.

## Use a custom trained model

Place the model under `detector/models/` and update the model-loading and inference adapter in `detector.py`. Keep the `/detect` response format unchanged so the extension continues to work.

For a Hugging Face-compatible model, set:

```bash
HF_MODEL=your-model-id uvicorn detector.detector:app --host 0.0.0.0 --port 8000
```

For a non-Hugging Face model, replace the Hugging Face loading code with the loader for your format, such as PyTorch, ONNX, TensorFlow, or scikit-learn. Only the backend adapter needs to change.

## Recommended deployment: FastAPI on a cloud server

The recommended setup is to run this FastAPI application on a cloud server. The cloud server hosts the trained model and exposes a public HTTPS API for the browser extension.

```text
Browser extension
        ↓ HTTPS
https://api.example.com
        ↓
FastAPI /detect and /feedback
        ↓
Trained model on the cloud server
```

### Step 1: Prepare the cloud server

Create a cloud server with Python 3.11 or newer. Choose a GPU-enabled server if the model requires GPU inference. Copy this project to the server and install the backend dependencies:

```bash
git clone <your-repository-url>
cd TechJam
python3 -m venv .venv
source .venv/bin/activate
pip install -r detector/requirements.txt
```

Place a custom local model in `detector/models/`, or configure a compatible model with `HF_MODEL`.

### Step 2: Start FastAPI

Run Uvicorn on all network interfaces so the cloud server can receive requests:

```bash
uvicorn detector.detector:app --host 0.0.0.0 --port 8000
```

Verify that the backend is running:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "ok": true,
  "model": "configured-model-name"
}
```

Keep the process running with a service manager such as systemd, or run the included Docker image. Do not rely on a temporary terminal session for production.

### Step 3: Add a domain and HTTPS

Point a domain such as `api.example.com` to the cloud server. Put a reverse proxy or managed HTTPS gateway in front of Uvicorn and forward HTTPS traffic to `127.0.0.1:8000`.

The public API must be accessible at:

```text
https://api.example.com/detect
https://api.example.com/feedback
https://api.example.com/health
```

Do not expose the API to users as `http://localhost:8000`. `localhost` refers to each user's own computer.

### Step 4: Configure production security

Before sharing the extension, configure the cloud firewall and API gateway to:

- Allow HTTPS traffic only.
- Require an API key or another authentication method.
- Restrict CORS to approved extension origins.
- Limit request body size and apply rate limiting.
- Avoid storing uploaded images in the database unless required; use object storage and save only an image URL when possible.
- Keep model files, API keys, and other secrets private.
- Monitor errors, latency, memory use, and GPU use.

### Step 5: Connect the extension

In `extension/background.js`, replace:

```javascript
http://localhost:8000/detect
http://localhost:8000/feedback
```

with:

```javascript
https://api.example.com/detect
https://api.example.com/feedback
```

Update `extension/manifest.json`:

```json
"host_permissions": [
  "<all_urls>",
  "https://api.example.com/*"
]
```

Reload the unpacked extension and test a capture. The UI will continue to work because it only depends on the existing `/detect` response format.

### Step 6: Deploy updates

When the model or backend changes:

```bash
git pull
source .venv/bin/activate
pip install -r detector/requirements.txt
sudo systemctl restart ai-image-check-backend
```

After a backend URL or manifest change, rebuild and resubmit the extension package as required by the target browser store.

## Docker

Build and run the backend from the project root:

```bash
docker build -f detector/Dockerfile -t ai-image-check-backend .
docker run --rm -p 8000:8000 ai-image-check-backend
```

For production, place the container behind HTTPS and configure authentication, CORS, logging, and rate limiting at the gateway or reverse proxy.
