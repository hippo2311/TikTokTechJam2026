import base64
import io
import json
import os
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 on the Cloud VM
    import tomli as tomllib
import secrets
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google.cloud import storage
from pathlib import Path
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import torch
from .database import init_database, list_feedback, list_predictions, save_feedback, save_prediction, update_prediction_review
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINO_ROOT = PROJECT_ROOT / "models" / "reelistic_dino"
DINO_SOURCE = DINO_ROOT / "src"
sys.path.insert(0, str(DINO_SOURCE))
from robust_aigc.data.transforms import basic_eval_transform
from robust_aigc.models import DINOv3Forensic

CHECKPOINT_PATH = Path(os.getenv("DINO_CHECKPOINT", str(PROJECT_ROOT / "models" / "reelistic_dino" / "checkpoints/best_competition_tpr_at_1_fpr.pt")))
CONFIG_PATH = Path(os.getenv("DINO_CONFIG", str(DINO_ROOT / "configs/dinov3_multiscale_full_mixed.toml")))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
# Deployment cutoff for the calibrated AI probability. Override only with a
# separately documented decision-policy evaluation.
DECISION_THRESHOLD = float(os.getenv("DINO_DECISION_THRESHOLD", "0.5"))
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GCS_CLIENT = storage.Client() if GCS_BUCKET else None
basic_auth = HTTPBasic()
app = FastAPI(title="AI Image Check Local Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
device = torch.device(MODEL_DEVICE)
with CONFIG_PATH.open("rb") as config_file:
    model_config = tomllib.load(config_file)
if not 0.0 <= DECISION_THRESHOLD <= 1.0:
    raise ValueError("DINO_DECISION_THRESHOLD must be between 0 and 1.")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
classifier = DINOv3Forensic(model_config).to(device)
classifier.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model_state")), strict=True)
classifier.eval()
processor = basic_eval_transform(image_size=model_config["data"].get("image_size", 224))
init_database()


def require_admin(credentials: HTTPBasicCredentials):
    username_ok = secrets.compare_digest(credentials.username, os.getenv("DASHBOARD_USER", "admin"))
    password_ok = secrets.compare_digest(credentials.password, os.getenv("DASHBOARD_PASSWORD", "change-me"))
    if not (username_ok and password_ok):
        from fastapi import status
        from fastapi.responses import Response
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials", headers={"WWW-Authenticate": "Basic"})


def archive_event(kind: str, payload: dict, image_data: str | None = None):
    if not GCS_CLIENT or not GCS_BUCKET:
        return None
    bucket = GCS_CLIENT.bucket(GCS_BUCKET)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = f"events/{stamp}-{kind}"
    if image_data:
        raw = image_data.split(",", 1)[1] if "," in image_data else image_data
        bucket.blob(f"{prefix}.png").upload_from_string(base64.b64decode(raw), content_type="image/png")
    bucket.blob(f"{prefix}.json").upload_from_string(json.dumps(payload, default=str), content_type="application/json")
    return f"gs://{GCS_BUCKET}/{prefix}.json"


@app.get("/dashboard")
def dashboard(credentials: HTTPBasicCredentials = Depends(basic_auth)):
    require_admin(credentials)
    return FileResponse(PROJECT_ROOT / "extension" / "dashboard.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/dashboard-assets/{asset}")
def dashboard_asset(asset: str):
    allowed = {"dashboard.css", "dashboard.js"}
    if asset not in allowed:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(PROJECT_ROOT / "extension" / asset, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/prediction-image/{prediction_id}")
def prediction_image(prediction_id: int):
    row = next((item for item in list_predictions(1000) if item.id == prediction_id), None)
    if not row or not row.storage_uri:
        raise HTTPException(status_code=404, detail="Image not found")
    object_uri = row.storage_uri.replace("gs://", "", 1)
    bucket_name, object_name = object_uri.split("/", 1)
    image_name = object_name.removesuffix(".json") + ".png"
    blob = storage.Client().bucket(bucket_name).blob(image_name)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return StreamingResponse(iter([blob.download_as_bytes()]), media_type="image/png")


class DetectRequest(BaseModel):
    image: str


class FeedbackRequest(BaseModel):
    feedback: str
    label: str
    prediction: str | None = None
    confidence: float | None = None
    image: str | None = None
    page: str | None = None
    createdAt: str | None = None
    predictionId: int | None = None


class ReviewRequest(BaseModel):
    status: str


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": "Reelistic DINOv3",
        "device": str(device),
        "decision_threshold": DECISION_THRESHOLD,
    }


@app.post("/detect")
def detect(request: DetectRequest):
    try:
        raw = request.image.split(",", 1)[1] if "," in request.image else request.image
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        inputs = processor(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            fake_score = float(torch.sigmoid(classifier(inputs)["logits"])[0])
        verdict = "ai-generated" if fake_score >= DECISION_THRESHOLD else "not-ai"
        confidence = round((fake_score if verdict == "ai-generated" else 1 - fake_score) * 100)
        storage_uri = archive_event("prediction", {"verdict": verdict, "confidence": confidence, "fake_probability": fake_score, "decision_threshold": DECISION_THRESHOLD}, request.image)
        prediction_id = save_prediction({"verdict": verdict, "confidence": confidence, "fake_probability": fake_score, "storage_uri": storage_uri})
        return {"id": prediction_id, "verdict": verdict, "confidence": confidence, "note": "Reelistic DINOv3 probability"}
    except Exception as error:
        # Keep failed/unsupported images visible to admins for review, but do
        # not treat them as model evidence or include them in metrics.
        error_message = str(error)[:500]
        storage_uri = None
        try:
            storage_uri = archive_event(
                "prediction-error",
                {"verdict": "unreviewed", "confidence": 0, "error": error_message},
                request.image,
            )
        except Exception:
            pass
        prediction_id = save_prediction({
            "verdict": "unreviewed",
            "confidence": 0,
            "fake_probability": 0.5,
            "storage_uri": storage_uri,
        })
        return {
            "id": prediction_id,
            "verdict": "unreviewed",
            "confidence": 0,
            "note": "Image could not be processed; queued for admin review",
        }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    record = request.model_dump()
    record["createdAt"] = record.get("createdAt") or datetime.now(timezone.utc).isoformat()
    save_feedback(record)
    archive_event("feedback", record, request.image)
    return {"ok": True}


@app.post("/predictions/{prediction_id}/review")
def review_prediction(prediction_id: int, request: ReviewRequest, credentials: HTTPBasicCredentials = Depends(basic_auth)):
    require_admin(credentials)
    prediction = next((item for item in list_predictions(1000) if item.id == prediction_id), None)
    if not prediction or request.status not in {"correct", "wrong", "unreviewed"}:
        raise HTTPException(status_code=400, detail="Invalid prediction or status")
    label = prediction.verdict if request.status == "correct" else ("not-ai" if prediction.verdict == "ai-generated" else "ai-generated")
    update_prediction_review(prediction_id, request.status, label)
    return {"ok": True}


def read_feedback():
    return [{"feedback": row.feedback, "label": row.label, "prediction": row.prediction, "predictionId": row.prediction_id,
             "confidence": row.confidence, "image": row.image, "page": row.page,
             "createdAt": row.created_at.isoformat()} for row in list_feedback()]


@app.get("/stats")
def stats(credentials: HTTPBasicCredentials = Depends(basic_auth)):
    require_admin(credentials)
    rows = read_feedback(); reviewed = [row for row in rows if row.get("feedback") in {"correct", "wrong"} and row.get("prediction") in {"ai-generated", "not-ai"} and row.get("label") in {"ai-generated", "not-ai"}]
    correct = sum(row.get("feedback") == "correct" for row in reviewed)
    wrong = [row for row in reviewed if row.get("feedback") == "wrong"]
    def effective_prediction(row):
        if row.get("prediction") in {"ai-generated", "not-ai"}: return row["prediction"]
        return row.get("label") if row.get("feedback") == "correct" else ("not-ai" if row.get("label") == "ai-generated" else "ai-generated")
    tp = sum(effective_prediction(row) == "ai-generated" and row.get("label") == "ai-generated" for row in reviewed)
    fp = sum(effective_prediction(row) == "ai-generated" and row.get("label") == "not-ai" for row in reviewed)
    fn = sum(effective_prediction(row) == "not-ai" and row.get("label") == "ai-generated" for row in reviewed)
    tn = sum(effective_prediction(row) == "not-ai" and row.get("label") == "not-ai" for row in reviewed)
    wrong_cases = wrong[-100:][::-1]
    history = []
    for row in reviewed:
        day = str(row.get("createdAt", ""))[:10] or "unknown"
        item = next((entry for entry in history if entry["date"] == day), None)
        if item is None:
            item = {"date": day, "total": 0, "correct": 0, "wrong": 0}
            history.append(item)
        item["total"] += 1
        item["correct"] += row.get("feedback") == "correct"
        item["wrong"] += row.get("feedback") == "wrong"
    for item in history:
        item["accuracy"] = round(item["correct"] / item["total"] * 100, 1)
    recent_predictions = []
    for row in list_predictions():
        matches = [item for item in rows if item.get("predictionId") == row.id]
        if not matches:
            matches = [item for item in rows if item.get("prediction") == row.verdict and item.get("createdAt")]
        review = min(matches, key=lambda item: abs(datetime.fromisoformat(item["createdAt"]).replace(tzinfo=None) - row.created_at.replace(tzinfo=None))) if matches else None
        recent_predictions.append({"id": row.id, "verdict": row.verdict, "confidence": row.confidence, "createdAt": row.created_at.isoformat(), "storageUri": row.storage_uri, "imageUrl": f"/prediction-image/{row.id}" if row.storage_uri else None, "actual": review.get("label") if review else None, "status": ("correct" if review.get("feedback") == "correct" else "wrong") if review else "unreviewed"})
    return {"total": len(reviewed), "correct": correct, "wrong": len(wrong), "accuracy": round(correct / len(reviewed) * 100, 1) if reviewed else 0, "wrongCases": wrong_cases, "recentPredictions": recent_predictions, "confusionMatrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "history": history}


@app.post("/seed-demo")
def seed_demo():
    labels = [("ai-generated", "ai-generated"), ("ai-generated", "not-ai"), ("not-ai", "not-ai"), ("not-ai", "ai-generated")]
    for index in range(24):
        predicted, actual = labels[index % len(labels)]
        save_feedback({"feedback": "correct" if predicted == actual else "wrong", "label": actual,
                       "prediction": predicted, "page": "demo://sample"})
    return {"ok": True, "added": 24}


@app.post("/retrain")
def retrain():
    rows = read_feedback(); wrong = [row for row in rows if row.get("feedback") == "wrong"]
    report = {"status": "queued", "samples": len(rows), "corrections": len(wrong), "message": "Training job placeholder created. Add a fine-tuning pipeline next."}
    return report
