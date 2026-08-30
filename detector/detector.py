import base64
import io
import json
import os
import sys
import secrets
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google.cloud import storage
from pathlib import Path
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import torch
from .database import init_database, list_feedback, list_predictions, save_feedback, save_prediction
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REELISTIC_ROOT = PROJECT_ROOT / "reelistic"
sys.path.insert(0, str(REELISTIC_ROOT))
from aigc_detector.data.augmentations import build_eval_transform
from aigc_detector.models.ensemble import AIGCDetectionEnsemble

CHECKPOINT_PATH = Path(os.getenv("REELISTIC_CHECKPOINT", str(REELISTIC_ROOT / "cluster_results/seed43/best_ensemble_calibrated.pt")))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GCS_CLIENT = storage.Client() if GCS_BUCKET else None
basic_auth = HTTPBasic()
app = FastAPI(title="AI Image Check Local Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
device = torch.device(MODEL_DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
checkpoint_args = checkpoint.get("args", {})
classifier = AIGCDetectionEnsemble(
    semantic_backbone=checkpoint_args.get("semantic_backbone", "mobilenetv3_small_100.lamb_in1k"),
    texture_backbone=checkpoint_args.get("texture_backbone", "resnet18"),
    semantic_pretrained=False, texture_pretrained=False,
    image_size=checkpoint_args.get("image_size", 128),
    top_k_patches=checkpoint_args.get("top_k_patches", 1),
    quality_aware_fusion=checkpoint_args.get("quality_aware_fusion", False),
    noise_version=checkpoint_args.get("noise_version", "legacy"),
    noise_enabled=checkpoint_args.get("noise_enabled", True),
    branch_dropout=checkpoint_args.get("branch_dropout", 0.15),
    explicit_gate_disagreement=checkpoint_args.get("explicit_gate_disagreement", False),
).to(device)
classifier.load_state_dict(checkpoint["model_state"])
classifier.eval()
processor = build_eval_transform(image_size=checkpoint_args.get("image_size", 128))
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
    return FileResponse(PROJECT_ROOT / "extension" / "dashboard.html")


@app.get("/dashboard-assets/{asset}")
def dashboard_asset(asset: str):
    allowed = {"dashboard.css", "dashboard.js"}
    if asset not in allowed:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(PROJECT_ROOT / "extension" / asset)


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


@app.get("/health")
def health():
    return {"ok": True, "model": "Reelistic seed-43", "device": str(device)}


@app.post("/detect")
def detect(request: DetectRequest):
    try:
        raw = request.image.split(",", 1)[1] if "," in request.image else request.image
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        inputs = processor(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            fake_score = float(classifier.predict_proba(inputs)[0])
        verdict = "ai-generated" if fake_score >= 0.5 else "not-ai"
        confidence = round((fake_score if verdict == "ai-generated" else 1 - fake_score) * 100)
        storage_uri = archive_event("prediction", {"verdict": verdict, "confidence": confidence, "fake_probability": fake_score}, request.image)
        save_prediction({"verdict": verdict, "confidence": confidence, "fake_probability": fake_score, "storage_uri": storage_uri})
        return {"verdict": verdict, "confidence": confidence, "note": "Reelistic calibrated FAKE probability"}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    record = request.model_dump()
    record["createdAt"] = record.get("createdAt") or datetime.now(timezone.utc).isoformat()
    save_feedback(record)
    archive_event("feedback", record, request.image)
    return {"ok": True}


def read_feedback():
    return [{"feedback": row.feedback, "label": row.label, "prediction": row.prediction,
             "confidence": row.confidence, "image": row.image, "page": row.page,
             "createdAt": row.created_at.isoformat()} for row in list_feedback()]


@app.get("/stats")
def stats(credentials: HTTPBasicCredentials = Depends(basic_auth)):
    require_admin(credentials)
    rows = read_feedback(); reviewed = [row for row in rows if row.get("feedback") in {"correct", "wrong"}]
    correct = sum(row.get("feedback") == "correct" for row in reviewed)
    wrong = [row for row in reviewed if row.get("feedback") == "wrong"]
    tp = sum(row.get("prediction") == "ai-generated" and row.get("label") == "ai-generated" for row in reviewed)
    fp = sum(row.get("prediction") == "ai-generated" and row.get("label") == "not-ai" for row in reviewed)
    fn = sum(row.get("prediction") == "not-ai" and row.get("label") == "ai-generated" for row in reviewed)
    tn = sum(row.get("prediction") == "not-ai" and row.get("label") == "not-ai" for row in reviewed)
    wrong_cases = wrong[-100:][::-1]
    history = []
    for row in reviewed:
        day = str(row.get("createdAt", ""))[:10] or "unknown"
        item = next((entry for entry in history if entry["date"] == day), None)
        if item is None:
            item = {"date": day, "total": 0, "correct": 0}
            history.append(item)
        item["total"] += 1
        item["correct"] += row.get("feedback") == "correct"
    for item in history:
        item["accuracy"] = round(item["correct"] / item["total"] * 100, 1)
    recent_predictions = [{"id": row.id, "verdict": row.verdict, "confidence": row.confidence, "createdAt": row.created_at.isoformat(), "storageUri": row.storage_uri} for row in list_predictions()]
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
