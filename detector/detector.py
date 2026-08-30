import base64
import io
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import torch
from .database import init_database, list_feedback, save_feedback
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REELISTIC_ROOT = PROJECT_ROOT / "reelistic"
sys.path.insert(0, str(REELISTIC_ROOT))
from aigc_detector.data.augmentations import build_eval_transform
from aigc_detector.models.ensemble import AIGCDetectionEnsemble

CHECKPOINT_PATH = Path(os.getenv("REELISTIC_CHECKPOINT", str(REELISTIC_ROOT / "cluster_results/seed43/best_ensemble_calibrated.pt")))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
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


class DetectRequest(BaseModel):
    image: str


class FeedbackRequest(BaseModel):
    feedback: str
    label: str
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
        return {"verdict": verdict, "confidence": round((fake_score if verdict == "ai-generated" else 1 - fake_score) * 100), "note": "Reelistic calibrated FAKE probability"}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    record = request.model_dump()
    record["createdAt"] = record.get("createdAt") or datetime.now(timezone.utc).isoformat()
    save_feedback(record)
    return {"ok": True}


def read_feedback():
    return [{"feedback": row.feedback, "label": row.label, "prediction": row.prediction,
             "confidence": row.confidence, "image": row.image, "page": row.page,
             "createdAt": row.created_at.isoformat()} for row in list_feedback()]


@app.get("/stats")
def stats():
    rows = read_feedback(); reviewed = [row for row in rows if row.get("feedback") in {"correct", "wrong"}]
    correct = sum(row.get("feedback") == "correct" for row in reviewed)
    wrong = [row for row in reviewed if row.get("feedback") == "wrong"]
    wrong_cases = [{key: value for key, value in row.items() if key != "image"} for row in wrong[-30:][::-1]]
    return {"total": len(reviewed), "correct": correct, "wrong": len(wrong), "accuracy": round(correct / len(reviewed) * 100, 1) if reviewed else 0, "wrongCases": wrong_cases}


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
