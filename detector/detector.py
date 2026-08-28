import base64
import io
import os
import json
import random
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
from .database import init_database, list_feedback, save_feedback

MODEL_ID = os.getenv("HF_MODEL", "jacoballessio/ai-image-detect-distilled")
app = FastAPI(title="AI Image Check Local Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
classifier = AutoModelForImageClassification.from_pretrained(MODEL_ID)
classifier.eval()
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
    return {"ok": True, "model": MODEL_ID}


@app.post("/detect")
def detect(request: DetectRequest):
    try:
        raw = request.image.split(",", 1)[1] if "," in request.image else request.image
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            probabilities = torch.softmax(classifier(**inputs).logits, dim=-1)[0]
        fake_index = next((index for index, label in classifier.config.id2label.items() if label.lower() in {"fake", "ai", "ai_generated", "generated"}), 0)
        fake_score = float(probabilities[fake_index])
        verdict = "ai-generated" if fake_score >= 0.5 else "not-ai"
        return {"verdict": verdict, "confidence": round((fake_score if verdict == "ai-generated" else 1 - fake_score) * 100), "note": f"{classifier.config.id2label[fake_index]} probability from local Hugging Face model"}
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
