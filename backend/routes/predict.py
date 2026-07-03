"""
Prediction endpoints:
  POST /api/predict/canvas  – base64 image from drawing canvas
  POST /api/predict/upload  – multipart file upload
  POST /api/predict/feedback – user correction label
  GET  /api/predict/history  – paginated prediction logs
  GET  /api/predict/models   – list available models
"""
import base64, io, uuid, logging
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database.db import get_db
from backend.database.models import PredictionLog, AuditLog
from backend.services.prediction_service import (
    predict, generate_gradcam, generate_saliency,
    pil_to_numpy, registry,
)

logger  = logging.getLogger(__name__)
router  = APIRouter(prefix="/api/predict", tags=["Prediction"])
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


def _save_prediction(
    db: Session,
    result: dict,
    input_type: str,
    image_path: Optional[str],
    gradcam_path: Optional[str],
    saliency_path: Optional[str],
    session_id: str,
) -> PredictionLog:
    log = PredictionLog(
        session_id         = session_id,
        input_type         = input_type,
        image_path         = image_path,
        model_used         = result["model_used"],
        predicted_digit    = result["predicted_digit"],
        confidence         = result["confidence"],
        top3_predictions   = result["top3_predictions"],
        all_probabilities  = result["all_probabilities"],
        gradcam_path       = gradcam_path,
        saliency_path      = saliency_path,
        processing_time_ms = result["processing_time_ms"],
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


# ── Canvas endpoint ──────────────────────────────────────────────────────────
class CanvasPayload(BaseModel):
    image_data:  str                        # "data:image/png;base64,..."
    model_name:  str = "cnn_medium"
    generate_xai: bool = True

@router.post("/canvas")
async def predict_canvas(payload: CanvasPayload, request: Request, db: Session = Depends(get_db)):
    try:
        # Decode base64
        header, b64 = payload.image_data.split(",", 1)
        raw  = base64.b64decode(b64)
        pil  = Image.open(io.BytesIO(raw)).convert("RGBA")
        arr  = pil_to_numpy(pil)

        result        = predict(arr, payload.model_name)
        gradcam_path  = generate_gradcam(arr, payload.model_name) if payload.generate_xai else None
        saliency_path = generate_saliency(arr, payload.model_name) if payload.generate_xai else None

        session_id = request.cookies.get("session_id", str(uuid.uuid4()))
        log = _save_prediction(db, result, "canvas", None,
                               gradcam_path, saliency_path, session_id)

        db.add(AuditLog(
            action="canvas_prediction",
            entity_type="prediction_log",
            entity_id=log.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            details={"model": payload.model_name},
        ))
        db.commit()

        return {**result, "log_id": log.id,
                "gradcam_path": gradcam_path, "saliency_path": saliency_path}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Canvas prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error")


# ── File-upload endpoint ─────────────────────────────────────────────────────
@router.post("/upload")
async def predict_upload(
    request: Request,
    file: UploadFile = File(...),
    model_name: str  = Form("cnn_medium"),
    generate_xai: bool = Form(True),
    db: Session = Depends(get_db),
):
    suffix = "." + (file.filename or "img").rsplit(".", 1)[-1].lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"File type '{suffix}' not allowed.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        pil      = Image.open(io.BytesIO(contents)).convert("RGB")
        arr      = pil_to_numpy(pil)
        result   = predict(arr, model_name)

        # Save image to static/uploads
        fname      = f"upload_{uuid.uuid4().hex[:10]}{suffix}"
        save_path  = f"frontend/static/uploads/{fname}"
        pil.save(save_path)
        img_url    = f"static/uploads/{fname}"

        gradcam_path  = generate_gradcam(arr, model_name)  if generate_xai else None
        saliency_path = generate_saliency(arr, model_name) if generate_xai else None

        session_id = request.cookies.get("session_id", str(uuid.uuid4()))
        log = _save_prediction(db, result, "upload", img_url,
                               gradcam_path, saliency_path, session_id)
        db.add(AuditLog(
            action="upload_prediction",
            entity_type="prediction_log",
            entity_id=log.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            details={"model": model_name, "filename": file.filename},
        ))
        db.commit()

        return {**result, "log_id": log.id, "image_url": img_url,
                "gradcam_path": gradcam_path, "saliency_path": saliency_path}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Upload prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error")


# ── Feedback endpoint ────────────────────────────────────────────────────────
class FeedbackPayload(BaseModel):
    log_id:     int
    true_label: int
    is_correct: bool

@router.post("/feedback")
def submit_feedback(payload: FeedbackPayload, db: Session = Depends(get_db)):
    log = db.get(PredictionLog, payload.log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Prediction log not found.")
    log.is_correct  = payload.is_correct
    log.true_label  = payload.true_label
    db.commit()
    return {"status": "ok", "log_id": payload.log_id}


# ── History endpoint ─────────────────────────────────────────────────────────
@router.get("/history")
def get_history(
    page: int = 1, per_page: int = 20,
    model_name: Optional[str] = None,
    digit: Optional[int] = None,
    db: Session = Depends(get_db),
):
    q = db.query(PredictionLog)
    if model_name:
        q = q.filter(PredictionLog.model_used == model_name)
    if digit is not None:
        q = q.filter(PredictionLog.predicted_digit == digit)
    total   = q.count()
    records = q.order_by(PredictionLog.created_at.desc()) \
               .offset((page - 1) * per_page).limit(per_page).all()
    return {
        "total": total, "page": page, "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "records": [
            {
                "id":                r.id,
                "session_id":        r.session_id,
                "input_type":        r.input_type,
                "model_used":        r.model_used,
                "predicted_digit":   r.predicted_digit,
                "confidence":        r.confidence,
                "top3_predictions":  r.top3_predictions,
                "gradcam_path":      r.gradcam_path,
                "saliency_path":     r.saliency_path,
                "image_path":        r.image_path,
                "processing_time_ms":r.processing_time_ms,
                "is_correct":        r.is_correct,
                "true_label":        r.true_label,
                "created_at":        str(r.created_at),
            }
            for r in records
        ],
    }


# ── Available models ─────────────────────────────────────────────────────────
@router.get("/models")
def list_models():
    return {"models": registry.available()}
