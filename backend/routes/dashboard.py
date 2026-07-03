"""Dashboard & reporting endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import PredictionLog, RuntimeLog, ModelMetrics
from backend.services.report_service import generate_csv, generate_excel, generate_pdf_report

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


def _build_summary(db: Session) -> dict:
    total = db.query(func.count(PredictionLog.id)).scalar() or 0
    avg_conf = db.query(func.avg(PredictionLog.confidence)).scalar() or 0.0
    avg_time = db.query(func.avg(PredictionLog.processing_time_ms)).scalar() or 0.0
    canvas_c = db.query(func.count(PredictionLog.id)) \
                 .filter(PredictionLog.input_type == "canvas").scalar() or 0
    upload_c = db.query(func.count(PredictionLog.id)) \
                 .filter(PredictionLog.input_type == "upload").scalar() or 0

    most_common = (
        db.query(PredictionLog.predicted_digit,
                 func.count(PredictionLog.id).label("cnt"))
        .group_by(PredictionLog.predicted_digit)
        .order_by(func.count(PredictionLog.id).desc())
        .first()
    )

    correct = db.query(func.count(PredictionLog.id)) \
                .filter(PredictionLog.is_correct == True).scalar() or 0

    return {
        "total":              total,
        "avg_confidence":     round(avg_conf, 2),
        "avg_time_ms":        round(avg_time, 2),
        "canvas_count":       canvas_c,
        "upload_count":       upload_c,
        "most_common_digit":  most_common[0] if most_common else None,
        "confirmed_correct":  correct,
    }


@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    return _build_summary(db)


@router.get("/digit-distribution")
def digit_distribution(db: Session = Depends(get_db)):
    rows = (
        db.query(PredictionLog.predicted_digit,
                 func.count(PredictionLog.id).label("count"))
        .group_by(PredictionLog.predicted_digit)
        .order_by(PredictionLog.predicted_digit)
        .all()
    )
    dist = {str(i): 0 for i in range(10)}
    for digit, cnt in rows:
        if digit is not None:
            dist[str(digit)] = cnt
    return {"distribution": dist}


@router.get("/confidence-histogram")
def confidence_histogram(db: Session = Depends(get_db)):
    records = db.query(PredictionLog.confidence).all()
    confs   = [r[0] for r in records if r[0] is not None]
    buckets = {f"{i*10}-{(i+1)*10}": 0 for i in range(10)}
    for c in confs:
        idx = min(int(c // 10), 9)
        key = f"{idx*10}-{(idx+1)*10}"
        buckets[key] += 1
    return {"histogram": buckets, "total": len(confs)}


@router.get("/model-comparison")
def model_comparison(db: Session = Depends(get_db)):
    rows = (
        db.query(PredictionLog.model_used,
                 func.count(PredictionLog.id).label("uses"),
                 func.avg(PredictionLog.confidence).label("avg_conf"),
                 func.avg(PredictionLog.processing_time_ms).label("avg_ms"))
        .group_by(PredictionLog.model_used)
        .all()
    )
    return {
        "models": [
            {
                "model":       r.model_used,
                "uses":        r.uses,
                "avg_conf":    round(r.avg_conf or 0, 2),
                "avg_ms":      round(r.avg_ms or 0, 2),
            }
            for r in rows
        ]
    }


@router.get("/runtime-logs")
def runtime_logs(
    level:    Optional[str] = None,
    limit:    int = 100,
    db: Session = Depends(get_db),
):
    q = db.query(RuntimeLog)
    if level:
        q = q.filter(RuntimeLog.level == level.upper())
    logs = q.order_by(RuntimeLog.created_at.desc()).limit(limit).all()
    return {
        "logs": [
            {"id": l.id, "level": l.level, "module": l.module,
             "message": l.message, "created_at": str(l.created_at)}
            for l in logs
        ]
    }


@router.get("/model-metrics")
def get_model_metrics(db: Session = Depends(get_db)):
    rows = db.query(ModelMetrics).all()
    return {
        "metrics": [
            {
                "model_name":    m.model_name,
                "accuracy":      m.accuracy,
                "precision":     m.precision_score,
                "recall":        m.recall_score,
                "f1_score":      m.f1_score,
                "auc_roc":       m.auc_roc,
                "parameters":    m.parameters,
                "training_time": m.training_time_seconds,
            }
            for m in rows
        ]
    }


# ── Export ───────────────────────────────────────────────────────────────────
@router.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    records = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()
    data    = [
        {
            "id": r.id, "session_id": r.session_id, "input_type": r.input_type,
            "model_used": r.model_used, "predicted_digit": r.predicted_digit,
            "confidence": r.confidence, "processing_time_ms": r.processing_time_ms,
            "is_correct": r.is_correct, "true_label": r.true_label,
            "created_at": str(r.created_at),
        }
        for r in records
    ]
    csv_bytes = generate_csv(data)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@router.get("/export/excel")
def export_excel(db: Session = Depends(get_db)):
    records = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()
    data    = [{"id": r.id, "session_id": r.session_id, "input_type": r.input_type,
                "model_used": r.model_used, "predicted_digit": r.predicted_digit,
                "confidence": r.confidence, "processing_time_ms": r.processing_time_ms,
                "is_correct": r.is_correct, "true_label": r.true_label,
                "created_at": str(r.created_at)} for r in records]
    xlsx_bytes = generate_excel(data)
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=predictions.xlsx"},
    )


@router.get("/export/pdf")
def export_pdf(db: Session = Depends(get_db)):
    records = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()
    data    = [{"id": r.id, "session_id": r.session_id, "input_type": r.input_type,
                "model_used": r.model_used, "predicted_digit": r.predicted_digit,
                "confidence": r.confidence, "processing_time_ms": r.processing_time_ms,
                "is_correct": r.is_correct, "true_label": r.true_label,
                "created_at": str(r.created_at)} for r in records]
    summary = _build_summary(db)
    pdf = generate_pdf_report(data, summary)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=digit_recognition_db_report.pdf"},
    )
