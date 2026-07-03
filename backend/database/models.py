"""SQLAlchemy ORM models for all database tables."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.sql import func
from backend.database.db import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    role          = Column(String(10), nullable=False, default="client")  # 'admin' | 'client'
    full_name     = Column(String(128), nullable=True)
    email         = Column(String(256), nullable=True)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    last_login    = Column(DateTime(timezone=True), nullable=True)


class OTPRecord(Base):
    """Stores one-time passwords for the Forgot Password flow."""
    __tablename__ = "otp_records"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, nullable=False, index=True)
    otp_code   = Column(String(6),  nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used       = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PredictionLog(Base):
    """Legacy digit-only prediction log. Kept for backward compatibility."""
    __tablename__ = "prediction_logs"

    id                 = Column(Integer, primary_key=True, index=True)
    session_id         = Column(String(64), index=True, nullable=False)
    user_id            = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    username           = Column(String(64), nullable=True, index=True)
    input_type         = Column(String(20), nullable=False)   # 'canvas' | 'upload'
    image_path         = Column(String(512), nullable=True)
    model_used         = Column(String(50), nullable=False)
    predicted_digit    = Column(Integer, nullable=False)
    confidence         = Column(Float, nullable=False)
    top3_predictions   = Column(JSON, nullable=True)
    all_probabilities  = Column(JSON, nullable=True)
    gradcam_path       = Column(String(512), nullable=True)
    lime_path          = Column(String(512), nullable=True)
    saliency_path      = Column(String(512), nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    is_correct         = Column(Boolean, nullable=True)
    true_label         = Column(Integer, nullable=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, index=True)
    action      = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id   = Column(Integer, nullable=True)
    user_agent  = Column(String(512), nullable=True)
    ip_address  = Column(String(45), nullable=True)
    details     = Column(JSON, nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())


class RuntimeLog(Base):
    __tablename__ = "runtime_logs"

    id         = Column(Integer, primary_key=True, index=True)
    level      = Column(String(10), nullable=False)   # INFO | WARNING | ERROR
    module     = Column(String(100), nullable=False)
    message    = Column(Text, nullable=False)
    extra      = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id                    = Column(Integer, primary_key=True, index=True)
    model_name            = Column(String(50), unique=True, nullable=False)
    accuracy              = Column(Float, nullable=True)
    precision_score       = Column(Float, nullable=True)
    recall_score          = Column(Float, nullable=True)
    f1_score              = Column(Float, nullable=True)
    auc_roc               = Column(Float, nullable=True)
    confusion_matrix      = Column(JSON, nullable=True)
    training_epochs       = Column(Integer, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    parameters            = Column(Integer, nullable=True)
    created_at            = Column(DateTime(timezone=True), server_default=func.now())
    updated_at            = Column(DateTime(timezone=True), onupdate=func.now())


class UniversalPredictionLog(Base):
    """
    Unified history log for all 4 prediction categories:
      - 'digit'      : single digit (CNN) — canvas or upload
      - 'multidigit' : multi-digit number detection — canvas or upload
      - 'alphabet'   : OCR text / alphabet / word — canvas or uploaded image
      - 'script'     : document OCR — PDF / Word / PPT
    """
    __tablename__ = "universal_prediction_logs"

    id                  = Column(Integer, primary_key=True, index=True)
    session_id          = Column(String(64), index=True, nullable=False)
    user_id             = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    username            = Column(String(64), nullable=True, index=True)   # denormalised

    # ── Category ─────────────────────────────────────────────────────────
    prediction_category = Column(String(20), nullable=False, index=True)
    # 'digit' | 'multidigit' | 'alphabet' | 'script'

    # ── Input metadata ───────────────────────────────────────────────────
    input_source        = Column(String(30), nullable=True)
    # 'canvas' | 'upload' | 'batch' | 'document'
    source_filename     = Column(String(512), nullable=True)   # for documents (PDF/Word/PPT name)
    image_path          = Column(String(512), nullable=True)   # saved image path

    # ── Model / engine ───────────────────────────────────────────────────
    model_used          = Column(String(100), nullable=True)
    ocr_mode            = Column(String(50), nullable=True)    # OCR reading mode

    # ── Digit / multi-digit fields ───────────────────────────────────────
    predicted_digit     = Column(Integer, nullable=True)       # single digit result
    confidence          = Column(Float, nullable=True)         # 0–100 for digit
    top3_predictions    = Column(JSON, nullable=True)
    multidigit_result   = Column(String(512), nullable=True)   # e.g. "12345"
    digit_count         = Column(Integer, nullable=True)

    # ── OCR / alphabet / script fields ───────────────────────────────────
    ocr_text            = Column(Text, nullable=True)          # full extracted text
    ocr_confidence      = Column(Float, nullable=True)         # 0–1
    line_count          = Column(Integer, nullable=True)

    # ── Timing ───────────────────────────────────────────────────────────
    processing_time_ms  = Column(Float, nullable=True)

    # ── Extra JSON (line details, row details, debug info, etc.) ─────────
    extra_data          = Column(JSON, nullable=True)

    # ── Feedback ─────────────────────────────────────────────────────────
    is_correct          = Column(Boolean, nullable=True)
    true_label          = Column(String(256), nullable=True)   # string so it works for text too

    created_at          = Column(DateTime(timezone=True), server_default=func.now())
