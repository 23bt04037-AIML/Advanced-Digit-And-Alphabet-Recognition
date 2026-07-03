"""
High-level report orchestration.
Generates PDF, CSV, and Excel reports from the database.
Can be run standalone: python -m reporting.report_generator
"""
import logging, os, sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("reports"); OUTPUT_DIR.mkdir(exist_ok=True)


def generate_all_reports(output_dir: Path = OUTPUT_DIR):
    """Generate all report formats and save to disk."""
    from backend.database.db import SessionLocal
    from backend.database.models import PredictionLog
    from backend.services.report_service import (
        generate_pdf_report, generate_csv, generate_excel
    )
    from sqlalchemy import func

    db = SessionLocal()
    try:
        records_orm = db.query(PredictionLog).order_by(
            PredictionLog.created_at.desc()
        ).all()

        records = [
            {
                "id":                  r.id,
                "session_id":          r.session_id,
                "input_type":          r.input_type,
                "model_used":          r.model_used,
                "predicted_digit":     r.predicted_digit,
                "confidence":          r.confidence,
                "processing_time_ms":  r.processing_time_ms,
                "is_correct":          r.is_correct,
                "true_label":          r.true_label,
                "created_at":          str(r.created_at),
            }
            for r in records_orm
        ]

        total   = len(records)
        avg_c   = sum(r["confidence"] or 0 for r in records) / max(total, 1)
        avg_t   = sum(r["processing_time_ms"] or 0 for r in records) / max(total, 1)
        canvas  = sum(1 for r in records if r["input_type"] == "canvas")
        upload  = sum(1 for r in records if r["input_type"] == "upload")
        digit_counts = {}
        for r in records:
            d = r["predicted_digit"]
            digit_counts[d] = digit_counts.get(d, 0) + 1
        most_common = max(digit_counts, key=digit_counts.get) if digit_counts else None

        summary = {
            "total":             total,
            "avg_confidence":    round(avg_c, 2),
            "avg_time_ms":       round(avg_t, 2),
            "canvas_count":      canvas,
            "upload_count":      upload,
            "most_common_digit": most_common,
        }

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # PDF
        pdf_bytes = generate_pdf_report(records, summary)
        pdf_path  = output_dir / f"report_{stamp}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        logger.info(f"PDF  → {pdf_path}")

        # CSV
        csv_bytes = generate_csv(records)
        csv_path  = output_dir / f"predictions_{stamp}.csv"
        csv_path.write_bytes(csv_bytes)
        logger.info(f"CSV  → {csv_path}")

        # Excel
        xlsx_bytes = generate_excel(records)
        xlsx_path  = output_dir / f"predictions_{stamp}.xlsx"
        xlsx_path.write_bytes(xlsx_bytes)
        logger.info(f"Excel→ {xlsx_path}")

        return {
            "pdf":   str(pdf_path),
            "csv":   str(csv_path),
            "excel": str(xlsx_path),
        }
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = generate_all_reports()
    for fmt, path in paths.items():
        print(f"{fmt.upper():6}: {path}")
