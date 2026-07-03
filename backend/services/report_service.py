"""
Report generation service: PDF, CSV, and Excel exports.
"""
import io, os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, HRFlowable, Image as RLImage,
)

PLOTS_DIR = Path("frontend/static/plots")


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "id", "session_id", "input_type", "model_used",
        "predicted_digit", "confidence", "processing_time_ms",
        "is_correct", "true_label", "created_at",
    ]
    rows = [{c: r.get(c) for c in cols} for r in records]
    return pd.DataFrame(rows, columns=cols)


def generate_csv(records: List[Dict[str, Any]]) -> bytes:
    df = records_to_dataframe(records)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def generate_excel(records: List[Dict[str, Any]]) -> bytes:
    df = records_to_dataframe(records)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Predictions", index=False)
        ws = writer.sheets["Predictions"]
        # Auto-size columns
        for col in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)
    return buf.getvalue()


def generate_pdf_report(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> bytes:
    buf   = io.BytesIO()
    doc   = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=18, textColor=colors.HexColor("#1a237e"),
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#283593"),
        spaceAfter=6,
    )
    body_style = styles["BodyText"]
    body_style.fontSize = 10

    story = []

    # ── Title ──────────────────────────────────────────────────────────
    story.append(Paragraph("Advanced Digit Recognition — Analytics Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        body_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#283593")))
    story.append(Spacer(1, 0.4*cm))

    # ── Summary Stats ──────────────────────────────────────────────────
    story.append(Paragraph("Summary Statistics", h2_style))
    summary_data = [
        ["Metric", "Value"],
        ["Total Predictions",        str(summary.get("total", 0))],
        ["Average Confidence",       f"{summary.get('avg_confidence', 0):.1f}%"],
        ["Avg Processing Time",      f"{summary.get('avg_time_ms', 0):.1f} ms"],
        ["Canvas Predictions",       str(summary.get("canvas_count", 0))],
        ["Upload Predictions",       str(summary.get("upload_count", 0))],
        ["Most Predicted Digit",     str(summary.get("most_common_digit", "–"))],
    ]
    tbl = Table(summary_data, colWidths=[9*cm, 7*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1,  -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#e8eaf6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#e8eaf6")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fa8da")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.6*cm))

    # ── Plots (if available) ───────────────────────────────────────────
    story.append(Paragraph("Model Evaluation Plots", h2_style))
    for model in ["cnn_small", "cnn_medium", "cnn_deep"]:
        cm_img = PLOTS_DIR / f"{model}_confusion_matrix.png"
        if cm_img.exists():
            story.append(Paragraph(f"Confusion Matrix — {model}", body_style))
            story.append(RLImage(str(cm_img), width=13*cm, height=10*cm))
            story.append(Spacer(1, 0.3*cm))

    # ── Recent Predictions Table ───────────────────────────────────────
    story.append(Paragraph("Recent Prediction Log (last 20)", h2_style))
    table_data = [["#", "Type", "Model", "Digit", "Conf%", "Time(ms)", "Date"]]
    for r in records[-20:]:
        table_data.append([
            str(r.get("id", "")),
            str(r.get("input_type", "")),
            str(r.get("model_used", "")),
            str(r.get("predicted_digit", "")),
            f"{r.get('confidence', 0):.1f}",
            f"{r.get('processing_time_ms', 0):.1f}",
            str(r.get("created_at", ""))[:16],
        ])
    rt = Table(table_data, colWidths=[1.2*cm, 2.2*cm, 3.2*cm, 1.5*cm, 1.8*cm, 2.2*cm, 4*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#e8eaf6")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    story.append(rt)

    doc.build(story)
    return buf.getvalue()
