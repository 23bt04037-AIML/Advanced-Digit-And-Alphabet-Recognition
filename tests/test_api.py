"""
Production test suite for DigitAI FastAPI backend.
Run with: pytest tests/ -v

Tests cover:
  - Health endpoint
  - Prediction (canvas + file upload)
  - Feedback submission
  - History pagination + filters
  - Dashboard endpoints
  - Export endpoints (CSV, Excel, PDF)
  - Error handling (bad input, missing model)
"""
import base64, io, os, pytest, struct, zlib
from typing import Generator

import httpx
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL",
    "mysql+mysqlconnector://digitai:digitai123@localhost:3306/digit_recognition_db")
os.environ.setdefault("APP_ENV", "testing")

# Import after env is set
from backend.app import app

client = TestClient(app)


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════
def make_white_png(w=28, h=28) -> bytes:
    """Create a minimal valid grayscale PNG in-memory (white background)."""
    def raw_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw_rows  = b"".join(b"\x00" + b"\xff" * (w * 3) for _ in range(h))
    idat_data = zlib.compress(raw_rows)

    return (
        b"\x89PNG\r\n\x1a\n"
        + raw_chunk(b"IHDR", ihdr_data)
        + raw_chunk(b"IDAT", idat_data)
        + raw_chunk(b"IEND", b"")
    )


def make_b64_canvas_image() -> str:
    """Return a base64-encoded data-URI of a blank 28×28 PNG."""
    png_bytes = make_white_png(280, 280)
    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"


# ════════════════════════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════════════════════════
class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_root_renders_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_dashboard_page(self):
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_history_page(self):
        r = client.get("/history")
        assert r.status_code == 200

    def test_docs(self):
        r = client.get("/docs")
        assert r.status_code == 200


# ════════════════════════════════════════════════════════════════════════════════
# MODELS LIST
# ════════════════════════════════════════════════════════════════════════════════
class TestModels:
    def test_list_models(self):
        r = client.get("/api/predict/models")
        assert r.status_code == 200
        assert "models" in r.json()
        assert isinstance(r.json()["models"], list)


# ════════════════════════════════════════════════════════════════════════════════
# CANVAS PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
class TestCanvasPrediction:
    def test_bad_model_returns_400(self):
        payload = {
            "image_data":  make_b64_canvas_image(),
            "model_name":  "nonexistent_model_xyz",
            "generate_xai": False,
        }
        r = client.post("/api/predict/canvas", json=payload)
        assert r.status_code in (400, 500)

    def test_missing_image_data(self):
        r = client.post("/api/predict/canvas", json={"model_name": "cnn_medium"})
        assert r.status_code == 422

    def test_invalid_base64(self):
        r = client.post("/api/predict/canvas",
                        json={"image_data": "not_base64!!!", "model_name": "cnn_medium"})
        assert r.status_code in (400, 500)


# ════════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
class TestUploadPrediction:
    def test_upload_bad_filetype(self):
        r = client.post(
            "/api/predict/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            data={"model_name": "cnn_medium"},
        )
        assert r.status_code == 400
        assert "not allowed" in r.json()["detail"].lower()

    def test_upload_oversized_file(self):
        big = b"\xff" * (11 * 1024 * 1024)   # 11 MB
        r = client.post(
            "/api/predict/upload",
            files={"file": ("big.png", big, "image/png")},
            data={"model_name": "cnn_medium"},
        )
        assert r.status_code == 413

    def test_upload_valid_png_no_model(self):
        """Valid PNG but model not trained – expect 400/500, not 422."""
        png = make_white_png(28, 28)
        r = client.post(
            "/api/predict/upload",
            files={"file": ("digit.png", png, "image/png")},
            data={"model_name": "cnn_medium", "generate_xai": "false"},
        )
        assert r.status_code in (200, 400, 500)   # 200 only if model exists

    def test_upload_missing_file(self):
        r = client.post("/api/predict/upload", data={"model_name": "cnn_medium"})
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════════════════════
# HISTORY
# ════════════════════════════════════════════════════════════════════════════════
class TestHistory:
    def test_history_default(self):
        r = client.get("/api/predict/history")
        assert r.status_code == 200
        body = r.json()
        for key in ("total", "page", "per_page", "pages", "records"):
            assert key in body

    def test_history_pagination(self):
        r = client.get("/api/predict/history?page=1&per_page=5")
        assert r.status_code == 200
        assert r.json()["per_page"] == 5

    def test_history_filter_digit(self):
        r = client.get("/api/predict/history?digit=7")
        assert r.status_code == 200
        for record in r.json()["records"]:
            assert record["predicted_digit"] == 7

    def test_history_filter_model(self):
        r = client.get("/api/predict/history?model_name=cnn_medium")
        assert r.status_code == 200


# ════════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ════════════════════════════════════════════════════════════════════════════════
class TestFeedback:
    def test_feedback_not_found(self):
        r = client.post("/api/predict/feedback",
                        json={"log_id": 999999, "true_label": 5, "is_correct": False})
        assert r.status_code == 404

    def test_feedback_invalid_payload(self):
        r = client.post("/api/predict/feedback", json={})
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
class TestDashboard:
    def test_summary(self):
        r = client.get("/api/dashboard/summary")
        assert r.status_code == 200
        for key in ("total", "avg_confidence", "canvas_count", "upload_count"):
            assert key in r.json()

    def test_digit_distribution(self):
        r = client.get("/api/dashboard/digit-distribution")
        assert r.status_code == 200
        dist = r.json()["distribution"]
        assert len(dist) == 10   # digits 0-9

    def test_confidence_histogram(self):
        r = client.get("/api/dashboard/confidence-histogram")
        assert r.status_code == 200
        assert "histogram" in r.json()

    def test_model_comparison(self):
        r = client.get("/api/dashboard/model-comparison")
        assert r.status_code == 200
        assert "models" in r.json()

    def test_runtime_logs(self):
        r = client.get("/api/dashboard/runtime-logs")
        assert r.status_code == 200
        assert "logs" in r.json()

    def test_runtime_logs_filter(self):
        r = client.get("/api/dashboard/runtime-logs?level=ERROR")
        assert r.status_code == 200

    def test_model_metrics(self):
        r = client.get("/api/dashboard/model-metrics")
        assert r.status_code == 200
        assert "metrics" in r.json()


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════
class TestExports:
    def test_csv_export(self):
        r = client.get("/api/dashboard/export/csv")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        assert "predictions.csv" in r.headers.get("content-disposition", "")

    def test_excel_export(self):
        r = client.get("/api/dashboard/export/excel")
        assert r.status_code == 200
        assert "spreadsheetml" in r.headers["content-type"]

    def test_pdf_export(self):
        r = client.get("/api/dashboard/export/pdf")
        assert r.status_code == 200
        assert "application/pdf" in r.headers["content-type"]
        assert r.content[:4] == b"%PDF"
