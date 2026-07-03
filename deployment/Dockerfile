# ════════════════════════════════════════════════════════════════
#  DigitAI – Production Dockerfile
#  Serves BOTH the FastAPI backend (port 8000)
#  and the Streamlit frontend (port 8501) via supervisor.
#
#  GPU variant: build with --build-arg USE_GPU=1
#  CPU variant: docker build -f deployment/Dockerfile .
# ════════════════════════════════════════════════════════════════

ARG USE_GPU=0

# ── Stage 1: builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU TF by default; GPU users override in docker-compose
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt \
    && pip install --prefix=/install --no-cache-dir supervisor streamlit


# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="DigitAI"
LABEL org.opencontainers.image.description="Advanced Digit Recognition – FastAPI + Streamlit"
LABEL org.opencontainers.image.source="https://github.com/your-org/AdvancedDigitRecognition"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TF_CPP_MIN_LOG_LEVEL=2 \
    APP_ENV=production

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# ── Application code ─────────────────────────────────────────────
COPY backend/          ./backend/
COPY frontend/         ./frontend/
COPY training/         ./training/
COPY preprocessing/    ./preprocessing/
COPY explainability/   ./explainability/
COPY adversarial/      ./adversarial/
COPY evaluation/       ./evaluation/
COPY dashboard/        ./dashboard/
COPY reporting/        ./reporting/
COPY models/           ./models/
COPY streamlit_app.py  ./streamlit_app.py
COPY seed_db.py        ./seed_db.py
COPY .env.example      .env

# ── Supervisor config (runs FastAPI + Streamlit together) ─────────
COPY deployment/supervisord.conf /etc/supervisor/conf.d/digitai.conf

# ── Runtime directories ───────────────────────────────────────────
RUN mkdir -p frontend/static/{uploads,heatmaps,plots} logs reports \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Seed the database on first run via entrypoint
COPY deployment/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -sf http://localhost:8000/health || exit 1
