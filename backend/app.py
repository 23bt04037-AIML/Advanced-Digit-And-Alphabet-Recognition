"""
FastAPI application entry point.

Start with:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from backend.database.db import create_tables, check_connection
from backend.routes.predict import router as predict_router
from backend.routes.dashboard import router as dashboard_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  Advanced Digit Recognition API  –  Starting up  ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    db_ok = check_connection()
    if db_ok:
        create_tables()
    else:
        logger.warning("DB unavailable – running without persistence.")

    yield

    logger.info("Application shutting down…")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Advanced Digit Recognition API",
    description=(
        "Production-ready REST API for handwritten digit recognition using CNN, "
        "MobileNetV2, ResNet50 with Grad-CAM, LIME, FGSM robustness testing, "
        "full audit trails, and PDF/CSV/Excel reporting."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & templates ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict_router)
app.include_router(dashboard_router)


# ── Frontend routes ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/history", response_class=HTMLResponse, include_in_schema=False)
async def history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/adversarial", response_class=HTMLResponse, include_in_schema=False)
async def adversarial(request: Request):
    return templates.TemplateResponse("adversarial.html", {"request": request})


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": app.version}