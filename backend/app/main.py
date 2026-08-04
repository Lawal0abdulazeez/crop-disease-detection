"""
FastAPI Application

Crop Disease Detection API.

Run:
  uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Docs:
  http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.predictor import predictor
from app.utils.logger import get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup (non-fatal if checkpoint missing)."""
    try:
        predictor.load("best")
        logger.info("Model loaded at startup.")
    except FileNotFoundError as exc:
        logger.warning(
            f"Model not loaded at startup: {exc}. "
            "API will still start; /predict will return 503 until a checkpoint exists."
        )
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Crop Disease Detection API",
    description=(
        "Upload leaf images to detect crop diseases using EfficientNet. "
        "Works with smoke, debug, or full training checkpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root():
    return {
        "message": "Crop Disease Detection API",
        "docs": "/docs",
        "health": "/health",
    }
