"""
API Routes
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
import io

from app.api.schemas import (
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ClassesResponse,
)
from app.services.predictor import predictor
from app.utils.logger import get_logger

logger = get_logger()
router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def _read_image(file: UploadFile) -> Image.Image:
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. "
            f"Use JPEG or PNG.",
        )
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        ) from exc
    return image


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    loaded = predictor._loaded
    device = str(predictor.device) if loaded else None
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        device=device,
    )


@router.get("/classes", response_model=ClassesResponse, tags=["model"])
def list_classes():
    try:
        predictor._ensure_loaded()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ClassesResponse(
        classes=predictor.class_names,
        num_classes=predictor.num_classes,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    try:
        info = predictor.info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelInfoResponse(**info)


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(file: UploadFile = File(...)):
    """
    Upload a leaf image and receive disease prediction.
    """
    try:
        predictor._ensure_loaded()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image = _read_image(file)
    result = predictor.predict(image)
    return PredictionResponse(**result)


@router.post("/batch-predict", tags=["inference"])
def batch_predict(files: list[UploadFile] = File(...)):
    """
    Upload multiple leaf images for batch prediction.
    """
    try:
        predictor._ensure_loaded()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    for file in files:
        image = _read_image(file)
        result = predictor.predict(image)
        result["filename"] = file.filename
        results.append(result)

    return {"count": len(results), "predictions": results}
