"""
API Response / Request Schemas
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopKPrediction(BaseModel):
    class_index: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    above_threshold: bool
    top_k: list[TopKPrediction]
    model_name: str
    num_classes: int


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    num_classes: int
    class_names: list[str]
    checkpoint: str | None
    device: str
    confidence_threshold: float
    top_k: int


class ClassesResponse(BaseModel):
    classes: list[str]
    num_classes: int
