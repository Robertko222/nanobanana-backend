from __future__ import annotations

from pydantic import BaseModel


class GenerateResponseItem(BaseModel):
    index: int
    ok: bool
    filename: str | None = None
    url: str | None = None
    error: str | None = None
    model_label: str | None = None
    prompt: str | None = None
    generated_prompt: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    api_key_loaded: bool
    gemini_api_key_loaded: bool
    byteplus_api_key_loaded: bool


class PresetsResponse(BaseModel):
    models: dict
    aspect_ratios: list[str]
    qualities: list[str]
    batch_min: int = 1
    batch_max: int = 8
    reference_slots: int = 4