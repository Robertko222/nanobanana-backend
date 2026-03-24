from __future__ import annotations

import mimetypes

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import (
    BACKEND_HOST,
    BACKEND_PORT,
    BYTEPLUS_API_KEY,
    CORS_ORIGINS,
    GEMINI_API_KEY,
    OUTPUTS_DIR,
)
from model import (
    ASPECT_RATIOS,
    IMAGE_QUALITIES,
    MODEL_PRESETS,
    ReferenceImage,
    generate_single_image,
    is_byteplus_model,
)
from schemas import GenerateResponseItem, HealthResponse, PresetsResponse

app = FastAPI(title="Nano Banana Local Web", version="1.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        api_key_loaded=bool(GEMINI_API_KEY or BYTEPLUS_API_KEY),
        gemini_api_key_loaded=bool(GEMINI_API_KEY),
        byteplus_api_key_loaded=bool(BYTEPLUS_API_KEY),
    )


@app.get("/api/presets", response_model=PresetsResponse)
def presets() -> PresetsResponse:
    return PresetsResponse(
        models=MODEL_PRESETS,
        aspect_ratios=ASPECT_RATIOS,
        qualities=IMAGE_QUALITIES,
    )


@app.post("/api/generate")
async def generate(
    prompt: str = Form(""),
    model_key: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    quality: str = Form("2K"),
    batch_count: int = Form(...),
    iphone_mode: bool = Form(False),
    sensor_grain: bool = Form(False),
    use_reference_prompt: bool = Form(False),
    custom_size: str = Form(""),
    ref1: UploadFile | None = File(None),
    ref2: UploadFile | None = File(None),
    ref3: UploadFile | None = File(None),
    ref4: UploadFile | None = File(None),
) -> dict:
    if model_key not in MODEL_PRESETS:
        raise HTTPException(status_code=400, detail="Neznámy model.")

    selected_model = MODEL_PRESETS[model_key]

    if selected_model["provider"] == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Chýba GEMINI_API_KEY v backend/.env")

    if selected_model["provider"] == "byteplus" and not BYTEPLUS_API_KEY:
        raise HTTPException(status_code=500, detail="Chýba BYTEPLUS_API_KEY v backend/.env")

    if use_reference_prompt and not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Pre Reference Prompt Mode potrebuješ GEMINI_API_KEY v backend/.env",
        )

    if not prompt.strip() and not use_reference_prompt:
        raise HTTPException(status_code=400, detail="Prompt je povinný.")

    if batch_count < 1 or batch_count > 8:
        raise HTTPException(status_code=400, detail="Batch count musí byť medzi 1 a 8.")

    uploads = [ref1, ref2, ref3, ref4]
    references: list[ReferenceImage] = []

    for idx, upload in enumerate(uploads, start=1):
        if not upload or not upload.filename:
            continue

        file_bytes = await upload.read()
        if not file_bytes:
            continue

        mime_type = upload.content_type or mimetypes.guess_type(upload.filename)[0] or "image/png"
        references.append(
            ReferenceImage(
                slot=idx,
                name=upload.filename,
                mime_type=mime_type,
                data=file_bytes,
            )
        )

    if use_reference_prompt:
        required_slots = {1, 2, 3}
        provided_slots = {ref.slot for ref in references}
        missing = sorted(required_slots - provided_slots)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Pre Reference Prompt Mode musíš nahrať image 1, image 2 a image 3. Chýba: {', '.join(f'image {slot}' for slot in missing)}",
            )

    if is_byteplus_model(model_key) and not custom_size.strip():
        raise HTTPException(status_code=400, detail="Pre Seedream model musí byť vyplnené rozlíšenie.")

    results: list[GenerateResponseItem] = []

    for i in range(batch_count):
        try:
            generated = generate_single_image(
                model_key=model_key,
                prompt=prompt,
                references=references,
                aspect_ratio=aspect_ratio,
                quality=quality,
                custom_size=custom_size,
                iphone_mode=iphone_mode,
                sensor_grain=sensor_grain,
                use_reference_prompt=use_reference_prompt,
            )

            filename = generated["filename"]

            results.append(
                GenerateResponseItem(
                    index=i,
                    ok=True,
                    filename=filename,
                    url=f"/api/outputs/{filename}",
                    model_label=generated["model_label"],
                    prompt=generated["prompt"],
                    generated_prompt=generated.get("generated_prompt"),
                )
            )
        except Exception as exc:
            results.append(
                GenerateResponseItem(
                    index=i,
                    ok=False,
                    error=str(exc),
                )
            )

    return {
        "ok": True,
        "results": [item.model_dump() for item in results],
    }


@app.get("/api/outputs/{filename}")
def get_output(filename: str):
    file_path = OUTPUTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Súbor neexistuje.")

    return FileResponse(path=file_path)


@app.get("/api/outputs")
def list_outputs() -> dict:
    files = sorted(
        [p.name for p in OUTPUTS_DIR.glob("*.jpg")],
        reverse=True,
    )
    return {"files": files}


if __name__ == "__main__":
    uvicorn.run("main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
