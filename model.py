from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from google import genai
from PIL import Image

from config import BYTEPLUS_API_KEY, BYTEPLUS_BASE_URL, GEMINI_API_KEY, OUTPUTS_DIR, REQUEST_TIMEOUT_SECONDS


ASPECT_RATIOS = ['auto', '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9']
IMAGE_QUALITIES = ['1K', '2K', '4K']


MODEL_PRESETS = {
    'nano-banana-2': {
        'label': 'Nano Banana 2',
        'provider': 'gemini',
        'model_id': 'gemini-3.1-flash-image-preview',
    },
    'nano-banana-pro': {
        'label': 'Nano Banana Pro',
        'provider': 'gemini',
        'model_id': 'gemini-3-pro-image-preview',
    },
    'seedream-4-0': {
        'label': 'Seedream 4.0',
        'provider': 'byteplus',
        'model_id': 'seedream-4-0-250828',
    },
    'seedream-4-5': {
        'label': 'Seedream 4.5',
        'provider': 'byteplus',
        'model_id': 'seedream-4-5-251128',
    },
    'seedream-5-0': {
        'label': 'Seedream 5.0',
        'provider': 'byteplus',
        'model_id': 'seedream-5-0',
    },
}


SEEDREAM_REFERENCE_PROMPT = """You are Gemini 2.5, an expert prompt engineer specializing in the Seedream 4.0 AI model. You create complete, detailed, and technically precise image generation prompts.
Primary Directive: Your task is to analyze Reference Image 3 (a complete scene) and generate a single, comprehensive prompt for Seedream 4.0. This prompt will instruct the model on how to use a total of three reference images.
Critical Context (Non-negotiable): Seedream will always receive 3 reference images in this specific order:
Images 1 & 2: Provide the subject's complete face structure, facial features, and identity.
Image 3: The complete scene reference (this is the image you will be given to analyze).
Your analysis must focus exclusively on Image 3. Your generated prompt must correctly instruct Seedream on this specific 3-image workflow.
Your Generation Task:
You will be given Image 3.
You will analyze Image 3 ONLY.
You will output ONLY the complete, formatted prompt for Seedream. Do not add any conversational preamble, explanation, or text outside the specified format.

Mandatory Output Format (Strict Template):
Use the first two reference images for the subject's complete face, features, and identity. Use reference image 3 as the complete reference for all other elements: clothing, pose, action, body type, scene composition, background environment, lighting, and overall atmosphere.
Subject details: [Describe the subject's clothing in exhaustive detail: every visible garment, accessories, jewelry, and footwear. Specify colors, patterns, textures, cuts, and styles.] [Describe the exact pose: sitting, standing, leaning. Detail the position of the torso, arms, legs, and head.] [Describe the subject's action or gesture and overall body language. Describe the facial expression type but NOT the features.]
The scene: [Describe the location type.] The environment features [describe all significant background and foreground elements]. The setting is [describe the spatial layout].
Lighting: [Describe the lighting in technical detail: identify the primary light source, direction, quality, shadows, time of day, and color temperature.]
Camera: [Describe the camera's properties: angle, shot type, depth of field, and composition.]
Atmosphere: [Describe the mood or ambiance of the scene. If outdoors, note weather conditions or environmental effects.]
Colors and textures: [Describe the dominant color palette of the entire image and highlight key materials and their surface textures.]
Technical quality: [Describe the image's aesthetic and technical style, e.g., high-resolution, photorealistic, sharp focus, professional studio photography, cinematic, 35mm film grain, editorial fashion shot, candid.]

CRITICAL RULES (ABSOLUTE):
DO NOT use double quotes
DO use generic terms: this person, the subject, the individual.
DO be extremely detailed about clothing, accessories, pose, and background elements. These are your primary focus.
DO describe the type of facial expression as this is part of the pose and action."""


@dataclass
class ReferenceImage:
    slot: int
    name: str
    mime_type: str
    data: bytes


def is_byteplus_model(model_key: str) -> bool:
    preset = MODEL_PRESETS.get(model_key)
    return bool(preset and preset['provider'] == 'byteplus')


def create_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError('Chýba GEMINI_API_KEY v backend/.env')
    return genai.Client(api_key=GEMINI_API_KEY)


def encode_reference_as_data_url(reference: ReferenceImage) -> str:
    encoded = base64.b64encode(reference.data).decode('utf-8')
    return f'data:{reference.mime_type};base64,{encoded}'


def build_inline_part(reference: ReferenceImage) -> dict[str, Any]:
    return {
        'inline_data': {
            'mime_type': reference.mime_type,
            'data': base64.b64encode(reference.data).decode('utf-8'),
        }
    }


def parse_gemini_image_bytes(response: Any) -> bytes:
    candidates = getattr(response, 'candidates', None) or []
    for candidate in candidates:
        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', None) or []
        for part in parts:
            inline_data = getattr(part, 'inline_data', None)
            if inline_data and getattr(inline_data, 'data', None):
                return inline_data.data
    raise RuntimeError('Gemini nevrátil obrázok.')


def call_gemini_generate(*, model_name: str, contents: list[Any], image_response: bool) -> Any:
    client = create_gemini_client()
    kwargs: dict[str, Any] = {
        'model': model_name,
        'contents': contents,
    }
    if image_response:
        kwargs['config'] = {'response_modalities': ['IMAGE']}
    return client.models.generate_content(**kwargs)


def generate_reference_prompt(references: list[ReferenceImage]) -> str:
    reference_map = {ref.slot: ref for ref in references}
    missing = [slot for slot in (1, 2, 3) if slot not in reference_map]
    if missing:
        raise RuntimeError(f"Reference Prompt Mode vyžaduje image 1, image 2 a image 3. Chýba: {', '.join(f'image {s}' for s in missing)}")

    response = call_gemini_generate(
        model_name='gemini-2.5-pro',
        contents=[
            build_inline_part(reference_map[1]),
            build_inline_part(reference_map[2]),
            build_inline_part(reference_map[3]),
            SEEDREAM_REFERENCE_PROMPT,
        ],
        image_response=False,
    )

    text = (getattr(response, 'text', None) or '').strip()
    if not text:
        raise RuntimeError('Gemini nevrátil prompt pre Reference Prompt Mode.')
    return text


def generate_gemini_image(*, model_name: str, prompt: str, references: list[ReferenceImage]) -> bytes:
    contents: list[Any] = [build_inline_part(ref) for ref in references]
    contents.append(prompt)
    response = call_gemini_generate(model_name=model_name, contents=contents, image_response=True)
    return parse_gemini_image_bytes(response)


def generate_seedream_image(*, model_name: str, prompt: str, references: list[ReferenceImage], custom_size: str, quality: str) -> bytes:
    if not BYTEPLUS_API_KEY:
        raise RuntimeError('Chýba BYTEPLUS_API_KEY v backend/.env')
    if not prompt.strip():
        raise RuntimeError('Seedream potrebuje neprázdny prompt.')

    payload: dict[str, Any] = {
        'model': model_name,
        'prompt': prompt,
        'sequential_image_generation': 'disabled',
        'response_format': 'b64_json',
        'stream': False,
        'watermark': False,
        'size': custom_size if custom_size and 'x' in custom_size else (quality if quality in {'2K', '4K'} else '2K'),
    }

    if references:
        payload['image'] = encode_reference_as_data_url(references[0]) if len(references) == 1 else [encode_reference_as_data_url(ref) for ref in references]

    request = urllib.request.Request(
        url=f'{BYTEPLUS_BASE_URL}/images/generations',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {BYTEPLUS_API_KEY}',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'BytePlus API chyba: {detail or exc.reason}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Nepodarilo sa pripojiť na BytePlus API: {exc.reason}') from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'BytePlus API vrátil neplatný JSON: {body}') from exc

    data = parsed.get('data') or []
    if not data:
        raise RuntimeError(f'BytePlus API nevrátil obrázok: {body}')

    first = data[0]
    if first.get('b64_json'):
        return base64.b64decode(first['b64_json'])
    if first.get('url'):
        with urllib.request.urlopen(first['url'], timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.read()
    raise RuntimeError(f'Neočakávaná odpoveď z BytePlus API: {body}')


def save_image(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    filename = f'{uuid.uuid4().hex}.jpg'
    path = OUTPUTS_DIR / filename
    image.save(path, 'JPEG', quality=95)
    return filename


def generate_single_image(
    *,
    model_key: str,
    prompt: str,
    references: list[ReferenceImage],
    aspect_ratio: str,
    quality: str,
    custom_size: str,
    iphone_mode: bool,
    sensor_grain: bool = False,
    use_reference_prompt: bool = False,
):
    if model_key not in MODEL_PRESETS:
        raise RuntimeError(f'Neznámy model key: {model_key}')

    preset = MODEL_PRESETS[model_key]
    generated_prompt = generate_reference_prompt(references) if use_reference_prompt else None
    effective_prompt = (generated_prompt or prompt or '').strip()

    if preset['provider'] == 'gemini':
        image_bytes = generate_gemini_image(
            model_name=preset['model_id'],
            prompt=effective_prompt,
            references=references,
        )
    else:
        image_bytes = generate_seedream_image(
            model_name=preset['model_id'],
            prompt=effective_prompt,
            references=references,
            custom_size=custom_size,
            quality=quality,
        )

    filename = save_image(image_bytes)
    return {
        'filename': filename,
        'prompt': effective_prompt,
        'generated_prompt': generated_prompt,
        'model_label': preset['label'],
    }
