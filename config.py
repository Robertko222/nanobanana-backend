from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
BYTEPLUS_API_KEY = os.getenv('BYTEPLUS_API_KEY', '').strip()
BYTEPLUS_BASE_URL = os.getenv('BYTEPLUS_BASE_URL', 'https://ark.ap-southeast.bytepluses.com/api/v3').strip().rstrip('/')
BACKEND_HOST = os.getenv('BACKEND_HOST', '0.0.0.0')
BACKEND_PORT = int(os.getenv('BACKEND_PORT', '8000'))
RAW_CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').strip()
CORS_ORIGINS = [origin.strip() for origin in RAW_CORS_ORIGINS.split(',') if origin.strip()] if RAW_CORS_ORIGINS != '*' else ['*']
REQUEST_TIMEOUT_SECONDS = int(os.getenv('REQUEST_TIMEOUT_SECONDS', '60'))
