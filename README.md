# DONUT-PHARMALINK

Production-ready FastAPI backend for medical prescription image processing.

## Project Overview

DONUT-PHARMALINK accepts a prescription image, extracts text with Donut OCR,
detects likely drug mentions, normalizes names against an FDA dataset, and uses
Gemini to produce structured JSON output and grounded pharmaceutical chat.

## Architecture

Frontend -> FastAPI -> Donut OCR -> Candidate Extraction ->
Normalization (Fuzzy + Embedding + FAISS + FDA dataset) ->
Gemini (Schema Enforced) -> JSON Response

Chat Query -> Candidate Extraction -> FAISS Grounding -> Gemini -> Safe Response

## Setup (Local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run locally:

```bash
uvicorn app.main:app --reload
```

## API Run Command

```bash
uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t pharmalink .
docker run -p 8000:8000 pharmalink
```

Required runtime variables:

- `GEMINI_API_KEY=...`
- `GEMINI_MODEL_NAME=models/gemini-2.0-flash-lite`
- `DONUT_MODEL_PATH=model-cache`

## Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/prescription.png"
```

## Example Chat Request

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Can I take ibuprofen with warfarin?"}'
```

## Test

```bash
pytest -q
```

## Deployment (Hugging Face Spaces)

1. Create a **Docker Space**.
2. Push this repository with `Dockerfile` at root.
3. Set runtime variables:
   - `GEMINI_API_KEY=...`
   - `GEMINI_MODEL_NAME=models/gemini-2.0-flash-lite`
   - `DONUT_MODEL_PATH=model-cache`
   - `LOCAL_FILES_ONLY=true`
   - `ENABLE_CHAT_ENDPOINT=true`
4. Space port: `8000`.
5. Health check URL: `/`.

## Troubleshooting

- `ModuleNotFoundError: No module named app`
  - Run tests from project root.
  - Ensure virtualenv is activated and use `python -m pytest`.
  - `pytest.ini` already sets `pythonpath = .`.
- `GET /` returns 404
  - Use latest code; root health endpoint is `GET /`.
- Gemini connection errors
  - Verify `GEMINI_API_KEY` is set.
  - Check outbound network access to the Gemini API.
- Slow first prediction
  - First run downloads/caches models and builds FAISS index.
