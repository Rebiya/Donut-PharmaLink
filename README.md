# DONUT-PHARMALINK

Production-ready FastAPI backend for medical prescription image processing.

## Project Overview

DONUT-PHARMALINK accepts a prescription image, extracts text with Donut OCR,
detects likely drug mentions, normalizes names against an FDA dataset, and uses
an OpenRouter LLM to produce structured JSON output.

## Architecture

Frontend -> FastAPI -> Donut OCR -> Candidate Extraction ->
Normalization (Fuzzy + Embedding + FAISS + FDA dataset) ->
OpenRouter LLM (Schema Enforced) -> JSON Response

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

- `OPENROUTER_API_KEY=...`
- `OPENROUTER_MODEL=deepseek/deepseek-chat`
- `DONUT_MODEL_PATH=model-cache`

## Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/prescription.png"
```

## Test

```bash
pytest -q
```

## Deployment (Hugging Face Spaces)

1. Create a **Docker Space**.
2. Push this repository with `Dockerfile` at root.
3. Set runtime variables:
   - `OPENROUTER_API_KEY=...`
   - `OPENROUTER_MODEL=deepseek/deepseek-chat`
   - `DONUT_MODEL_PATH=model-cache`
   - `LOCAL_FILES_ONLY=true`
4. Space port: `8000`.
5. Health check URL: `/`.

## Troubleshooting

- `ModuleNotFoundError: No module named app`
  - Run tests from project root.
  - Ensure virtualenv is activated and use `python -m pytest`.
  - `pytest.ini` already sets `pythonpath = .`.
- `GET /` returns 404
  - Use latest code; root health endpoint is `GET /`.
- OpenRouter connection errors
  - Verify `OPENROUTER_API_KEY` is set.
  - Check outbound network access to `https://openrouter.ai/api/v1/chat/completions`.
- Slow first prediction
  - First run downloads/caches models and builds FAISS index.