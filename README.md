# DONUT-PHARMALINK

Production-ready FastAPI backend for medical prescription image processing.

## Project Overview

DONUT-PHARMALINK accepts a prescription image, extracts text with Donut OCR,
detects likely drug mentions, normalizes names against an FDA dataset, and uses
a local Ollama model to produce structured JSON output.

## Architecture

Frontend -> FastAPI -> Donut OCR -> Candidate Extraction ->
Normalization (Fuzzy + Embedding + FAISS + FDA dataset) ->
Ollama LLM (Schema Enforced) -> JSON Response

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

To pull `llama3.1` during image build:

```bash
docker build --build-arg OLLAMA_PULL_DURING_BUILD=1 -t pharmalink .
```

## Ollama Setup

Install Ollama and pull the model:

```bash
ollama pull llama3.1
```

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
   - `OLLAMA_HOST=http://127.0.0.1:11434`
   - `OLLAMA_MODEL=llama3.1`
   - `OLLAMA_PULL_ON_START=1` (first boot can be slow)
4. Space port: `8000`.
5. Health check URL: `/`.

## Troubleshooting

- `ModuleNotFoundError: No module named app`
  - Run tests from project root.
  - Ensure virtualenv is activated and use `python -m pytest`.
  - `pytest.ini` already sets `pythonpath = .`.
- `GET /` returns 404
  - Use latest code; root health endpoint is `GET /`.
- Ollama connection errors
  - Start Ollama locally: `ollama serve`
  - Pull model: `ollama pull llama3.1`
  - Verify host with `OLLAMA_HOST`.
- Slow first prediction
  - First run downloads/caches models and builds FAISS index.