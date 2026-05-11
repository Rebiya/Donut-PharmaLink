FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DONUT_MODEL_PATH=model-cache
ENV FAISS_INDEX_PATH=artifacts/faiss_drugs.index
ENV DRUGS_CACHE_PATH=artifacts/drug_names.json
ENV EMBEDDINGS_CACHE_PATH=artifacts/drug_embeddings.npy
ENV GEMINI_MODEL_NAME=models/gemini-2.0-flash-lite
ENV ENABLE_CHAT_ENDPOINT=true
ENV LOCAL_FILES_ONLY=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

CMD ["/app/docker-entrypoint.sh"]
