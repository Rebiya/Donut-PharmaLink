FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://127.0.0.1:11434
ENV OLLAMA_MODEL=llama3.1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . /app

# Prefetch Donut model at build time
ARG DOWNLOAD_DONUT_DURING_BUILD=0
RUN if [ "$DOWNLOAD_DONUT_DURING_BUILD" = "1" ]; then python model_download.py; fi

RUN chmod +x /app/docker-entrypoint.sh

# Pull local LLM model at build if enabled
ARG OLLAMA_PULL_DURING_BUILD=0
RUN if [ "$OLLAMA_PULL_DURING_BUILD" = "1" ]; then ollama serve & sleep 5 && ollama pull "$OLLAMA_MODEL"; fi

EXPOSE 8000

CMD ["/app/docker-entrypoint.sh"]
