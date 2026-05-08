#!/usr/bin/env bash
set -euo pipefail

ollama serve >/tmp/ollama.log 2>&1 &

# Give Ollama a short startup window.
sleep 3

if [[ "${OLLAMA_PULL_ON_START:-0}" == "1" ]]; then
  ollama pull "${OLLAMA_MODEL:-llama3.1}" || true
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
