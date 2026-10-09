#!/bin/bash
# Starts Shiftr on this laptop with AI mode using Ollama (local model, no API key, nothing leaves the laptop).
cd "$(dirname "$0")"

MODEL="qwen2.5:7b"

echo "Checking Ollama..."
if ! curl -s -m 3 http://localhost:11434/api/tags > /dev/null; then
  echo "Starting Ollama..."
  open -a Ollama
  for i in $(seq 1 20); do
    curl -s -m 2 http://localhost:11434/api/tags > /dev/null && break
    sleep 1
  done
fi
if ! curl -s -m 3 http://localhost:11434/api/tags > /dev/null; then
  echo "Ollama is not running. Open the Ollama app from Applications, then run this again."
  read -p "Press Enter to close."
  exit 1
fi
if ! ollama list | grep -q "$MODEL"; then
  echo "Downloading the AI model $MODEL (about 4.7 GB, first time only)..."
  ollama pull "$MODEL"
fi

export SHIFTR_AI_API_KEY="ollama"            # Ollama needs no real key
export SHIFTR_AI_MODEL="$MODEL"
export SHIFTR_AI_BASE_URL="http://localhost:11434/v1"

echo "Starting Shiftr... your browser will open. Keep this window open; press Ctrl+C here to stop."
.venv/bin/streamlit run app.py
