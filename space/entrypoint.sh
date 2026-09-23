#!/bin/sh
# Space entrypoint: pre-download the model (persistent /data), then serve.
set -e
MODEL_DIR="/data/models/Qwen3-0.6B"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[space] downloading $OPENJEV_MODEL ..."
  python scripts/download_model.py --model "$OPENJEV_MODEL" --out "$MODEL_DIR"
fi
exec python -m openjev.webapp --host 0.0.0.0 --port "$PORT" --model "$MODEL_DIR"
