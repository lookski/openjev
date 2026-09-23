#!/bin/sh
# OpenJev docker entrypoint: download the model if missing, then serve.
# OPENJEV_BACKEND=jev skips the model download and proxies to the cloud API.
set -e
if [ "$OPENJEV_BACKEND" = "jev" ]; then
  exec python -m openjev.server \
    --backend jev \
    --host 0.0.0.0 \
    --port "$OPENJEV_PORT" \
    --preload "$@"
fi
MODEL_DIR="/models/$(basename "$OPENJEV_MODEL")"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[openjev] downloading $OPENJEV_MODEL into $MODEL_DIR ..."
  python scripts/download_model.py --model "$OPENJEV_MODEL" --out "$MODEL_DIR"
fi
exec python -m openjev.server \
  --host 0.0.0.0 \
  --port "$OPENJEV_PORT" \
  --model "$MODEL_DIR" \
  --device "$OPENJEV_DEVICE" \
  --preload "$@"
