# OpenJev local decision-engine server
# build:  docker build -t openjev .
# run:    docker run -p 8771:8771 -v openjev-models:/models openjev
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image ~2 GB instead of ~8 GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY openjev ./openjev
COPY scripts ./scripts
RUN pip install --no-cache-dir .

# model cache volume; pre-download at first start, reuse afterwards
ENV HF_HUB_DISABLE_XET=1 \
    OPENJEV_MODEL=Qwen/Qwen3-0.6B \
    OPENJEV_DEVICE=cpu \
    OPENJEV_PORT=8771
VOLUME /models
ENV HF_HOME=/models/hf

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
EXPOSE 8771
ENTRYPOINT ["docker-entrypoint.sh"]
