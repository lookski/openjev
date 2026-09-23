# Hugging Face Space (sdk: docker) for the OpenJev yin-yang detector.
# The Space exposes app_port 7860 publicly; model persists in /data.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY openjev ./openjev
COPY scripts ./scripts
RUN pip install --no-cache-dir .

ENV HF_HUB_DISABLE_XET=1 \
    OPENJEV_MODEL=Qwen/Qwen3-0.6B \
    HF_HOME=/data/hf \
    PORT=7860

# /data is the Space persistent storage mount when enabled
RUN mkdir -p /data/hf

COPY space/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 7860
ENTRYPOINT ["entrypoint.sh"]
