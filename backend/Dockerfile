FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./

# CPU inference stack for API
RUN uv pip install --system --no-cache \
    "fastapi" \
    "uvicorn[standard]" \
    "python-multipart" \
    "python-dotenv" \
    "pillow" \
    "numpy" \
    "torch" \
    "torchvision" \
    "tqdm"

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p models/checkpoints data/metadata outputs logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Checkpoint via volume (local) or MODEL_URL download (Render)
CMD ["sh", "-c", "python -m scripts.download_model_if_needed && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
