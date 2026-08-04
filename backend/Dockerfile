FROM python:3.11-slim

WORKDIR /app

# System deps for pillow / opencv-ish builds if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project metadata first for better layer caching
COPY pyproject.toml ./
# Optional lockfile if present
COPY uv.lock* ./

# Install runtime deps (API + inference stack)
# CPU torch is enough for Render free/starter tiers
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

# Application code
COPY app ./app
COPY scripts ./scripts

# Model checkpoint directory (mount or copy at build/runtime)
RUN mkdir -p models/checkpoints data/metadata outputs logs

# Optional: copy checkpoint if committed outside gitignore for deploy builds
# Prefer downloading via MODEL_URL env at startup for large files.
COPY models/checkpoints* ./models/checkpoints/
COPY data/metadata* ./data/metadata/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Start API; MODEL_URL can download checkpoint on boot if missing
CMD ["sh", "-c", "python -m scripts.download_model_if_needed && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
