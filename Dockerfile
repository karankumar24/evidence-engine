# ── Stage 1: build CSS ────────────────────────────────────────────────────────
FROM node:20-slim AS css-builder
WORKDIR /app
COPY package.json package-lock.json* ./
# devDependencies include @tailwindcss/cli which is required for build:css.
# This stage is discarded after CSS is compiled, so the size cost is free.
RUN npm ci
COPY frontend/ ./frontend/
COPY src/evidenceengine/templates/ ./src/evidenceengine/templates/
RUN npx tailwindcss -i ./frontend/css/input.css -o ./src/evidenceengine/static/css/app.css --minify

# ── Stage 2: Python application ───────────────────────────────────────────────
FROM python:3.13-slim AS app

# System deps:
#   - curl       : used by the container HEALTHCHECK
#   - gosu       : drop privileges from root → ee in the entrypoint (after
#                  chown'ing the Fly volume mount, owned by root on first boot)
#   - libmagic1  : C lib backing the `python-magic` MIME sniffer used in
#                  ingestion/validation.py (upload route)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gosu libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user the app actually runs as.
# --no-create-home + --home-dir=/app so pwd.getpwuid() returns /app — without
# this, libs that resolve `~` via pwd (ignoring $HOME) tried to write to
# /home/ee which exists but is root-owned, producing PermissionError [Errno 13]
# in ~30% of prod runs (2026-04-20).
RUN groupadd --system ee && useradd --system --gid ee --no-create-home --home-dir /app ee

WORKDIR /app

# Install Python deps from requirements.txt (regenerated via `uv export`).
# --extra-index-url pytorch/cpu is needed because uv pins torch to the CPU-only
# wheel (torch==2.8.0+cpu) to avoid ~2GB of CUDA/NVIDIA transitive deps on
# Fly's CPU-only machines.
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application source + static assets (fonts, vendor JS, etc.)
COPY src/ ./src/
COPY eval/ ./eval/

# Alembic migrations — run at release time (see fly.toml [deploy])
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Overwrite the committed app.css with the freshly-built one from css-builder
COPY --from=css-builder /app/src/evidenceengine/static/css/app.css \
     ./src/evidenceengine/static/css/app.css

# Entrypoint drops to the ee user after fixing volume-mount ownership
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Fallback upload/index/cache dirs. HF cache is needed for the cross-encoder
# reranker download; NLTK data dir is needed for punkt_tab tokenizer. Without
# HF_HOME/NLTK_DATA set these default to $HOME which resolves to /home/ee on
# Fly (that dir does not exist — PermissionError).
# Pre-download the reranker model at build time so cold starts don't need
# HuggingFace network access. 44MB downloads in seconds during build.
# The bge-reranker-v2-m3 (2.27GB) is excluded — too large for the image;
# set RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2 in fly secrets to use this.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')" \
    && echo "Reranker model pre-downloaded OK"

# Pre-download dense retrieval model (all-MiniLM-L6-v2, 22MB) so cold starts
# don't hit HuggingFace network. Separate RUN layer so layer cache is only
# invalidated when this model changes.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && echo "Dense retrieval model pre-downloaded OK"

RUN mkdir -p /app/uploads /app/indexes /app/nltk_data \
    && chown -R ee:ee /app

# Pre-download NLTK punkt_tab at build time so first-use retrieval does not
# have to download it (would fail on read-only Fly rootfs if HOME misconfigured).
RUN pip install --no-cache-dir nltk \
    && python -m nltk.downloader -d /app/nltk_data punkt punkt_tab \
    && chown -R ee:ee /app/nltk_data

EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/app/uploads
ENV INDEX_DIR=/app/indexes
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV NLTK_DATA=/app/nltk_data
ENV HOME=/app

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

# Single worker keeps the BM25 index warm in memory; scale horizontally via
# `fly scale count N` instead of bumping workers inside one container.
CMD ["python", "-m", "uvicorn", "evidenceengine.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
