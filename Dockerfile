# ── Stage 1: build CSS ────────────────────────────────────────────────────────
FROM node:20-slim AS css-builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev
COPY frontend/ ./frontend/
COPY src/evidenceengine/templates/ ./src/evidenceengine/templates/
RUN npm run build:css

# ── Stage 2: Python application ───────────────────────────────────────────────
FROM python:3.13-slim AS app

# Non-root user
RUN groupadd --system ee && useradd --system --gid ee ee

WORKDIR /app

# Install Python deps from requirements.txt (reproducible)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY --from=css-builder /app/src/evidenceengine/static/css/app.css \
     ./src/evidenceengine/static/css/app.css

# Pre-built static assets (fonts, vendor JS)
# These are committed to the repo and available in the build context.
COPY src/evidenceengine/static/ ./src/evidenceengine/static/

# Ensure upload/index directories exist with correct ownership
RUN mkdir -p /app/uploads /app/indexes && chown -R ee:ee /app

USER ee

EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "evidenceengine.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
