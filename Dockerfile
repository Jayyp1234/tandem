# TANDEM API — single-container build for BCT submission.
# Builds both endpoints (Task A simulator + Task B recommender) in one FastAPI app.
#
# Usage:
#   docker compose up   (preferred — reads .env for GROQ_API_KEY)
#   docker build -t tandem . && docker run -p 8000:8000 -e GROQ_API_KEY=$GROQ_API_KEY tandem

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# uv = fast resolver/installer
RUN pip install --no-cache-dir uv

WORKDIR /app

# Layer 1: dependency manifests only (good cache reuse on code changes)
COPY pyproject.toml README.md ./

# Layer 2: install deps system-wide (no venv inside the image)
RUN uv pip install --system --no-cache .

# Layer 3: copy source code
COPY src/ ./src/

# Layer 4: copy the response cache so judges can reproduce results offline
# (cache/ is shipped with the submission tarball; gitignored from the repo)
COPY cache/ ./cache/

# Layer 5: optional — preprocessed data for offline operation
COPY data/ ./data/

EXPOSE 8000

# Healthcheck so docker compose can detect readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as r; r.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
