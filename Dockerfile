# Build the React dashboard once, then serve it through FastAPI in production.
FROM node:22-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Full product image: served forecast, PPO, and GAT endpoints need the ML extra.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY quantmind ./quantmind
RUN pip install --upgrade pip && pip install -e ".[ml]"

COPY web ./web
COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "quantmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
