# ╔══════════════════════════════════════════════════════╗
# ║  Sentinel-Detect  |  Dockerfile                      ║
# ║  Multi-stage build on shared python:3.11-slim base   ║
# ║  Final image < 350 MB · runs in 1 GB RAM             ║
# ╚══════════════════════════════════════════════════════╝

# ── STAGE 1: builder ─────────────────────────────────────
# Installs all Python deps into an isolated prefix.
# Nothing from this stage leaks into the final image.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed only to compile certain wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first → Docker layer cache
COPY requirements.txt .

# Install into /install — not the system site-packages
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── STAGE 2: runtime ─────────────────────────────────────
# Starts from the same slim base → shared layer on disk.
# Copies only the compiled packages from the builder stage.
FROM python:3.11-slim AS runtime

LABEL maintainer="sentinel-detect"
LABEL description="AI-powered employee behaviour anomaly detection"

# Install curl for healthcheck + non-root user
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appuser . .

# Copy and set entrypoint
COPY --chown=appuser:appuser entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create required directories
RUN mkdir -p logs data \
    && chown -R appuser:appuser /app

USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT_API=8000 \
    PORT_UI=8501

EXPOSE 8000 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]