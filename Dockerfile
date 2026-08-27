FROM python:3.12-slim

# ============================================================
# Environment
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# ============================================================
# System dependencies and security updates
# ============================================================

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1 \
        git \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt


# ============================================================
# Application files
# ============================================================

COPY app /app/app
COPY models /app/models
COPY scripts /app/scripts
COPY pyproject.toml /app/pyproject.toml


# ============================================================
# Runtime directories
# ============================================================

RUN mkdir -p \
        /app/data/uploads \
        /app/data/dicom_uploads \
        /app/outputs \
    && groupadd --system medsegops \
    && useradd --system \
        --gid medsegops \
        --create-home \
        --home-dir /home/medsegops \
        medsegops \
    && chown -R medsegops:medsegops \
        /app/data \
        /app/outputs \
        /app/app \
        /home/medsegops


# ============================================================
# Run as non-root user
# ============================================================

USER medsegops


# ============================================================
# Network
# ============================================================

EXPOSE 8000


# ============================================================
# Docker health check
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"


# ============================================================
# Start FastAPI
# ============================================================

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]