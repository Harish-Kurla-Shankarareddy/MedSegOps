FROM python:3.12-slim

# ------------------------------------------------------------
# System configuration
# ------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# ------------------------------------------------------------
# System dependencies + security updates
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt


# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

COPY app /app/app
COPY models /app/models
COPY scripts /app/scripts
COPY pyproject.toml /app/pyproject.toml


# ------------------------------------------------------------
# Runtime directories
# ------------------------------------------------------------

RUN mkdir -p \
    /app/data/uploads \
    /app/data/dicom_uploads \
    /app/outputs


# ------------------------------------------------------------
# Port
# ------------------------------------------------------------

EXPOSE 8000


# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]