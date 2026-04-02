# ── Stage 1: base image ────────────────────────────────────────────────────
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY app/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model class definition
COPY app/model/ /app/model/

# Copy API source
COPY app/api/main.py /app/main.py

# Copy model checkpoint (built into the image)
COPY saved_models/lstm_cnn_best_model.pth /app/saved_models/lstm_cnn_best_model.pth

# Environment
ENV MODEL_PATH=/app/saved_models/lstm_cnn_best_model.pth
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
