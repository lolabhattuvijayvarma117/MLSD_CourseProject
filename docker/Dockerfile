# Use Python 3.12-slim as base image to match modern developer environment
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory in container
WORKDIR /app

# Install critical system tools (including curl for container health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching optimization
COPY requirements.txt .

# Install pip dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy service and model directories
COPY fraud_detection_service/ fraud_detection_service/
COPY models/ models/
COPY params.yaml .
COPY setup.py .

# Expose FastAPI listening port
EXPOSE 8000

# Define container health monitoring
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI app serving engine
CMD ["uvicorn", "fraud_detection_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
