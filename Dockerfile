# ==============================================================================
# DreamTrip AI — Production Multi-Agent Travel Planner Dockerfile
# ==============================================================================
# Base Image: Python 3.11 Slim (Debian-based, lightweight, highly compatible)
FROM python:3.11-slim

# ------------------------------------------------------------------------------
# 1. Environment Configurations
# ------------------------------------------------------------------------------
# - PYTHONUNBUFFERED=1: Flushes stdout & stderr immediately (real-time docker logs)
# - PYTHONDONTWRITEBYTECODE=1: Avoids generating .pyc bytecode files inside container
# - PORT=8000: Standardized HTTP application port
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# ------------------------------------------------------------------------------
# 2. Container Working Directory
# ------------------------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------------------------
# 3. Install System Dependencies
# ------------------------------------------------------------------------------
# Installs curl for healthchecks and ca-certificates for trusted SSL handshakes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 4. Dependency Installation with Docker Layer Caching
# ------------------------------------------------------------------------------
# Copy requirements first so dependency installation is cached unless requirements.txt changes
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# 5. Copy Application Source Code
# ------------------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------------------
# 6. Networking & Container Healthcheck
# ------------------------------------------------------------------------------
EXPOSE 8000

# Container healthcheck querying FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ------------------------------------------------------------------------------
# 7. Application Launch Command
# ------------------------------------------------------------------------------
# Runs FastAPI with Uvicorn on 0.0.0.0 (binding to all network interfaces inside container)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
