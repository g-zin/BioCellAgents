# BioCellAgents - Multi-stage Docker build
# Stage 1: Build & install dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m -r biocell && mkdir -p /app/logs /app/checkpoints /app/data
USER biocell

# Copy application code
COPY src/ src/
COPY config/ config/
COPY main.py app.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV BIOMED_LOG_LEVEL=INFO

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default: run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]