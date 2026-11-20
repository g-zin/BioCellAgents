# BioCellAgents - Multi-stage Docker build
# Stage 1: Build & install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for building Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libraries required by OpenCV (cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Create non-root user and writable app directories
RUN useradd -m -r biocell \
    && mkdir -p /app/logs /app/checkpoints /app/data /app/reports \
    && chown -R biocell:biocell /app

# Copy application code
COPY --chown=biocell:biocell src/ src/
COPY --chown=biocell:biocell config/ config/
COPY --chown=biocell:biocell .streamlit/ .streamlit/
COPY --chown=biocell:biocell main.py app.py biocell_app.py ./

USER biocell

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
