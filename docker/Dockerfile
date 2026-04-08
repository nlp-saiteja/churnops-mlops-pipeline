# Use official Python slim image — smaller size, faster builds
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker caches this layer)
# If requirements.txt doesn't change, Docker skips this step on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories the app needs at runtime
RUN mkdir -p data/processed models mlruns

# Expose the port FastAPI will run on
EXPOSE 8000

# Health check — Docker will ping this every 30s to verify container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
