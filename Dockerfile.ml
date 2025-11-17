FROM python:3.11-slim

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts directory
COPY scripts/ ./scripts/

# Create directories for models and reports
RUN mkdir -p models reports

# Set environment variables
ENV PYTHONPATH=/app \
    JAVA_HOME=/usr/lib/jvm/default-java

# Keep container running; run ML scripts on demand via docker compose exec
CMD ["bash", "-c", "sleep infinity"]

