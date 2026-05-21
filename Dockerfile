FROM python:3.12-slim

WORKDIR /app

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY configs/       configs/
COPY data/          data/
COPY experiments/   experiments/
COPY models/        models/
COPY scripts/       scripts/
COPY uncertainty_lab/ uncertainty_lab/
COPY web/           web/

# Install the uncertainty_lab package in editable mode
RUN pip install --no-cache-dir -e .

# Create directories that the app writes to (volumes override these at runtime)
RUN mkdir -p evaluation/figures checkpoints data/raw

ENV REPO_ROOT=/app
EXPOSE 5000

# Run the web UI bound on all interfaces
CMD ["python", "web/app.py", "--host", "0.0.0.0", "--port", "5000"]
