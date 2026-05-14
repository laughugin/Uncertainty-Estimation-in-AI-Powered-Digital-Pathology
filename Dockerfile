# Uncertainty Estimation in AI-Powered Digital Pathology
# Python 3.12, full project + web UI. Data can be mounted or downloaded via UI.
FROM python:3.12-slim

WORKDIR /app

# System deps (optional, for some torch builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Project deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code (excluding data/venv via .dockerignore)
COPY . .

RUN pip install --no-cache-dir -e .

# Web UI on 5000; bind all interfaces for Docker
ENV FLASK_APP=web.app
ENV HOST=0.0.0.0
ENV PORT=5000
EXPOSE 5000

# Run web app (data/model download available via UI or mount volumes)
CMD ["python", "-m", "flask", "run", "--host", "0.0.0.0", "--port", "5000"]
