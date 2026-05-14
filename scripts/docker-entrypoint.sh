#!/bin/sh
# Optional entrypoint: run one-time setup then start web app.
# Use in Dockerfile: ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
# Or keep default CMD and use UI "Run" page to download data/model.

set -e
cd /app

# Optional: uncomment to auto-download PCAM on first start (takes time and disk)
# python data/download_datasets.py --root data/raw --dataset pcam || true

exec python -m flask run --host 0.0.0.0 --port 5000
