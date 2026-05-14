#!/usr/bin/env bash
# Create virtual environment and install dependencies for the project.
# On Debian/Ubuntu, install venv first: sudo apt install python3-venv
set -e
cd "$(dirname "$0")"

if ! python3 -m venv --help &>/dev/null; then
  echo "python3-venv not found. Install it with:"
  echo "  sudo apt install python3-venv"
  echo "Or install dependencies in the current environment:"
  echo "  pip install -r requirements.txt"
  exit 1
fi

echo "Creating virtual environment in ./venv ..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip ..."
pip install -U pip

echo "Installing requirements ..."
pip install -r requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  source venv/bin/activate"
echo ""
echo "Then download data:"
echo "  python data/download_datasets.py --root data/raw --dataset pcam"
echo ""
echo "Download/cache model (optional, will also happen on first run):"
echo "  python models/load_model.py --model_id google/vit-base-patch16-224"
