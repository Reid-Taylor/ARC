#!/bin/bash
# filepath: scripts/setup_gcp_instance.sh

set -e

echo "🚀 Setting up ARC training environment on GCP..."

sudo apt-get update
sudo apt-get install -y git htop nvtop

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

git clone https://github.com/Reid-Taylor/ARC.git
cd ARC

mkdir data
cd data 
git clone https://github.com/fchollet/ARC-AGI.git
cd .. 

uv venv --python 3.13.11
source .venv/bin/activate

uv pip install -r pyproject.toml

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "✅ Setup complete! Ready for training."