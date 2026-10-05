#!/usr/bin/env bash
set -euo pipefail
# Use the platform's CUDA-enabled PyTorch (Colab/Kaggle already provide it).
# Standard NVIDIA Linux: install the CUDA PyTorch build for your driver first.
python -m pip install -e '.[dev,training]'
python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch and NVIDIA GPU required"; print(torch.cuda.get_device_name())'
python -m src.cli validate --config experiments/final/protocol.yaml
