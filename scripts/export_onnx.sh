#!/usr/bin/env bash
set -euo pipefail

python -u export_onnx.py \
  --ckpt runs/exp0/best.pt \
  --onnx_out surgical_planner.onnx \
  --img_size 512 512 --device cuda
