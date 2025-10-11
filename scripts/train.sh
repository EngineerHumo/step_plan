#!/usr/bin/env bash
set -euo pipefail

python -u train.py \
  --train_items path/to/train_items.json \
  --val_items path/to/val_items.json \
  --epochs 100 --batch_size 4 --lr 3e-4 \
  --img_size 512 512 --num_queries 4 \
  --save_dir runs/exp0
