#!/usr/bin/env bash
set -euo pipefail

python -u train.py \
  --train_items data/surgical_plan/items/train_items.json \
  --val_items data/surgical_plan/items/val_items.json \
  --epochs 400 --batch_size 3 --lr 1e-3 \
  --img_size 1024 1024 --num_queries 6 \
  --save_dir runs/exp0 \
  --visdom
