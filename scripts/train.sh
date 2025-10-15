#!/usr/bin/env bash
set -euo pipefail

python -u train.py \
  --train_items data/surgical_plan/items/train_items.json \
  --val_items data/surgical_plan/items/val_items.json \
  --epochs 400 --batch_size 4 --lr 3e-4 \
  --img_size 1024 1024 --num_queries 4 \
  --save_dir runs/exp0 \
  --visdom
