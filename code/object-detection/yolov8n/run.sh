#!/usr/bin/env bash
# Reproduce YOLOv8n (object detection, Ultralytics via mlx).
# Trains from scratch (random init, no COCO pretraining), same protocol as YOLO26:
# 100 epochs, batch 16, 512x512. The bare model name resolves to yolov8.yaml at
# nano scale (Ultralytics "Assuming scale='n'"), matching the yolo26 convention.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX="$HERE/../../_mlx"
DATA_YAML="$HERE/../../../dataset/object-detection/dataset.yaml"
OUT="$HERE"
MODEL="yolov8n"

cd "$MLX"
# Ultralytics saves to <project>/<name>; project=$HERE and name=results lands the
# run artifacts directly in yolov8n/results/ (flat, matching the yolo26 layout).
python -m mlx --mode object_detection --action train \
  --dataset "$DATA_YAML" --model "$MODEL" --output "$OUT" \
  --run-name results --epochs 100 --batch-size 16 \
  --height 512 --width 512 --device cuda --seed 42 --cache
