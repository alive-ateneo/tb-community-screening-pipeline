#!/usr/bin/env bash
# Train the from-scratch detection baselines sequentially: yolo11n then yolov8n.
#
# Safe to re-run after an interrupt (reboot/crash). Each model's run.sh invokes
# mlx, which auto-detects results/<model>/weights/last.pt and resumes from the
# last saved epoch (Ultralytics writes last.pt/best.pt every epoch). A model
# that already finished 100/100 epochs cannot be resumed: Ultralytics reports
# "nothing to resume" and this loop continues to the next model.
#
# Usage (detached, survives terminal close):
#   nohup ./run_all.sh > train_all.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/../.."

source "$REPO/code/_mlx/.venv/bin/activate"
export TMPDIR="${TMPDIR:-/home/ase/.cache/pip-tmp}"

for M in yolo11n yolov8n; do
  echo "===== TRAIN $M start $(date -Is) ====="
  ( cd "$HERE/$M" && bash run.sh )
  echo "===== TRAIN $M end exit=$? $(date -Is) ====="
done
echo "ALL_TRAIN_DONE"
