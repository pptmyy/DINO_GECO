#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/models/DeformableDETR:${PYTHONPATH:-}"

CONFIG="${CONFIG:-$PROJECT_ROOT/configs/fourth_fsc147_epoch47.json}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/FSC147_384_V2}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/inference}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-$PROJECT_ROOT/src/models/backbones/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
COCO_MAX_DETS="${COCO_MAX_DETS:-1000}"

python infer.py \
  --config "$CONFIG" \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --dinov3_pretrained_weights "$DINOV3_WEIGHTS" \
  --coco-max-dets "$COCO_MAX_DETS" \
  "$@"
