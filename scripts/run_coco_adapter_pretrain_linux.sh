#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/models/DeformableDETR:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

COCO_PATH="${COCO_PATH:-$PROJECT_ROOT/data/coco}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train2017}"
VAL_SPLIT="${VAL_SPLIT:-val2017}"

MODEL_NAME="${MODEL_NAME:-dinov3_adapter_coco_class_agnostic}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/coco_adapter}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/coco_adapter}"
RUN_NAME="${RUN_NAME:-}"

GPU="${GPU:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-12}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-}"

DINOV3_MODEL_SIZE="${DINOV3_MODEL_SIZE:-base}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-$PROJECT_ROOT/src/models/backbones/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"

TRAIN_SCALES="${TRAIN_SCALES:-c1,c2,c3}"
SCALE_LOSS_WEIGHTS="${SCALE_LOSS_WEIGHTS:-c1:0.25,c2:0.5,c3:1.0}"
MAX_CANDIDATES="${MAX_CANDIDATES:-4096}"
MAX_CANDIDATES_PER_SCALE="${MAX_CANDIDATES_PER_SCALE:-c1:2048,c2:4096,c3:4096}"
MIN_BOX_SIZE="${MIN_BOX_SIZE:-2.0}"
MAX_BOXES_PER_IMAGE="${MAX_BOXES_PER_IMAGE:-0}"
HORIZONTAL_FLIP_P="${HORIZONTAL_FLIP_P:-0.5}"
COLOR_JITTER_P="${COLOR_JITTER_P:-0.2}"

LR="${LR:-0.0001}"
HEAD_LR="${HEAD_LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
AMP="${AMP:-0}"

BBOX_LOSS_COEF="${BBOX_LOSS_COEF:-1.0}"
GIOU_LOSS_COEF="${GIOU_LOSS_COEF:-2.0}"
CE_LOSS_COEF="${CE_LOSS_COEF:-2.0}"
COST_CLASS="${COST_CLASS:-2.0}"
COST_BBOX="${COST_BBOX:-1.0}"
COST_GIOU="${COST_GIOU:-2.0}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.5}"

SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.20}"
SCORE_RATIO="${SCORE_RATIO:-0.50}"
THRESHOLD_MODE="${THRESHOLD_MODE:-static_ratio}"
PRE_NMS_TOPK="${PRE_NMS_TOPK:-16000}"
MAX_DETECTIONS="${MAX_DETECTIONS:-300}"
NMS_IOU="${NMS_IOU:-0.5}"

AUTO_EXTRACT="${AUTO_EXTRACT:-0}"

require_file() {
  local path="$1"
  local message="$2"
  if [[ ! -f "$path" ]]; then
    echo "$message" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  local message="$2"
  if [[ ! -d "$path" ]]; then
    echo "$message" >&2
    return 1
  fi
}

extract_if_requested() {
  if [[ "$AUTO_EXTRACT" != "1" ]]; then
    return 0
  fi

  require_file "$COCO_PATH/train2017.zip" "Missing $COCO_PATH/train2017.zip" || exit 1
  require_file "$COCO_PATH/val2017.zip" "Missing $COCO_PATH/val2017.zip" || exit 1
  require_file "$COCO_PATH/annotations_trainval2017.zip" "Missing $COCO_PATH/annotations_trainval2017.zip" || exit 1

  if [[ ! -d "$COCO_PATH/train2017" ]]; then
    unzip -q "$COCO_PATH/train2017.zip" -d "$COCO_PATH"
  fi
  if [[ ! -d "$COCO_PATH/val2017" ]]; then
    unzip -q "$COCO_PATH/val2017.zip" -d "$COCO_PATH"
  fi
  if [[ ! -d "$COCO_PATH/annotations" ]]; then
    unzip -q "$COCO_PATH/annotations_trainval2017.zip" -d "$COCO_PATH"
  fi
}

print_extract_help() {
  cat >&2 <<EOF
COCO2017 is not extracted under:
  $COCO_PATH

Expected:
  $COCO_PATH/train2017/
  $COCO_PATH/val2017/
  $COCO_PATH/annotations/instances_train2017.json
  $COCO_PATH/annotations/instances_val2017.json

Extract manually:
  unzip -q "$COCO_PATH/train2017.zip" -d "$COCO_PATH"
  unzip -q "$COCO_PATH/val2017.zip" -d "$COCO_PATH"
  unzip -q "$COCO_PATH/annotations_trainval2017.zip" -d "$COCO_PATH"

Or let this script extract them:
  AUTO_EXTRACT=1 bash scripts/run_coco_adapter_pretrain_linux.sh
EOF
}

extract_if_requested

if [[ ! -d "$COCO_PATH/train2017" ]] \
  || [[ ! -d "$COCO_PATH/val2017" ]] \
  || [[ ! -f "$COCO_PATH/annotations/instances_train2017.json" ]] \
  || [[ ! -f "$COCO_PATH/annotations/instances_val2017.json" ]]; then
  print_extract_help
  exit 1
fi

if [[ ! -f "$DINOV3_WEIGHTS" ]]; then
  echo "DINOV3_WEIGHTS does not exist: $DINOV3_WEIGHTS" >&2
  exit 1
fi

mkdir -p "$MODEL_PATH" "$LOG_DIR"

cmd=(
  "$PYTHON_BIN" -u tools/train_coco_adapter.py
  --coco-path "$COCO_PATH"
  --train-split "$TRAIN_SPLIT"
  --val-split "$VAL_SPLIT"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --eval-interval "$EVAL_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --log-dir "$LOG_DIR"
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --gpu "$GPU"
  --dinov3-model-size "$DINOV3_MODEL_SIZE"
  --dinov3-pretrained-weights "$DINOV3_WEIGHTS"
  --train-scales "$TRAIN_SCALES"
  --scale-loss-weights "$SCALE_LOSS_WEIGHTS"
  --max-candidates "$MAX_CANDIDATES"
  --max-candidates-per-scale "$MAX_CANDIDATES_PER_SCALE"
  --min-box-size "$MIN_BOX_SIZE"
  --max-boxes-per-image "$MAX_BOXES_PER_IMAGE"
  --horizontal-flip-p "$HORIZONTAL_FLIP_P"
  --color-jitter-p "$COLOR_JITTER_P"
  --lr "$LR"
  --head-lr "$HEAD_LR"
  --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM"
  --bbox-loss-coef "$BBOX_LOSS_COEF"
  --giou-loss-coef "$GIOU_LOSS_COEF"
  --ce-loss-coef "$CE_LOSS_COEF"
  --cost-class "$COST_CLASS"
  --cost-bbox "$COST_BBOX"
  --cost-giou "$COST_GIOU"
  --focal-alpha "$FOCAL_ALPHA"
  --score-threshold "$SCORE_THRESHOLD"
  --score-ratio "$SCORE_RATIO"
  --threshold-mode "$THRESHOLD_MODE"
  --pre-nms-topk "$PRE_NMS_TOPK"
  --max-detections "$MAX_DETECTIONS"
  --nms-iou "$NMS_IOU"
)

if [[ -n "$RUN_NAME" ]]; then
  cmd+=(--run-name "$RUN_NAME")
fi

if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
  cmd+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi

if [[ -n "$MAX_VAL_BATCHES" ]]; then
  cmd+=(--max-val-batches "$MAX_VAL_BATCHES")
fi

if [[ -n "$MAX_TRAIN_IMAGES" ]]; then
  cmd+=(--max-train-images "$MAX_TRAIN_IMAGES")
fi

if [[ -n "$MAX_VAL_IMAGES" ]]; then
  cmd+=(--max-val-images "$MAX_VAL_IMAGES")
fi

if [[ "$AMP" == "1" ]]; then
  cmd+=(--amp)
else
  cmd+=(--no-amp)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'

"${cmd[@]}" "$@"
