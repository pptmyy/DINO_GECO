#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/models/DeformableDETR:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-DGECO2FSCD}"
MODEL_NAME_RESUMED="${MODEL_NAME_RESUMED:-DGECO2FSCD}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/FSC147_384_V2}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
RUN_NAME="${RUN_NAME:-}"

GPU="${GPU:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-$PROJECT_ROOT/src/models/backbones/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
DINOV3_MODEL_SIZE="${DINOV3_MODEL_SIZE:-base}"
BACKBONE="${BACKBONE:-DINOV3}"
REDUCTION="${REDUCTION:-16}"
KERNEL_DIM="${KERNEL_DIM:-1}"
NUM_OBJECTS="${NUM_OBJECTS:-3}"

EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.00005}"
BACKBONE_LR="${BACKBONE_LR:-0.000005}"
LR_DROP="${LR_DROP:-20}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
TILING_P="${TILING_P:-0.2}"

SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.20}"
SCORE_RATIO="${SCORE_RATIO:-0.50}"
THRESHOLD_MODE="${THRESHOLD_MODE:-static_ratio}"
SCORE_QUANTILE="${SCORE_QUANTILE:-0.98}"
MIN_SCORE_GAP="${MIN_SCORE_GAP:-0.0}"
PRE_NMS_TOPK="${PRE_NMS_TOPK:-4096}"
MAX_DETECTIONS="${MAX_DETECTIONS:-4096}"
NMS_IOU="${NMS_IOU:-0.3}"
MIN_BOX_AREA="${MIN_BOX_AREA:-0.0}"
MAX_BOX_AREA="${MAX_BOX_AREA:-0.0}"
ADAPTIVE_SPARSE_SCORE_RATIO="${ADAPTIVE_SPARSE_SCORE_RATIO:-0.50}"
ADAPTIVE_DENSE_SCORE_RATIO="${ADAPTIVE_DENSE_SCORE_RATIO:-0.45}"
ADAPTIVE_SPARSE_NMS_IOU="${ADAPTIVE_SPARSE_NMS_IOU:-0.25}"
ADAPTIVE_DENSE_NMS_IOU="${ADAPTIVE_DENSE_NMS_IOU:-0.25}"
ADAPTIVE_DENSE_CANDIDATE_THRESHOLD="${ADAPTIVE_DENSE_CANDIDATE_THRESHOLD:-128}"
VERIFICATION_MODE="${VERIFICATION_MODE:-none}"
VERIFICATION_THRESHOLD="${VERIFICATION_THRESHOLD:-0.0}"
VERIFICATION_TOPK="${VERIFICATION_TOPK:-0}"
VERIFICATION_MIN_AREA_RATIO="${VERIFICATION_MIN_AREA_RATIO:-0.0}"
VERIFICATION_MAX_AREA_RATIO="${VERIFICATION_MAX_AREA_RATIO:-0.0}"
VERIFICATION_FILTER_MODE="${VERIFICATION_FILTER_MODE:-hard}"
VERIFICATION_SCORE_GAMMA="${VERIFICATION_SCORE_GAMMA:-0.0}"
VERIFICATION_HARD_CANDIDATE_LIMIT="${VERIFICATION_HARD_CANDIDATE_LIMIT:-0}"

GIOU_LOSS_COEF="${GIOU_LOSS_COEF:-2}"
BBOX_LOSS_COEF="${BBOX_LOSS_COEF:-1}"
CE_LOSS_COEF="${CE_LOSS_COEF:-2}"
AUX_LOSS_COEF="${AUX_LOSS_COEF:-0.5}"
COST_CLASS="${COST_CLASS:-2}"
COST_BBOX="${COST_BBOX:-1}"
COST_GIOU="${COST_GIOU:-2}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.5}"

MODEL_NAME_RESUME_FROM="${MODEL_NAME_RESUME_FROM:-base_3_shot_softmax1}"
RESUME_TRAINING="${RESUME_TRAINING:-0}"
ZERO_SHOT="${ZERO_SHOT:-0}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "DATA_PATH does not exist: $DATA_PATH" >&2
  exit 1
fi

if [[ ! -f "$DINOV3_WEIGHTS" ]]; then
  echo "DINOV3_WEIGHTS does not exist: $DINOV3_WEIGHTS" >&2
  exit 1
fi

mkdir -p "$MODEL_PATH" "$LOG_DIR"

cmd=(
  python -u train.py
  --model_name "$MODEL_NAME"
  --model_name_resumed "$MODEL_NAME_RESUMED"
  --data_path "$DATA_PATH"
  --dataset fsc147
  --image_size "$IMAGE_SIZE"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --log_dir "$LOG_DIR"
  --log_interval "$LOG_INTERVAL"
  --model_path "$MODEL_PATH"
  --gpu "$GPU"
  --max_grad_norm "$MAX_GRAD_NORM"
  --score-threshold "$SCORE_THRESHOLD"
  --score-ratio "$SCORE_RATIO"
  --threshold-mode "$THRESHOLD_MODE"
  --score-quantile "$SCORE_QUANTILE"
  --min-score-gap "$MIN_SCORE_GAP"
  --pre-nms-topk "$PRE_NMS_TOPK"
  --max-detections "$MAX_DETECTIONS"
  --nms-iou "$NMS_IOU"
  --min-box-area "$MIN_BOX_AREA"
  --max-box-area "$MAX_BOX_AREA"
  --adaptive-sparse-score-ratio "$ADAPTIVE_SPARSE_SCORE_RATIO"
  --adaptive-dense-score-ratio "$ADAPTIVE_DENSE_SCORE_RATIO"
  --adaptive-sparse-nms-iou "$ADAPTIVE_SPARSE_NMS_IOU"
  --adaptive-dense-nms-iou "$ADAPTIVE_DENSE_NMS_IOU"
  --adaptive-dense-candidate-threshold "$ADAPTIVE_DENSE_CANDIDATE_THRESHOLD"
  --verification-mode "$VERIFICATION_MODE"
  --verification-threshold "$VERIFICATION_THRESHOLD"
  --verification-topk "$VERIFICATION_TOPK"
  --verification-min-area-ratio "$VERIFICATION_MIN_AREA_RATIO"
  --verification-max-area-ratio "$VERIFICATION_MAX_AREA_RATIO"
  --verification-filter-mode "$VERIFICATION_FILTER_MODE"
  --verification-score-gamma "$VERIFICATION_SCORE_GAMMA"
  --verification-hard-candidate-limit "$VERIFICATION_HARD_CANDIDATE_LIMIT"
  --dinov3_model_size "$DINOV3_MODEL_SIZE"
  --dinov3_pretrained_weights "$DINOV3_WEIGHTS"
  --backbone "$BACKBONE"
  --reduction "$REDUCTION"
  --kernel_dim "$KERNEL_DIM"
  --num_objects "$NUM_OBJECTS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --backbone_lr "$BACKBONE_LR"
  --lr_drop "$LR_DROP"
  --weight_decay "$WEIGHT_DECAY"
  --tiling_p "$TILING_P"
  --bbox_loss_coef "$BBOX_LOSS_COEF"
  --giou_loss_coef "$GIOU_LOSS_COEF"
  --ce_loss_coef "$CE_LOSS_COEF"
  --aux_loss_coef "$AUX_LOSS_COEF"
  --cost_class "$COST_CLASS"
  --cost_bbox "$COST_BBOX"
  --cost_giou "$COST_GIOU"
  --focal_alpha "$FOCAL_ALPHA"
  --model_name_resume_from "$MODEL_NAME_RESUME_FROM"
)

if [[ -n "$RUN_NAME" ]]; then
  cmd+=(--run_name "$RUN_NAME")
fi

if [[ -n "$VAL_BATCH_SIZE" ]]; then
  cmd+=(--val_batch_size "$VAL_BATCH_SIZE")
fi

if [[ -n "$TEST_BATCH_SIZE" ]]; then
  cmd+=(--test_batch_size "$TEST_BATCH_SIZE")
fi

if [[ "$RESUME_TRAINING" == "1" ]]; then
  cmd+=(--resume_training)
fi

if [[ "$ZERO_SHOT" == "1" ]]; then
  cmd+=(--zero_shot)
fi

if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
  cmd+=(--max_train_batches "$MAX_TRAIN_BATCHES")
fi

if [[ -n "$MAX_VAL_BATCHES" ]]; then
  cmd+=(--max_val_batches "$MAX_VAL_BATCHES")
fi

if [[ -n "$MAX_TEST_BATCHES" ]]; then
  cmd+=(--max_test_batches "$MAX_TEST_BATCHES")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'

"${cmd[@]}" "$@"
