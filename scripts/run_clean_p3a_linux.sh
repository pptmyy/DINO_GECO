#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/models/DeformableDETR:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Clean P3-A: keep the P3 training/postprocess protocol and only enable semantic anchor.
MODEL_NAME="${MODEL_NAME:-DGECO2FSCD_P3A_CLEAN}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/FSC147_384_V2}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/p3a_clean}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
RUN_NAME="${RUN_NAME:-p3a_clean_semantic_anchor}"

GPU="${GPU:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
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
NMS_METHOD="${NMS_METHOD:-hard}"
NMS_IOU="${NMS_IOU:-0.30}"
MIN_BOX_AREA="${MIN_BOX_AREA:-0.0}"
MAX_BOX_AREA="${MAX_BOX_AREA:-0.0}"
ADAPTIVE_SPARSE_SCORE_RATIO="${ADAPTIVE_SPARSE_SCORE_RATIO:-0.50}"
ADAPTIVE_DENSE_SCORE_RATIO="${ADAPTIVE_DENSE_SCORE_RATIO:-0.45}"
ADAPTIVE_SPARSE_NMS_IOU="${ADAPTIVE_SPARSE_NMS_IOU:-0.25}"
ADAPTIVE_DENSE_NMS_IOU="${ADAPTIVE_DENSE_NMS_IOU:-0.25}"
ADAPTIVE_DENSE_CANDIDATE_THRESHOLD="${ADAPTIVE_DENSE_CANDIDATE_THRESHOLD:-128}"

# Default to stride-4 main detection; stride-2 is used by the refinement branch.
QUERY_OUTPUT_STRIDE="${QUERY_OUTPUT_STRIDE:-4}"
STRIDE2_REFINEMENT="${STRIDE2_REFINEMENT:-1}"
CENTER_GAUSSIAN_HEAD="${CENTER_GAUSSIAN_HEAD:-1}"
CENTER_GAUSSIAN_LOSS_COEF="${CENTER_GAUSSIAN_LOSS_COEF:-1.0}"
CENTER_GAUSSIAN_SIGMA="${CENTER_GAUSSIAN_SIGMA:-2.0}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-4}"
PROTOTYPE_PRED_TOPK="${PROTOTYPE_PRED_TOPK:-32}"
PROTOTYPE_PRED_SCORE_THRESHOLD="${PROTOTYPE_PRED_SCORE_THRESHOLD:-0.5}"
PROTOTYPE_EMA_MOMENTUM="${PROTOTYPE_EMA_MOMENTUM:-0.9}"
MUTUAL_ADAPTER_LAYERS="${MUTUAL_ADAPTER_LAYERS:-1}"
DECOUPLED_HEADS="${DECOUPLED_HEADS:-1}"

VERIFICATION_MODE="${VERIFICATION_MODE:-none}"
VERIFICATION_THRESHOLD="${VERIFICATION_THRESHOLD:-0.0}"
VERIFICATION_TOPK="${VERIFICATION_TOPK:-0}"
VERIFICATION_MIN_AREA_RATIO="${VERIFICATION_MIN_AREA_RATIO:-0.0}"
VERIFICATION_MAX_AREA_RATIO="${VERIFICATION_MAX_AREA_RATIO:-0.0}"
VERIFICATION_FILTER_MODE="${VERIFICATION_FILTER_MODE:-hard}"
VERIFICATION_SCORE_GAMMA="${VERIFICATION_SCORE_GAMMA:-0.0}"
VERIFICATION_HARD_CANDIDATE_LIMIT="${VERIFICATION_HARD_CANDIDATE_LIMIT:-0}"
VAL_IOU_THRESHOLD="${VAL_IOU_THRESHOLD:-0.5}"
DETECTION_METRIC_INTERVAL="${DETECTION_METRIC_INTERVAL:-1}"
DETECTION_GATE_RATIO="${DETECTION_GATE_RATIO:-0.98}"

BBOX_LOSS_COEF="${BBOX_LOSS_COEF:-1.0}"
GIOU_LOSS_COEF="${GIOU_LOSS_COEF:-2.0}"
CE_LOSS_COEF="${CE_LOSS_COEF:-2.0}"
AUX_LOSS_COEF="${AUX_LOSS_COEF:-0.5}"
COST_CLASS="${COST_CLASS:-2.0}"
COST_BBOX="${COST_BBOX:-1.0}"
COST_GIOU="${COST_GIOU:-2.0}"
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
  "$PYTHON_BIN" -u train.py
  --model_name "$MODEL_NAME"
  --data_path "$DATA_PATH"
  --dataset fsc147
  --image_size "$IMAGE_SIZE"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --val_batch_size "$VAL_BATCH_SIZE"
  --test_batch_size "$TEST_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --log_dir "$LOG_DIR"
  --run_name "$RUN_NAME"
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
  --nms-method "$NMS_METHOD"
  --nms-iou "$NMS_IOU"
  --min-box-area "$MIN_BOX_AREA"
  --max-box-area "$MAX_BOX_AREA"
  --adaptive-sparse-score-ratio "$ADAPTIVE_SPARSE_SCORE_RATIO"
  --adaptive-dense-score-ratio "$ADAPTIVE_DENSE_SCORE_RATIO"
  --adaptive-sparse-nms-iou "$ADAPTIVE_SPARSE_NMS_IOU"
  --adaptive-dense-nms-iou "$ADAPTIVE_DENSE_NMS_IOU"
  --adaptive-dense-candidate-threshold "$ADAPTIVE_DENSE_CANDIDATE_THRESHOLD"
  --query-output-stride "$QUERY_OUTPUT_STRIDE"
  --center-gaussian-loss-coef "$CENTER_GAUSSIAN_LOSS_COEF"
  --center-gaussian-sigma "$CENTER_GAUSSIAN_SIGMA"
  --num-prototypes "$NUM_PROTOTYPES"
  --prototype-pred-topk "$PROTOTYPE_PRED_TOPK"
  --prototype-pred-score-threshold "$PROTOTYPE_PRED_SCORE_THRESHOLD"
  --prototype-ema-momentum "$PROTOTYPE_EMA_MOMENTUM"
  --mutual-adapter-layers "$MUTUAL_ADAPTER_LAYERS"
  --use-semantic-anchor
  --verification-mode "$VERIFICATION_MODE"
  --verification-threshold "$VERIFICATION_THRESHOLD"
  --verification-topk "$VERIFICATION_TOPK"
  --verification-min-area-ratio "$VERIFICATION_MIN_AREA_RATIO"
  --verification-max-area-ratio "$VERIFICATION_MAX_AREA_RATIO"
  --verification-filter-mode "$VERIFICATION_FILTER_MODE"
  --verification-score-gamma "$VERIFICATION_SCORE_GAMMA"
  --verification-hard-candidate-limit "$VERIFICATION_HARD_CANDIDATE_LIMIT"
  --val-iou-threshold "$VAL_IOU_THRESHOLD"
  --detection-metric-interval "$DETECTION_METRIC_INTERVAL"
  --detection-gate-ratio "$DETECTION_GATE_RATIO"
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

if [[ "$RESUME_TRAINING" == "1" ]]; then
  cmd+=(--resume_training)
fi

if [[ "$ZERO_SHOT" == "1" ]]; then
  cmd+=(--zero_shot)
fi

if [[ "$STRIDE2_REFINEMENT" != "1" ]]; then
  cmd+=(--no-stride2-refinement)
fi

if [[ "$CENTER_GAUSSIAN_HEAD" != "1" ]]; then
  cmd+=(--no-center-gaussian-head)
fi

if [[ "$DECOUPLED_HEADS" != "1" ]]; then
  cmd+=(--no-decoupled-heads)
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

cat <<EOF
Clean P3-A semantic-anchor experiment
  model_name:          $MODEL_NAME
  model_path:          $MODEL_PATH
  log_dir:             $LOG_DIR
  data_path:           $DATA_PATH
  use_semantic_anchor: true
  query_output_stride: $QUERY_OUTPUT_STRIDE
  stride2_refinement:  $STRIDE2_REFINEMENT
  center_loss_coef:    $CENTER_GAUSSIAN_LOSS_COEF
  num_prototypes:      $NUM_PROTOTYPES
  pre_nms_topk:        $PRE_NMS_TOPK
  max_detections:      $MAX_DETECTIONS
  nms_method:          $NMS_METHOD
  nms_iou:             $NMS_IOU
  verification_mode:   $VERIFICATION_MODE
EOF

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'

"${cmd[@]}" "$@"
