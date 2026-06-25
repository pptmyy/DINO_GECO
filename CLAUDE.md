# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DINO-SAM-GECO2 is a research codebase for few-shot object counting and detection on the FSC147 dataset. An image plus a few exemplar boxes go in; counted/detected object boxes come out. It combines a frozen DINOv3 backbone, exemplar-conditioned prototype generation (GECO-style), dense box prediction, and optional SAM3 mask refinement.

## Commands

Environment (Python 3.12, CUDA PyTorch):

```bash
conda env create -f environment.yml && conda activate dino-sam-geco2
# or: pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/src:$PWD/src/models/DeformableDETR:${PYTHONPATH:-}"
```

Tests (smoke tests only; they mock DINOv3/query-generator/post-processing so no weights or data are needed):

```bash
python -m pytest test_dsgeco.py test_postprocess_verification.py
# single test: python -m pytest test_dsgeco.py -k <name>
```

Training (writes logs/<run_name>/{train.log,metrics.jsonl,args.json} and checkpoints/<model_name>*.pth):

```bash
bash scripts/run_train_linux_new.sh   # env overrides: DATA_PATH, MODEL_PATH, LOG_DIR, GPU, BATCH_SIZE, EPOCHS
# or directly:
python train.py --data_path data/FSC147_384_V2 --model_path checkpoints \
  --dinov3_pretrained_weights src/models/backbones/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --model_name DGECO2FSCD --batch_size 2 --epochs 200
```

Inference (split-level, single-image with `--image --boxes "x1,y1,x2,y2;..."`, optional `--sam3-mask`):

```bash
python infer.py --config configs/fourth_fsc147_epoch47.json \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --split val --output-dir outputs/inference_val --save-vis
```

Evaluation and post-processing sweeps:

```bash
python tools/eval_checkpoint.py --config configs/fourth_fsc147_epoch47.json \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --split val --output-dir outputs/eval_runs/baseline_val

python tools/sweep_postprocess.py --config configs/fourth_fsc147_epoch47.json \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --split val --output-dir outputs/eval_runs/sweep_val \
  --score-thresholds 0.20,0.25,0.30 --nms-ious 0.25,0.30 \
  --threshold-modes static_ratio,regime_adaptive
```

## Configuration Flow

All entry points share `arg_parser.py` (100+ args). Precedence: argparse defaults < JSON config passed via `--config` (see `configs/fourth_fsc147_epoch47.json`) < explicit CLI flags. Training snapshots the resolved config to `logs/<run_name>/args.json` — use that to reproduce a run.

## Architecture

Forward path of the main model (`src/models/DGECO.py`, class `DGECO`, built by `build_model()`):

1. **Backbone** — `DINOv3Adapter` (`src/models/backbones/dinov3_adapter.py`) wraps a frozen DINOv3 ViT (`src/models/backbones/dinov3/`) and emits dense `vision_features`, multi-scale `backbone_fpn` (L1/L2), and positional encodings. Pretrained weights live at `src/models/backbones/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
2. **Prototypes** — exemplar boxes are ROI-aligned and fused with learned geometry embeddings (9-dim shape/scale descriptors) to produce per-scale prototype embeddings.
3. **Query adaptation** — `QueryGenerator` (`src/models/query_generator.py`) runs prototype attention (`src/models/transformer.py`) and deformable image attention (`MSDeformAttn` from `src/models/DeformableDETR/`), then scale aggregation (`src/models/scale_query_aggregator.py`).
4. **Heads** — centerness logits + normalized boxes on a dense grid, plus an auxiliary branch weighted toward small objects.
5. **Loss** — `SetCriterion` (`src/utils/losses.py`) with Hungarian matching (`src/models/matcher.py`): focal classification, L1 + GIoU box losses.
6. **Post-processing** — `filter_detections` (`src/utils/postprocess.py`): score threshold, NMS, threshold modes (`static_ratio`/`quantile`/`regime_adaptive`); optional `verify_detections` (`src/utils/verification.py`) by exemplar geometry or feature similarity; validation-time box refinement in `src/models/box_corr.py`.
7. **SAM3 (optional)** — `src/models/sam_mask.py` / `sam_utils.py` generate instance masks from predicted boxes; SAM3 code is vendored at `src/models/backbones/sam3/` (not a git submodule) and expects weights at `src/models/backbones/sam3/checkpoints/sam3.pt`.

**Data** — `FSC147DATASET` (`src/datasets/data.py`) loads images, exemplar boxes, density maps (count-preserving through resize/pad), optional COCO-style GT boxes, with tiling augmentation. Expected layout under `data/FSC147_384_V2/` is documented in README.md.

## Notes

- On this machine the working environment is the `ScientificResearch` conda env (`~/.conda/envs/ScientificResearch/python.exe`, Python 3.12, torch 2.10.0+cu130) — the `dino-sam-geco2` env from environment.yml does not exist locally, and the base `python` on PATH lacks the project dependencies.
- Large assets (`data/`, `checkpoints/`, `logs/`, `outputs/`, `*.pth`, `*.pt`) are local-only and should never be committed.
- Shell scripts target Linux; on this Windows machine use Git Bash or run the `python` entry points directly.
- Validation baseline (FSC147 val, best_val_rmse checkpoint): MAE 20.70 / RMSE 49.58 — re-run `tools/eval_checkpoint.py` after any change to preprocessing, model, or post-processing.
