# DINO_GECO
DINOv3为骨干深度学习网络，解决FSCD147数据集，少标注计数检测问题
# DINO-SAM-GECO2

DINO-SAM-GECO2 is a research codebase for few-shot object counting and detection on the FSC147 dataset. The model uses a frozen DINOv3 visual backbone to extract dense image features, builds object prototypes from a small set of exemplar boxes, predicts object boxes on a dense grid, and can optionally refine predictions with SAM3 masks.

This repository is intended for reproducible experiments, checkpoint evaluation, post-processing sweeps, and single-image or split-level inference.

## Highlights

- DINOv3-based feature extractor with lightweight trainable adapter layers.
- Exemplar-conditioned prototype generation from ROI-aligned object examples.
- Geometry-aware prototype embeddings that inject exemplar shape and scale priors.
- Dense box prediction head with centerness scores and auxiliary supervision.
- FSC147 data loader with resize/pad handling, density-map count preservation, COCO-style box loading, and tiling augmentation.
- Configurable post-processing: score thresholds, static ratio, quantile thresholding, adaptive dense/sparse regimes, NMS, and optional verification.
- Optional SAM3 instance-mask generation from predicted boxes.
- Training logs, checkpoint evaluation, per-image diagnostics, COCO bbox evaluation, and post-processing grid search tools.

## Project Structure

```text
.
|-- arg_parser.py                 # Shared experiment arguments
|-- train.py                      # Single-GPU training entry point
|-- infer.py                      # Dataset and single-image inference
|-- configs/                      # JSON experiment configs
|-- scripts/                      # Linux helper scripts
|-- tools/
|   |-- eval_checkpoint.py        # Split-level checkpoint diagnostics
|   |-- sweep_postprocess.py      # Post-processing/verification sweeps
|   `-- diagnostics.py            # Metrics and CSV/JSON writers
|-- src/
|   |-- datasets/data.py          # FSC147 dataset loader
|   |-- models/DGECO.py           # Main DGECO model
|   |-- models/backbones/         # DINOv3/SAM3 backbone adapters
|   `-- utils/                    # Losses, box ops, post-processing, verification
`-- tests                         # Smoke tests for model and post-processing behavior
```

## Method Overview

The model receives an image and a small number of exemplar boxes. DINOv3 produces dense visual features and multi-scale FPN-like features. Exemplar regions are extracted with `roi_align`, combined with learned geometry embeddings, and passed through a query-generation module that adapts dense image features to the target object category. The detection heads then predict foreground confidence and normalized boxes for each candidate location.

During training, Hungarian matching supervises class and box predictions using focal classification loss, L1 box loss, and GIoU loss. An auxiliary detection branch is weighted more strongly for small-object cases. During inference, predicted boxes are filtered by score, NMS, optional adaptive thresholding, and optional verification against exemplar geometry or feature similarity.

## Environment

The repository targets Python 3.12 with CUDA-enabled PyTorch. A Conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate dino-sam-geco2
```

Or install with pip inside an existing environment:

```bash
pip install -r requirements.txt
```

For Linux runs, make sure the repository root and Deformable DETR modules are on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PWD/src:$PWD/src/models/DeformableDETR:${PYTHONPATH:-}"
```

## Data and Weights

Expected FSC147 layout:

```text
data/FSC147_384_V2/
|-- annotations/
|   |-- Train_Test_Val_FSC_147.json
|   |-- annotation_FSC147_384.json
|   |-- instances_train.json
|   |-- instances_val.json
|   `-- instances_test.json
|-- images_384_VarV2/
`-- gt_density_map_adaptive_384_VarV2/
```

Required model weights:

```text
src/models/backbones/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth
```

Optional SAM3 mask refinement expects:

```text
src/models/backbones/sam3/checkpoints/sam3.pt
```

Large datasets, pretrained weights, checkpoints, logs, and generated outputs should normally be excluded from a public GitHub repository and provided through release assets, cloud storage, or documented download links.

## Training

Use the Linux helper script:

```bash
bash scripts/run_train_linux_new.sh
```

Useful environment overrides:

```bash
DATA_PATH=/path/to/FSC147_384_V2 \
MODEL_PATH=/path/to/checkpoints \
LOG_DIR=/path/to/logs \
GPU=0 \
BATCH_SIZE=2 \
EPOCHS=200 \
bash scripts/run_train_linux_new.sh
```

Or call Python directly:

```bash
python train.py \
  --data_path data/FSC147_384_V2 \
  --model_path checkpoints \
  --dinov3_pretrained_weights src/models/backbones/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --model_name DGECO2FSCD \
  --batch_size 2 \
  --epochs 200
```

Training writes:

- `logs/<run_name>/train.log`
- `logs/<run_name>/metrics.jsonl`
- `logs/<run_name>/args.json`
- `checkpoints/<model_name>.pth`
- `checkpoints/<model_name>_best_val_rmse.pth`
- `checkpoints/<model_name>_best_val_mae.pth`
- `checkpoints/<model_name>_last.pth`

## Inference

Run inference on a dataset split:

```bash
python infer.py \
  --config configs/fourth_fsc147_epoch47.json \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --split val \
  --output-dir outputs/inference_val \
  --save-vis
```

Run inference on one image with exemplar boxes:

```bash
python infer.py \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --image path/to/image.jpg \
  --boxes "x1,y1,x2,y2;x1,y1,x2,y2;x1,y1,x2,y2" \
  --output-dir outputs/single_image \
  --save-vis
```

Enable SAM3 mask refinement:

```bash
python infer.py \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --image path/to/image.jpg \
  --boxes "x1,y1,x2,y2;x1,y1,x2,y2;x1,y1,x2,y2" \
  --sam3-mask \
  --save-mask-png \
  --save-vis
```

Outputs include JSON predictions, optional visualization images, optional COCO-format detections, and optional mask PNG files.

## Evaluation

Evaluate a checkpoint with per-image diagnostics:

```bash
python tools/eval_checkpoint.py \
  --config configs/fourth_fsc147_epoch47.json \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --split val \
  --output-dir outputs/eval_runs/baseline_val
```

Run a post-processing sweep:

```bash
python tools/sweep_postprocess.py \
  --config configs/fourth_fsc147_epoch47.json \
  --checkpoint checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth \
  --split val \
  --output-dir outputs/eval_runs/sweep_val \
  --score-thresholds 0.20,0.25,0.30 \
  --nms-ious 0.25,0.30 \
  --threshold-modes static_ratio,regime_adaptive
```

## Current Validation Results

Local validation diagnostics on FSC147 `val` using `checkpoints/DGECO2FSCD/DGECO2FSCD_best_val_rmse.pth`:

| Setting | Images | MAE | RMSE | Signed Error |
| --- | ---: | ---: | ---: | ---: |
| Baseline post-processing | 1286 | 20.70 | 49.58 | -5.28 |
| Sweep best by MAE | 1286 | 20.53 | 50.35 | -8.29 |
| Sweep best by RMSE | 1286 | 20.61 | 48.60 | -6.52 |

These numbers come from local files under `outputs/eval_runs/` and should be re-run after changing checkpoints, data preprocessing, or post-processing parameters.

## Tests

Run the smoke tests:

```bash
python -m pytest test_dsgeco.py test_postprocess_verification.py
```

The model smoke tests replace heavy DINOv3, query-generator, and post-processing dependencies with fake modules so that shape contracts and forward-path behavior can be checked without loading full pretrained weights.

## GitHub Publishing Notes

Before publishing, add a `.gitignore` that excludes at least:

```gitignore
data/
checkpoints/
logs/
outputs/
__pycache__/
.pytest_cache/
*.pth
*.pt
*.zip
```

Recommended release layout:

- Keep source code, configs, scripts, and tests in Git.
- Put FSC147 data preparation instructions in the README instead of committing the dataset.
- Provide DINOv3, SAM3, and trained DGECO checkpoint links separately.
- Include one or two lightweight example images only if licensing permits.

## Acknowledgements

This project builds on ideas and components from DINOv3, SAM3, Deformable DETR, and FSC147-style few-shot counting research.
