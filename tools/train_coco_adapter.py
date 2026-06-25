import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Mapping

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.coco_class_agnostic import (  # noqa: E402
    CocoClassAgnosticDataset,
    coco_class_agnostic_collate,
)
from src.models.coco_adapter_pretrain import (  # noqa: E402
    CocoAdapterPretrainModel,
    merge_scale_outputs,
    parse_scale_float_map,
    parse_scale_int_map,
    parse_scale_list,
)
from src.models.matcher import PointLossHungarianMatcher  # noqa: E402
from src.utils.adapter_checkpoint import save_dinov3_adapter_checkpoint  # noqa: E402
from src.utils.detection_metrics import (  # noqa: E402
    average_precision_from_records,
    detection_ap_records,
    greedy_detection_counts,
    precision_recall_f1,
)
from src.utils.losses import SetCriterion  # noqa: E402
from src.utils.postprocess import filter_detections  # noqa: E402


def _positive_int(value: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value_int


def _non_negative_int(value: str) -> int:
    value_int = int(value)
    if value_int < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value_int


def _non_negative_float(value: str) -> float:
    value_float = float(value)
    if value_float < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value_float


def _probability(value: str) -> float:
    value_float = float(value)
    if value_float < 0 or value_float > 1:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return value_float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("COCO class-agnostic DINOv3Adapter pretraining")
    parser.add_argument("--coco-path", default=str(PROJECT_ROOT / "data" / "coco2017"))
    parser.add_argument("--train-split", default="train2017")
    parser.add_argument("--val-split", default="val2017")
    parser.add_argument("--train-annotation-file", default=None)
    parser.add_argument("--val-annotation-file", default=None)
    parser.add_argument("--train-image-dir", default=None)
    parser.add_argument("--val-image-dir", default=None)
    parser.add_argument("--image-size", default=1024, type=_positive_int)
    parser.add_argument("--batch-size", default=2, type=_positive_int)
    parser.add_argument("--grad-accum-steps", default=4, type=_positive_int)
    parser.add_argument("--num-workers", default=8, type=_non_negative_int)
    parser.add_argument("--epochs", default=12, type=_positive_int)
    parser.add_argument("--max-train-batches", default=None, type=_positive_int)
    parser.add_argument("--max-val-batches", default=None, type=_positive_int)
    parser.add_argument("--max-train-images", default=None, type=_positive_int)
    parser.add_argument("--max-val-images", default=None, type=_positive_int)
    parser.add_argument("--eval-interval", default=1, type=_non_negative_int)
    parser.add_argument("--log-interval", default=50, type=_non_negative_int)
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs" / "coco_adapter"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoints" / "coco_adapter"))
    parser.add_argument("--model-name", default="dinov3_adapter_coco_class_agnostic")
    parser.add_argument("--gpu", default=0, type=_non_negative_int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dinov3-model-size", default="base", choices=["small", "base", "large"])
    parser.add_argument(
        "--dinov3-pretrained-weights",
        default=str(
            PROJECT_ROOT
            / "src"
            / "models"
            / "backbones"
            / "checkpoint"
            / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ),
    )
    parser.add_argument("--train-scales", default="c1,c2,c3")
    parser.add_argument("--scale-loss-weights", default="")
    parser.add_argument("--max-candidates", default=4096, type=_positive_int)
    parser.add_argument("--max-candidates-per-scale", default="c1:2048,c2:4096,c3:4096")
    parser.add_argument("--min-box-size", default=2.0, type=_non_negative_float)
    parser.add_argument("--max-boxes-per-image", default=0, type=_non_negative_int)
    parser.add_argument("--horizontal-flip-p", default=0.5, type=_probability)
    parser.add_argument("--color-jitter-p", default=0.2, type=_probability)
    parser.add_argument("--lr", default=1e-4, type=_non_negative_float)
    parser.add_argument("--head-lr", default=1e-4, type=_non_negative_float)
    parser.add_argument("--weight-decay", default=1e-4, type=_non_negative_float)
    parser.add_argument("--max-grad-norm", default=0.1, type=_non_negative_float)
    parser.add_argument("--bbox-loss-coef", default=1.0, type=_non_negative_float)
    parser.add_argument("--giou-loss-coef", default=2.0, type=_non_negative_float)
    parser.add_argument("--ce-loss-coef", default=2.0, type=_non_negative_float)
    parser.add_argument("--cost-class", default=2.0, type=_non_negative_float)
    parser.add_argument("--cost-bbox", default=1.0, type=_non_negative_float)
    parser.add_argument("--cost-giou", default=2.0, type=_non_negative_float)
    parser.add_argument("--focal-alpha", default=0.5, type=_probability)
    parser.add_argument("--score-threshold", default=0.20, type=_probability)
    parser.add_argument("--score-ratio", default=0.50, type=_non_negative_float)
    parser.add_argument(
        "--threshold-mode",
        default="static_ratio",
        choices=["static_ratio", "quantile", "regime_adaptive"],
    )
    parser.add_argument("--pre-nms-topk", default=16000, type=_positive_int)
    parser.add_argument("--max-detections", default=300, type=_positive_int)
    parser.add_argument("--nms-iou", default=0.5, type=_probability)
    return parser


def setup_logging(args: argparse.Namespace) -> tuple[logging.Logger, Path]:
    run_name = args.run_name or f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_coco_adapter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    with (run_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)
    return logger, run_dir


def get_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        return torch.device(f"cuda:{int(args.gpu)}")
    return torch.device("cpu")


def write_jsonl(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def valid_target_boxes(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    valid = torch.logical_not((boxes == 0).all(dim=1))
    boxes = boxes[valid]
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    wh = boxes[:, 2:] - boxes[:, :2]
    boxes = boxes[(wh[:, 0] > 0) & (wh[:, 1] > 0)]
    return boxes / float(image_size)


def make_criterion(args: argparse.Namespace) -> SetCriterion:
    matcher = PointLossHungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
    )
    return SetCriterion(
        1,
        matcher,
        {
            "loss_ce": args.ce_loss_coef,
            "loss_bbox": args.bbox_loss_coef,
            "loss_giou": args.giou_loss_coef,
        },
        ["ce", "bboxes"],
        focal_alpha=args.focal_alpha,
    )


def default_scale_weights(scales: tuple[str, ...]) -> Dict[str, float]:
    defaults = {"c1": 0.25, "c2": 0.5, "c3": 1.0}
    return {scale: defaults.get(scale, 1.0) for scale in scales}


def empty_stats(scales: tuple[str, ...]) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "loss_total": 0.0,
        "targets": 0,
        "preds": 0,
        "det_tp": 0,
        "det_fp": 0,
        "det_fn": 0,
        "det_gt": 0,
        "ap_records": [],
    }
    for scale in scales:
        stats[f"{scale}_loss_total"] = 0.0
        stats[f"{scale}_loss_ce"] = 0.0
        stats[f"{scale}_loss_bbox"] = 0.0
        stats[f"{scale}_loss_giou"] = 0.0
        stats[f"{scale}_mask_sum"] = 0.0
    return stats


def add_loss_stats(
    stats: Dict[str, object],
    scale: str,
    losses: Mapping[str, torch.Tensor],
    *,
    weight: float,
) -> None:
    stats[f"{scale}_loss_total"] += float((losses["loss_total"] * weight).detach().item())
    stats[f"{scale}_loss_ce"] += float((losses["loss_ce"] * weight).detach().item())
    stats[f"{scale}_loss_bbox"] += float((losses["loss_bbox"] * weight).detach().item())
    stats[f"{scale}_loss_giou"] += float((losses["loss_giou"] * weight).detach().item())
    stats[f"{scale}_mask_sum"] += float(losses["mask_sum"].detach().item())


def postprocess_for_metrics(output_i: Mapping[str, torch.Tensor], args: argparse.Namespace):
    return filter_detections(
        output_i,
        score_threshold=args.score_threshold,
        score_ratio=args.score_ratio,
        threshold_mode=args.threshold_mode,
        pre_nms_topk=args.pre_nms_topk,
        max_detections=args.max_detections,
        nms_iou=args.nms_iou,
        nms_method="hard",
        return_stats=False,
    )


def compute_batch_loss(
    *,
    predictions: Mapping[str, Mapping[str, object]],
    gt_boxes: torch.Tensor,
    criterion: SetCriterion,
    scale_weights: Mapping[str, float],
    args: argparse.Namespace,
    device: torch.device,
    collect_metrics: bool,
) -> tuple[torch.Tensor, Dict[str, object]]:
    scales = tuple(predictions.keys())
    stats = empty_stats(scales)
    sample_losses = []
    for sample_idx in range(gt_boxes.shape[0]):
        target_boxes = valid_target_boxes(gt_boxes[sample_idx], args.image_size).to(device)
        target = [
            {
                "boxes": target_boxes,
                "labels": torch.zeros(target_boxes.shape[0], dtype=torch.long, device=device),
            }
        ]
        stats["targets"] += int(target_boxes.shape[0])
        sample_total = None
        for scale, scale_prediction in predictions.items():
            losses = criterion(
                scale_prediction["outputs"][sample_idx],
                target,
                scale_prediction["centerness"][sample_idx],
                scale_prediction["ref_points"][sample_idx],
            )
            weight = float(scale_weights[scale])
            weighted = losses["loss_total"] * weight
            sample_total = weighted if sample_total is None else sample_total + weighted
            add_loss_stats(stats, scale, losses, weight=weight)
        if sample_total is None:
            raise RuntimeError("No scale losses were produced")
        sample_losses.append(sample_total)
        stats["loss_total"] += float(sample_total.detach().item())

        if collect_metrics:
            merged = merge_scale_outputs(predictions, sample_idx)
            boxes, scores = postprocess_for_metrics(merged, args)
            stats["preds"] += int(boxes.shape[0])
            tp, fp, fn = greedy_detection_counts(boxes, target_boxes, iou_threshold=0.5)
            stats["det_tp"] += tp
            stats["det_fp"] += fp
            stats["det_fn"] += fn
            stats["det_gt"] += int(target_boxes.shape[0])
            stats["ap_records"].extend(
                detection_ap_records(boxes, scores, target_boxes, iou_threshold=0.5)
            )

    loss = torch.stack(sample_losses).mean()
    batch_size = max(int(gt_boxes.shape[0]), 1)
    for key, value in list(stats.items()):
        if key == "ap_records":
            continue
        if key.startswith("det_") or key in {"targets", "preds"}:
            continue
        stats[key] = float(value) / batch_size
    return loss, stats


def merge_epoch_stats(total: Dict[str, object], update: Mapping[str, object]) -> Dict[str, object]:
    for key, value in update.items():
        if key == "ap_records":
            total.setdefault(key, [])
            total[key].extend(value)
        else:
            total[key] = total.get(key, 0) + value
    return total


def finalize_epoch_stats(stats: Dict[str, object], batches: int) -> Dict[str, object]:
    batches = max(int(batches), 1)
    finalized = dict(stats)
    for key, value in list(finalized.items()):
        if key == "ap_records":
            continue
        if key.startswith("det_") or key in {"targets", "preds"}:
            finalized[key] = int(value)
        else:
            finalized[key] = float(value) / batches
    precision, recall, f1 = precision_recall_f1(
        int(finalized.get("det_tp", 0)),
        int(finalized.get("det_fp", 0)),
        int(finalized.get("det_fn", 0)),
    )
    finalized["precision_iou50"] = precision
    finalized["recall_iou50"] = recall
    finalized["f1_iou50"] = f1
    finalized["ap50"] = average_precision_from_records(
        finalized.get("ap_records", []),
        int(finalized.get("det_gt", 0)),
    )
    finalized.pop("ap_records", None)
    return finalized


def make_dataloader(dataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=coco_class_agnostic_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def build_model(args: argparse.Namespace, train_scales, max_candidates_per_scale):
    return CocoAdapterPretrainModel(
        image_size=args.image_size,
        dinov3_model_size=args.dinov3_model_size,
        dinov3_pretrained_weights=args.dinov3_pretrained_weights,
        train_scales=train_scales,
        max_candidates_per_scale=max_candidates_per_scale,
    )


def save_checkpoint(
    *,
    model: CocoAdapterPretrainModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    model_dir: Path,
    suffix: str,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    full_path = model_dir / f"{args.model_name}_{suffix}_full.pth"
    adapter_path = model_dir / f"{args.model_name}_{suffix}_adapter.pth"
    torch.save(checkpoint, full_path)
    save_dinov3_adapter_checkpoint(
        adapter_path,
        model,
        metadata={
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "source": "coco_class_agnostic_adapter_pretrain",
            "args": vars(args),
        },
    )


def train(args: argparse.Namespace) -> None:
    train_scales = parse_scale_list(args.train_scales)
    scale_weights = default_scale_weights(train_scales)
    if args.scale_loss_weights:
        scale_weights.update(parse_scale_float_map(args.scale_loss_weights, train_scales, 1.0))
    max_candidates_per_scale = parse_scale_int_map(
        args.max_candidates_per_scale,
        train_scales,
        args.max_candidates,
    )

    logger, run_dir = setup_logging(args)
    metrics_path = run_dir / "metrics.jsonl"
    model_dir = Path(args.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args)
    logger.info("Using device: %s", device)
    logger.info("Train scales: %s", ",".join(train_scales))
    logger.info("Scale weights: %s", scale_weights)
    logger.info("Max candidates per scale: %s", max_candidates_per_scale)

    train_dataset = CocoClassAgnosticDataset(
        args.coco_path,
        split=args.train_split,
        image_size=args.image_size,
        annotation_file=args.train_annotation_file,
        image_dir=args.train_image_dir,
        training=True,
        horizontal_flip_p=args.horizontal_flip_p,
        color_jitter_p=args.color_jitter_p,
        min_box_size=args.min_box_size,
        max_boxes_per_image=args.max_boxes_per_image,
        max_images=args.max_train_images,
    )
    val_dataset = CocoClassAgnosticDataset(
        args.coco_path,
        split=args.val_split,
        image_size=args.image_size,
        annotation_file=args.val_annotation_file,
        image_dir=args.val_image_dir,
        training=False,
        horizontal_flip_p=0.0,
        color_jitter_p=0.0,
        min_box_size=args.min_box_size,
        max_boxes_per_image=args.max_boxes_per_image,
        max_images=args.max_val_images,
    )
    train_loader = make_dataloader(train_dataset, args, shuffle=True)
    val_loader = make_dataloader(val_dataset, args, shuffle=False)
    logger.info("Dataset sizes: train=%d val=%d", len(train_dataset), len(val_dataset))

    model = build_model(args, train_scales, max_candidates_per_scale).to(device)
    criterion = make_criterion(args)
    adapter_params = []
    head_params = []
    frozen_params = 0
    for name, parameter in model.named_parameters():
        if name.startswith("backbone.dinov3."):
            parameter.requires_grad_(False)
            frozen_params += parameter.numel()
        elif name.startswith("heads."):
            head_params.append(parameter)
        else:
            adapter_params.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for param in adapter_params if param.requires_grad], "lr": args.lr},
            {"params": [param for param in head_params if param.requires_grad], "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    logger.info(
        "Parameters: adapter=%d head=%d frozen_dinov3=%d",
        sum(param.numel() for param in adapter_params if param.requires_grad),
        sum(param.numel() for param in head_params if param.requires_grad),
        frozen_params,
    )

    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        train_total: Dict[str, object] = {"ap_records": []}
        train_batches = 0
        for batch_idx, (images, gt_boxes, _image_ids) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            gt_boxes = gt_boxes.to(device, non_blocking=True)
            with autocast(enabled=args.amp and device.type == "cuda"):
                predictions = model(images)
                loss, batch_stats = compute_batch_loss(
                    predictions=predictions,
                    gt_boxes=gt_boxes,
                    criterion=criterion,
                    scale_weights=scale_weights,
                    args=args,
                    device=device,
                    collect_metrics=False,
                )
            scaler.scale(loss / args.grad_accum_steps).backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            merge_epoch_stats(train_total, batch_stats)
            train_batches += 1
            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                logger.info(
                    "Epoch %d | batch %d/%d | loss=%.4f | targets=%d",
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    float(loss.detach().item()),
                    int(batch_stats["targets"]),
                )

        if train_batches > 0 and train_batches % args.grad_accum_steps != 0:
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_stats = finalize_epoch_stats(train_total, train_batches)

        do_eval = args.eval_interval > 0 and (
            epoch % args.eval_interval == 0 or epoch == args.epochs
        )
        val_stats = {}
        val_batches = 0
        if do_eval:
            model.eval()
            val_total: Dict[str, object] = {"ap_records": []}
            with torch.no_grad():
                for batch_idx, (images, gt_boxes, _image_ids) in enumerate(val_loader):
                    if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                        break
                    images = images.to(device, non_blocking=True)
                    gt_boxes = gt_boxes.to(device, non_blocking=True)
                    predictions = model(images)
                    _loss, batch_stats = compute_batch_loss(
                        predictions=predictions,
                        gt_boxes=gt_boxes,
                        criterion=criterion,
                        scale_weights=scale_weights,
                        args=args,
                        device=device,
                        collect_metrics=True,
                    )
                    merge_epoch_stats(val_total, batch_stats)
                    val_batches += 1
            val_stats = finalize_epoch_stats(val_total, val_batches)

        current_val_loss = float(val_stats.get("loss_total", train_stats["loss_total"]))
        is_best = bool(do_eval and current_val_loss < best_val_loss)
        if is_best:
            best_val_loss = current_val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
                model_dir=model_dir,
                suffix="best_val_loss",
            )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            args=args,
            model_dir=model_dir,
            suffix="last",
        )

        metrics = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
            "best_val_loss_epoch": is_best,
            "epoch_time_sec": perf_counter() - start,
        }
        write_jsonl(metrics_path, metrics)
        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%s | val_ap50=%s | val_f1=%s | time=%.1fs%s",
            epoch,
            float(train_stats["loss_total"]),
            f"{float(val_stats['loss_total']):.4f}" if val_stats else "n/a",
            f"{float(val_stats['ap50']):.4f}" if val_stats else "n/a",
            f"{float(val_stats['f1_iou50']):.4f}" if val_stats else "n/a",
            metrics["epoch_time_sec"],
            " | best_val_loss" if is_best else "",
        )


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    os.makedirs(parsed_args.model_path, exist_ok=True)
    train(parsed_args)
