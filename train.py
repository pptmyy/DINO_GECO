import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader

from src.models.DGECO import build_model
from src.models.matcher import build_matcher
from src.datasets.data import FSC147DATASET
from src.datasets.data import pad_collate
from src.utils.losses import SetCriterion
from src.utils.postprocess import filter_detections
from src.utils.verification import verify_detections
import tqdm

from arg_parser import get_argparser

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DATASETS = {'fsc147':FSC147DATASET}


def setup_logging(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_name}_{timestamp}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train")
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

    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    return logger, run_dir


def write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_device(args):

    if torch.cuda.is_available():

        gpu = int(getattr(args,"gpu",0))
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    
    return torch.device("cpu")


def select_candidate_features(outputs_i, indices):
    features = outputs_i.get("candidate_features")
    if features is None:
        return None
    features = features.reshape(-1, features.shape[-1])
    if indices.numel() == 0:
        return features.new_zeros((0, features.shape[-1]))
    indices = indices.to(device=features.device)
    if int(indices.max().item()) >= features.shape[0]:
        return None
    return features[indices]


def count_predictions(outputs_i, args, exemplar_boxes=None):
    boxes, scores, postprocess_stats, keep_indices = filter_detections(
        outputs_i,
        score_threshold=args.score_threshold,
        score_ratio=args.score_ratio,
        threshold_mode=args.threshold_mode,
        score_quantile=args.score_quantile,
        min_score_gap=args.min_score_gap,
        pre_nms_topk=args.pre_nms_topk,
        max_detections=args.max_detections,
        nms_iou=args.nms_iou,
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        adaptive_sparse_score_ratio=getattr(args, "adaptive_sparse_score_ratio", 0.50),
        adaptive_dense_score_ratio=getattr(args, "adaptive_dense_score_ratio", 0.45),
        adaptive_sparse_nms_iou=getattr(args, "adaptive_sparse_nms_iou", 0.25),
        adaptive_dense_nms_iou=getattr(args, "adaptive_dense_nms_iou", 0.25),
        adaptive_dense_candidate_threshold=getattr(
            args, "adaptive_dense_candidate_threshold", 128
        ),
        return_stats=True,
        return_indices=True,
    )
    candidate_features = select_candidate_features(outputs_i, keep_indices)
    boxes, scores, verification_stats = verify_detections(
        boxes,
        scores,
        exemplar_boxes=exemplar_boxes,
        candidate_features=candidate_features,
        exemplar_features=outputs_i.get("exemplar_features"),
        mode=args.verification_mode,
        threshold=args.verification_threshold,
        topk=args.verification_topk,
        min_area_ratio=args.verification_min_area_ratio,
        max_area_ratio=args.verification_max_area_ratio,
        filter_mode=getattr(args, "verification_filter_mode", "hard"),
        score_gamma=getattr(args, "verification_score_gamma", 0.0),
        hard_candidate_limit=getattr(args, "verification_hard_candidate_limit", 0),
        return_stats=True,
    )
    return boxes, postprocess_stats, verification_stats

def make_labels(num_boxes, device):
    """Create class labels for SetCriterion on the same device as the model."""
    return torch.zeros(num_boxes, dtype=torch.long, device=device)



def get_valid_target_boxes(gt_bboxes_i, image_size):
    """Remove padded zero boxes and normalize coordinates to 0-1 range."""
    valid_mask = torch.logical_not((gt_bboxes_i == 0).all(dim=1))
    return gt_bboxes_i[valid_mask] / image_size


def aux_detection_weight(target_bboxes, image_size):
    if target_bboxes.numel() == 0:
        return 0.0
    mean_h = (target_bboxes[:, 3] - target_bboxes[:, 1]).mean() * image_size
    mean_w = (target_bboxes[:, 2] - target_bboxes[:, 0]).mean() * image_size
    return 0.3 if min(mean_h, mean_w) < 25 else 0.0


def build_weight_dict(args):
    return {
        "loss_ce": args.ce_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }


def eval_batch_size(args, split):
    split_value = getattr(args, f"{split}_batch_size")
    return split_value or args.eval_batch_size or args.batch_size


def empty_loss_stats():
    return {
        "loss_total": 0.0,
        "loss_giou": 0.0,
        "loss_bbox": 0.0,
        "loss_ce": 0.0,
        "aux_loss_total": 0.0,
        "aux_loss_giou": 0.0,
        "aux_loss_bbox": 0.0,
        "aux_loss_ce": 0.0,
        "mask_sum": 0.0,
        "aux_mask_sum": 0.0,
        "positive_sum": 0.0,
        "targets": 0,
        "preds_before_nms": 0,
        "preds_after_nms": 0,
        "preds_after_verify": 0,
        "preds_final": 0,
        "effective_threshold_sum": 0.0,
        "effective_threshold_mean": 0.0,
        "verification_filtered": 0,
    }


def add_loss_stats(stats, losses, *, prefix="", scale=1.0):
    for key in ("loss_total", "loss_giou", "loss_bbox", "loss_ce"):
        stats[f"{prefix}{key}"] += (losses[key] * scale).detach().item()


def average_loss_stats(stats, batch_size):
    if batch_size <= 0:
        return stats
    for key in (
        "loss_total",
        "loss_giou",
        "loss_bbox",
        "loss_ce",
        "aux_loss_total",
        "aux_loss_giou",
        "aux_loss_bbox",
        "aux_loss_ce",
    ):
        stats[key] /= batch_size
    stats["effective_threshold_mean"] = stats["effective_threshold_sum"] / batch_size
    return stats


def compute_detection_batch_loss(
    *,
    criterion,
    outputs,
    ref_points,
    centerness,
    aux,
    gt_bboxes,
    exemplar_bboxes=None,
    args,
    device,
):
    outputs_aux, ref_points_aux, centerness_aux, _ = aux
    sample_losses = []
    num_objects_pred = []
    num_objects_gt = []
    stats = empty_loss_stats()

    for idx in range(gt_bboxes.shape[0]):
        target_bboxes = get_valid_target_boxes(gt_bboxes[idx], args.image_size)
        labels = make_labels(target_bboxes.shape[0], device)
        target = [{"boxes": target_bboxes, "labels": labels}]
        num_objects_gt.append(target_bboxes.shape[0])

        main_losses = criterion(outputs[idx], target, centerness[idx], ref_points[idx])
        aux_losses = criterion(outputs_aux[idx], target, centerness_aux[idx], ref_points_aux[idx])
        alpha = args.aux_loss_coef * aux_detection_weight(target_bboxes, args.image_size)
        total = main_losses["loss_total"] + alpha * aux_losses["loss_total"]
        sample_losses.append(total)

        exemplar_boxes = None
        if exemplar_bboxes is not None:
            exemplar_boxes = get_valid_target_boxes(exemplar_bboxes[idx], args.image_size)
        boxes, postprocess_stats, verification_stats = count_predictions(
            outputs[idx],
            args,
            exemplar_boxes=exemplar_boxes,
        )
        num_objects_pred.append(len(boxes))

        add_loss_stats(stats, main_losses)
        add_loss_stats(stats, aux_losses, prefix="aux_", scale=alpha)
        stats["mask_sum"] += float(main_losses["mask_sum"].item())
        stats["aux_mask_sum"] += float(aux_losses["mask_sum"].item())
        stats["positive_sum"] += float(main_losses["positive_sum"].item())
        stats["targets"] += int(target_bboxes.shape[0])
        stats["preds_before_nms"] += postprocess_stats.preds_before_nms
        stats["preds_after_nms"] += postprocess_stats.preds_after_nms
        stats["preds_after_verify"] += verification_stats.kept_count
        stats["preds_final"] += int(len(boxes))
        stats["effective_threshold_sum"] += postprocess_stats.effective_threshold
        stats["verification_filtered"] += verification_stats.filtered_count

    loss = torch.stack(sample_losses).mean()
    num_objects_gt = torch.tensor(num_objects_gt, device=device, dtype=torch.float32)
    num_objects_pred = torch.tensor(num_objects_pred, device=device, dtype=torch.float32)
    return loss, num_objects_gt, num_objects_pred, average_loss_stats(stats, gt_bboxes.shape[0])


def train(args):

    logger, run_dir = setup_logging(args)
    metrics_path = run_dir / "metrics.jsonl"

    device = get_device(args)
    logger.info("Args: %s", vars(args))
    logger.info("Running pure single-GPU training on device: %s", device)
    logger.info("Logs will be saved to: %s", run_dir)

    model = build_model(args).to(device)  

    frozen_dinov3_params = {}
    adapter_params = {}
    non_backbone_params = {}
    for name, param in model.named_parameters():
        if name.startswith("backbone.dinov3."):
            param.requires_grad_(False)
            frozen_dinov3_params[name] = param
        elif name.startswith("backbone."):
            adapter_params[name] = param
        else:
            non_backbone_params[name] = param

    logger.info(
        "Trainable parameter groups: non_backbone=%d, dinov3_adapter=%d, frozen_dinov3=%d",
        sum(p.numel() for p in non_backbone_params.values() if p.requires_grad),
        sum(p.numel() for p in adapter_params.values() if p.requires_grad),
        sum(p.numel() for p in frozen_dinov3_params.values()),
    )

    optimizer = torch.optim.AdamW(
        [
            {'params': [p for p in non_backbone_params.values() if p.requires_grad]},
            {'params': [p for p in adapter_params.values() if p.requires_grad], 'lr': args.backbone_lr},
        ],
        lr = args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    # Keep AMP disabled because the custom MSDeformAttn CUDA op does not support Half.
    amp_enabled = False
    scaler = GradScaler(enabled=amp_enabled)
    logger.info("AMP mixed precision enabled: %s", amp_enabled)

    if args.resume_training:
        checkpoint = torch.load(
            os.path.join(args.model_path, f"{args.model_name_resume_from}.pth"),
            map_location=device,
        )
        model.load_state_dict(checkpoint['model'],strict=False)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_rmse = float(checkpoint.get("best_val_rmse", "inf"))
        best_val_mae = float(checkpoint.get("best_val_mae", "inf"))
    else:
        start_epoch = 0
        best_val_rmse = float("inf")
        best_val_mae = float("inf")
    
    matcher = build_matcher(args)
    
    criterion = SetCriterion(
        0,
        matcher,
        build_weight_dict(args),
        ["bboxes", "ce"],
        focal_alpha=args.focal_alpha,
    )

    criterion.to(device)

    train_dataset = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="train",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
        training=True,
    )
    val_dataset = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="val",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        training=True,
    )
    test_dataset = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="test",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        training=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size(args, "val"),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size(args, "test"),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )
    logger.info(
        "Dataloader batch sizes: train=%d, val=%d, test=%d",
        args.batch_size,
        eval_batch_size(args, "val"),
        eval_batch_size(args, "test"),
    )

    epoch_bar = tqdm.tqdm(
        range(start_epoch+1,args.epochs+1),
        total = args.epochs - start_epoch,
        desc = "Training",
        dynamic_ncols = True
    )

    for epoch in epoch_bar:

        start = perf_counter()

        train_loss = torch.tensor(0.0, device=device)
        train_ae = torch.tensor(0.0, device=device)
        val_loss = torch.tensor(0.0, device=device)
        val_ae = torch.tensor(0.0, device=device)
        val_rmse = torch.tensor(0.0, device=device)
        test_loss = torch.tensor(0.0, device=device)
        test_ae = torch.tensor(0.0, device=device)
        test_rmse = torch.tensor(0.0, device=device)
        train_seen = 0
        val_seen = 0
        test_seen = 0

        model.train()
        criterion.train()

        for batch_idx, (img, bboxes, _density_map, _image_ids, gt_bboxes) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            img = img.to(device)
            bboxes = bboxes.to(device)
            gt_bboxes = gt_bboxes.to(device)

            optimizer.zero_grad()

            with autocast(enabled=amp_enabled):
                outputs, ref_points, centerness, _outputs_coord, aux = model(img, bboxes)

                loss, num_objects_gt, num_objects_pred, log_stats = compute_detection_batch_loss(
                    criterion=criterion,
                    outputs=outputs,
                    ref_points=ref_points,
                    centerness=centerness,
                    aux=aux,
                    gt_bboxes=gt_bboxes,
                    exemplar_bboxes=bboxes,
                    args=args,
                    device=device,
                )

            scaler.scale(loss).backward()

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach() * img.size(0)
            train_ae += torch.abs(num_objects_gt - num_objects_pred).sum()
            train_seen += img.size(0)

            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                logger.info(
                    "Epoch %d | batch %d/%d | loss=%.4f | giou=%.4f | bbox=%.4f | ce=%.4f | "
                    "aux_giou=%.4f | aux_bbox=%.4f | aux_ce=%.4f | "
                    "mask=%.0f | aux_mask=%.0f | pos=%.0f | targets=%d | "
                    "pre_nms=%d | post_nms=%d | post_verify=%d | thr=%.3f | "
                    "centerness=[%.3f, %.3f] | train_seen=%d",
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    loss.item(),
                    log_stats["loss_giou"],
                    log_stats["loss_bbox"],
                    log_stats["loss_ce"],
                    log_stats["aux_loss_giou"],
                    log_stats["aux_loss_bbox"],
                    log_stats["aux_loss_ce"],
                    log_stats["mask_sum"],
                    log_stats["aux_mask_sum"],
                    log_stats["positive_sum"],
                    log_stats["targets"],
                    log_stats["preds_before_nms"],
                    log_stats["preds_after_nms"],
                    log_stats["preds_after_verify"],
                    log_stats["effective_threshold_mean"],
                    centerness.detach().min().item(),
                    centerness.detach().max().item(),
                    train_seen,
                )

        criterion.eval()
        model.eval()

        with torch.no_grad():
            for batch_idx, (img, bboxes, _density_map, _image_ids, gt_bboxes) in enumerate(val_loader):
                if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                    break

                img = img.to(device)
                bboxes = bboxes.to(device)
                gt_bboxes = gt_bboxes.to(device)

                with autocast(enabled=amp_enabled):
                    outputs, ref_points, centerness, _outputs_coord, aux = model(img, bboxes)

                    loss, num_objects_gt, num_objects_pred, _ = compute_detection_batch_loss(
                        criterion=criterion,
                        outputs=outputs,
                        ref_points=ref_points,
                        centerness=centerness,
                        aux=aux,
                        gt_bboxes=gt_bboxes,
                        exemplar_bboxes=bboxes,
                        args=args,
                        device=device,
                    )

                val_loss += loss * img.size(0)
                val_ae += torch.abs(num_objects_gt - num_objects_pred).sum()
                val_rmse += torch.pow(num_objects_gt - num_objects_pred, 2).sum()
                val_seen += img.size(0)

            for batch_idx, (img, bboxes, _density_map, _image_ids, gt_bboxes) in enumerate(test_loader):
                if args.max_test_batches is not None and batch_idx >= args.max_test_batches:
                    break

                img = img.to(device)
                bboxes = bboxes.to(device)
                gt_bboxes = gt_bboxes.to(device)

                with autocast(enabled=amp_enabled):
                    outputs, ref_points, centerness, _outputs_coord, aux = model(img, bboxes)

                    loss, num_objects_gt, num_objects_pred, _ = compute_detection_batch_loss(
                        criterion=criterion,
                        outputs=outputs,
                        ref_points=ref_points,
                        centerness=centerness,
                        aux=aux,
                        gt_bboxes=gt_bboxes,
                        exemplar_bboxes=bboxes,
                        args=args,
                        device=device,
                    )

                test_loss += loss * img.size(0)
                test_ae += torch.abs(num_objects_gt - num_objects_pred).sum()
                test_rmse += torch.pow(num_objects_gt - num_objects_pred, 2).sum()
                test_seen += img.size(0)

        scheduler.step()

        end = perf_counter()
        best_val_rmse_epoch = False
        best_val_mae_epoch = False

        train_denominator = max(train_seen, 1)
        val_denominator = max(val_seen, 1)
        test_denominator = max(test_seen, 1)
        avg_train_loss = train_loss.item() / train_denominator
        avg_val_loss = val_loss.item() / val_denominator
        avg_test_loss = test_loss.item() / test_denominator
        current_val_mae = val_ae.item() / val_denominator
        current_val_rmse = torch.sqrt(val_rmse / val_denominator).item()
        current_test_mae = test_ae.item() / test_denominator
        current_test_rmse = torch.sqrt(test_rmse / test_denominator).item()

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "best_val_rmse": best_val_rmse,
            "best_val_mae": best_val_mae,
            "val_mae": current_val_mae,
            "val_rmse": current_val_rmse,
            "test_mae": current_test_mae,
            "test_rmse": current_test_rmse,
        }
        os.makedirs(args.model_path, exist_ok=True)

        if current_val_rmse < best_val_rmse:
            best_val_rmse = current_val_rmse
            checkpoint["best_val_rmse"] = best_val_rmse
            checkpoint["best_val_mae"] = best_val_mae
            best_val_rmse_epoch = True

        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            checkpoint["best_val_rmse"] = best_val_rmse
            checkpoint["best_val_mae"] = best_val_mae
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_best_val_mae.pth"))
            best_val_mae_epoch = True

        checkpoint["best_val_rmse"] = best_val_rmse
        checkpoint["best_val_mae"] = best_val_mae
        if best_val_rmse_epoch:
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_best_val_rmse.pth"))
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}.pth"))
        torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_last.pth"))

        metrics = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "test_loss": avg_test_loss,
            "train_mae": train_ae.item() / train_denominator,
            "val_mae": current_val_mae,
            "val_rmse": current_val_rmse,
            "test_mae": current_test_mae,
            "test_rmse": current_test_rmse,
            "epoch_time_sec": end - start,
            "best_val_rmse": best_val_rmse,
            "best_val_mae": best_val_mae,
            "best_val_rmse_epoch": best_val_rmse_epoch,
            "best_val_mae_epoch": best_val_mae_epoch,
            "train_seen": train_seen,
            "val_seen": val_seen,
            "test_seen": test_seen,
        }

        logger.info(
            "Epoch %d | train_loss=%.3f | val_loss=%.3f | train_mae=%.3f | "
            "val_mae=%.3f | val_rmse=%.2f | test_mae=%.3f | test_rmse=%.2f | "
            "time=%.3fs%s",
            epoch,
            metrics["train_loss"],
            metrics["val_loss"],
            metrics["train_mae"],
            metrics["val_mae"],
            metrics["val_rmse"],
            metrics["test_mae"],
            metrics["test_rmse"],
            metrics["epoch_time_sec"],
            " | best_rmse" if best_val_rmse_epoch else (" | best_mae" if best_val_mae_epoch else ""),
        )
        write_jsonl(metrics_path, metrics)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    train(args)
