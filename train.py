import argparse
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
from src.utils.checkpoint import load_model_state_dict
from src.utils.detection_metrics import (
    average_precision_from_records,
    detection_ap_records,
    greedy_detection_counts,
    precision_recall_f1,
)
from src.utils.postprocess import filter_detections
from src.utils.training_diagnostics import (
    add_loss_stats,
    add_postprocess_diagnostics,
    average_loss_stats,
    diagnostic_metrics,
    empty_loss_stats,
    format_optional_metric,
    merge_diagnostic_stats,
)
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
        nms_method=getattr(args, "nms_method", "dense_soft"),
        soft_nms_sigma=getattr(args, "soft_nms_sigma", 0.5),
        soft_nms_score_threshold=getattr(args, "soft_nms_score_threshold", 0.001),
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        adaptive_sparse_score_ratio=getattr(args, "adaptive_sparse_score_ratio", 0.50),
        adaptive_dense_score_ratio=getattr(args, "adaptive_dense_score_ratio", 0.35),
        adaptive_sparse_nms_iou=getattr(args, "adaptive_sparse_nms_iou", 0.25),
        adaptive_dense_nms_iou=getattr(args, "adaptive_dense_nms_iou", 0.45),
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
    return boxes, scores, postprocess_stats, verification_stats

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
        "loss_center_gaussian": args.center_gaussian_loss_coef,
    }


def eval_batch_size(args, split):
    split_value = getattr(args, f"{split}_batch_size")
    return split_value or args.eval_batch_size or args.batch_size


def train_batches_per_epoch(args, train_loader):
    if args.max_train_batches is None:
        return len(train_loader)
    return min(len(train_loader), args.max_train_batches)


def accumulation_window_size(batch_idx, total_batches, grad_accum_steps):
    window_start = batch_idx - (batch_idx % grad_accum_steps)
    return min(grad_accum_steps, total_batches - window_start)


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
    collect_detection_metrics=False,
):
    refine_outputs = aux["refine_outputs"]
    refine_ref_points = aux["refine_ref_points"]
    refine_centerness = aux["refine_centerness"]
    center_heatmap = aux.get("center_heatmap")
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
        aux_losses = criterion(
            refine_outputs[idx],
            target,
            refine_centerness[idx],
            refine_ref_points[idx],
        )
        alpha = args.aux_loss_coef
        total = main_losses["loss_total"] + alpha * aux_losses["loss_total"]
        center_losses = None
        center_weight = float(getattr(args, "center_gaussian_loss_coef", 0.0))
        if center_heatmap is not None and center_weight > 0:
            center_losses = criterion.center_gaussian_loss(target, center_heatmap[idx])
            total = total + center_weight * center_losses["loss_center_gaussian"]
        sample_losses.append(total)

        exemplar_boxes = None
        if exemplar_bboxes is not None:
            exemplar_boxes = get_valid_target_boxes(exemplar_bboxes[idx], args.image_size)
        with torch.no_grad():
            boxes, scores, postprocess_stats, verification_stats = count_predictions(
                outputs[idx],
                args,
                exemplar_boxes=exemplar_boxes,
            )
            num_objects_pred.append(len(boxes))
            if collect_detection_metrics:
                tp, fp, fn = greedy_detection_counts(
                    boxes,
                    target_bboxes,
                    iou_threshold=getattr(args, "val_iou_threshold", 0.5),
                )
                stats["det_tp_iou50"] += tp
                stats["det_fp_iou50"] += fp
                stats["det_fn_iou50"] += fn
                stats["det_gt_iou50"] += int(target_bboxes.shape[0])
                stats["det_ap_records_iou50"].extend(
                    detection_ap_records(
                        boxes,
                        scores,
                        target_bboxes,
                        iou_threshold=getattr(args, "val_iou_threshold", 0.5),
                    )
                )

        add_loss_stats(stats, main_losses)
        add_loss_stats(stats, aux_losses, prefix="aux_", scale=alpha)
        if center_losses is not None:
            add_loss_stats(stats, center_losses, scale=center_weight)
            stats["center_gaussian_positive_sum"] += float(
                center_losses["center_gaussian_positive_sum"].item()
            )
            stats["center_gaussian_weight_sum"] += center_weight
        stats["mask_sum"] += float(main_losses["mask_sum"].item())
        stats["aux_mask_sum"] += float(aux_losses["mask_sum"].item())
        aux_positive = float(aux_losses["positive_sum"].item())
        stats["aux_positive_sum"] += aux_positive
        stats["aux_active_samples"] += int(aux_positive > 0)
        stats["aux_weight_sum"] += float(alpha)
        stats["positive_sum"] += float(main_losses["positive_sum"].item())
        stats["targets"] += int(target_bboxes.shape[0])
        add_postprocess_diagnostics(
            stats,
            postprocess_stats,
            verification_stats,
            final_count=len(boxes),
        )

    loss = torch.stack(sample_losses).mean()
    num_objects_gt = torch.tensor(num_objects_gt, device=device, dtype=torch.float32)
    num_objects_pred = torch.tensor(num_objects_pred, device=device, dtype=torch.float32)
    return loss, num_objects_gt, num_objects_pred, average_loss_stats(stats, gt_bboxes.shape[0])


def train(args):

    logger, run_dir = setup_logging(args)
    metrics_path = run_dir / "metrics.jsonl"
    query_output_stride = int(getattr(args, "query_output_stride", 4))
    use_semantic_anchor = bool(getattr(args, "use_semantic_anchor", False))

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
    logger.info(
        "Detection config: query_stride=%s pre_nms_topk=%d max_detections=%d "
        "threshold=%s score_ratio=%.3f min_score=%.3f nms=%s nms_iou=%.3f "
        "dense_ratio=%.3f dense_nms_iou=%.3f dense_candidate_threshold=%d "
        "semantic_anchor=%s refinement=stride2:%s center_loss=%.3f "
        "num_prototypes=%d mutual_layers=%d decoupled_heads=%s "
        "verification=%s/%s detection_gate_ratio=%.3f",
        query_output_stride,
        args.pre_nms_topk,
        args.max_detections,
        args.threshold_mode,
        args.score_ratio,
        args.score_threshold,
        getattr(args, "nms_method", "dense_soft"),
        args.nms_iou,
        getattr(args, "adaptive_dense_score_ratio", 0.35),
        getattr(args, "adaptive_dense_nms_iou", 0.45),
        getattr(args, "adaptive_dense_candidate_threshold", 128),
        use_semantic_anchor,
        getattr(args, "stride2_refinement", True),
        getattr(args, "center_gaussian_loss_coef", 0.0),
        getattr(args, "num_prototypes", 4),
        getattr(args, "mutual_adapter_layers", 1),
        getattr(args, "decoupled_heads", True),
        args.verification_mode,
        getattr(args, "verification_filter_mode", "hard"),
        getattr(args, "detection_gate_ratio", 0.98),
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
        load_model_state_dict(
            model,
            checkpoint["model"],
            allow_partial_load=getattr(args, "allow_partial_load", False),
            logger=logger.warning,
            context=os.path.join(args.model_path, f"{args.model_name_resume_from}.pth"),
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_rmse = float(checkpoint.get("best_val_rmse", "inf"))
        best_val_mae = float(checkpoint.get("best_val_mae", "inf"))
        best_val_loss = float(checkpoint.get("best_val_loss", "inf"))
        best_val_ap50_iou50 = float(checkpoint.get("best_val_ap50_iou50", "-inf"))
        best_val_f1_iou50 = float(checkpoint.get("best_val_f1_iou50", "-inf"))
        best_val_gated_rmse = float(checkpoint.get("best_val_gated_rmse", "inf"))
        best_val_gated_mae = float(checkpoint.get("best_val_gated_mae", "inf"))
    else:
        start_epoch = 0
        best_val_rmse = float("inf")
        best_val_mae = float("inf")
        best_val_loss = float("inf")
        best_val_ap50_iou50 = float("-inf")
        best_val_f1_iou50 = float("-inf")
        best_val_gated_rmse = float("inf")
        best_val_gated_mae = float("inf")
    
    matcher = build_matcher(args)
    
    criterion = SetCriterion(
        0,
        matcher,
        build_weight_dict(args),
        ["bboxes", "ce"],
        focal_alpha=args.focal_alpha,
        center_gaussian_sigma=args.center_gaussian_sigma,
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
        allow_missing_coco=args.allow_missing_coco,
        exemplar_scale_mode=args.exemplar_scale_mode,
    )
    val_dataset = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="val",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
        training=True,
        allow_missing_coco=args.allow_missing_coco,
        exemplar_scale_mode=args.exemplar_scale_mode,
    )
    test_dataset = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="test",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
        training=True,
        allow_missing_coco=args.allow_missing_coco,
        exemplar_scale_mode=args.exemplar_scale_mode,
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
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    logger.info(
        "Gradient accumulation: steps=%d, effective_train_batch_size=%d",
        grad_accum_steps,
        args.batch_size * grad_accum_steps,
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
        train_diag_stats = empty_loss_stats()
        val_diag_stats = empty_loss_stats()
        test_diag_stats = empty_loss_stats()
        val_det_tp = 0
        val_det_fp = 0
        val_det_fn = 0
        val_det_gt = 0
        val_det_ap_records = []
        test_det_tp = 0
        test_det_fp = 0
        test_det_fn = 0
        test_det_gt = 0
        test_det_ap_records = []
        collect_eval_detection_metrics = (
            args.detection_metric_interval > 0
            and (
                epoch == start_epoch + 1
                or epoch % args.detection_metric_interval == 0
                or epoch == args.epochs
            )
        )

        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        total_train_batches = train_batches_per_epoch(args, train_loader)

        for batch_idx, (img, bboxes, _density_map, _image_ids, gt_bboxes) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            img = img.to(device)
            bboxes = bboxes.to(device)
            gt_bboxes = gt_bboxes.to(device)
            accum_window = accumulation_window_size(
                batch_idx,
                total_train_batches,
                grad_accum_steps,
            )

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

            scaler.scale(loss / accum_window).backward()

            should_step = (
                (batch_idx + 1) % grad_accum_steps == 0
                or (batch_idx + 1) >= total_train_batches
            )
            if should_step:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            train_loss += loss.detach() * img.size(0)
            train_ae += torch.abs(num_objects_gt - num_objects_pred).sum()
            train_seen += img.size(0)
            merge_diagnostic_stats(train_diag_stats, log_stats)

            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                logger.info(
                    "Epoch %d | batch %d/%d | loss=%.4f | giou=%.4f | bbox=%.4f | ce=%.4f | "
                    "aux_giou=%.4f | aux_bbox=%.4f | aux_ce=%.4f | center=%.4f | "
                    "mask=%.0f | aux_mask=%.0f | pos=%.0f | aux_pos=%.0f | "
                    "center_pos=%.0f | aux_active=%d/%d | aux_w=%.3f | targets=%d | "
                    "pre_nms=%d | post_nms=%d | post_verify=%d | thr=%.3f | "
                    "dense=%.2f | soft_nms=%.2f | keep=%.3f | score=%.3f | "
                    "score_ratio=%.3f | nms_iou=%.3f | "
                    "centerness=[%.3f, %.3f] | accum=%d/%d | opt_steps=%d | train_seen=%d",
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
                    log_stats["loss_center_gaussian"],
                    log_stats["mask_sum"],
                    log_stats["aux_mask_sum"],
                    log_stats["positive_sum"],
                    log_stats["aux_positive_sum"],
                    log_stats["center_gaussian_positive_sum"],
                    log_stats["aux_active_samples"],
                    img.size(0),
                    log_stats["aux_weight_sum"],
                    log_stats["targets"],
                    log_stats["preds_before_nms"],
                    log_stats["preds_after_nms"],
                    log_stats["preds_after_verify"],
                    log_stats["effective_threshold_mean"],
                    log_stats["adaptive_dense_samples"]
                    / max(log_stats["postprocess_samples"], 1),
                    log_stats["nms_soft_samples"]
                    / max(log_stats["postprocess_samples"], 1),
                    log_stats["preds_after_nms"] / max(log_stats["preds_before_nms"], 1),
                    log_stats["score_mean"],
                    log_stats["effective_score_ratio_mean"],
                    log_stats["effective_nms_iou_mean"],
                    centerness.detach().min().item(),
                    centerness.detach().max().item(),
                    (batch_idx % grad_accum_steps) + 1,
                    accum_window,
                    optimizer_steps,
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

                    loss, num_objects_gt, num_objects_pred, val_stats = compute_detection_batch_loss(
                        criterion=criterion,
                        outputs=outputs,
                        ref_points=ref_points,
                        centerness=centerness,
                        aux=aux,
                        gt_bboxes=gt_bboxes,
                        exemplar_bboxes=bboxes,
                        args=args,
                        device=device,
                        collect_detection_metrics=collect_eval_detection_metrics,
                    )

                val_loss += loss * img.size(0)
                val_ae += torch.abs(num_objects_gt - num_objects_pred).sum()
                val_rmse += torch.pow(num_objects_gt - num_objects_pred, 2).sum()
                val_seen += img.size(0)
                val_det_tp += int(val_stats["det_tp_iou50"])
                val_det_fp += int(val_stats["det_fp_iou50"])
                val_det_fn += int(val_stats["det_fn_iou50"])
                val_det_gt += int(val_stats["det_gt_iou50"])
                val_det_ap_records.extend(val_stats["det_ap_records_iou50"])
                merge_diagnostic_stats(val_diag_stats, val_stats)

            for batch_idx, (img, bboxes, _density_map, _image_ids, gt_bboxes) in enumerate(test_loader):
                if args.max_test_batches is not None and batch_idx >= args.max_test_batches:
                    break

                img = img.to(device)
                bboxes = bboxes.to(device)
                gt_bboxes = gt_bboxes.to(device)

                with autocast(enabled=amp_enabled):
                    outputs, ref_points, centerness, _outputs_coord, aux = model(img, bboxes)

                    loss, num_objects_gt, num_objects_pred, test_stats = compute_detection_batch_loss(
                        criterion=criterion,
                        outputs=outputs,
                        ref_points=ref_points,
                        centerness=centerness,
                        aux=aux,
                        gt_bboxes=gt_bboxes,
                        exemplar_bboxes=bboxes,
                        args=args,
                        device=device,
                        collect_detection_metrics=collect_eval_detection_metrics,
                    )

                test_loss += loss * img.size(0)
                test_ae += torch.abs(num_objects_gt - num_objects_pred).sum()
                test_rmse += torch.pow(num_objects_gt - num_objects_pred, 2).sum()
                test_seen += img.size(0)
                test_det_tp += int(test_stats["det_tp_iou50"])
                test_det_fp += int(test_stats["det_fp_iou50"])
                test_det_fn += int(test_stats["det_fn_iou50"])
                test_det_gt += int(test_stats["det_gt_iou50"])
                test_det_ap_records.extend(test_stats["det_ap_records_iou50"])
                merge_diagnostic_stats(test_diag_stats, test_stats)

        scheduler.step()

        end = perf_counter()
        best_val_rmse_epoch = False
        best_val_mae_epoch = False
        best_val_loss_epoch = False
        best_val_ap50_iou50_epoch = False
        best_val_f1_iou50_epoch = False
        best_val_gated_rmse_epoch = False
        best_val_gated_mae_epoch = False

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
        if collect_eval_detection_metrics:
            val_precision_iou50, val_recall_iou50, current_val_f1_iou50 = precision_recall_f1(
                val_det_tp,
                val_det_fp,
                val_det_fn,
            )
            current_val_ap50_iou50 = average_precision_from_records(
                val_det_ap_records,
                val_det_gt,
            )
            test_precision_iou50, test_recall_iou50, current_test_f1_iou50 = precision_recall_f1(
                test_det_tp,
                test_det_fp,
                test_det_fn,
            )
            current_test_ap50_iou50 = average_precision_from_records(
                test_det_ap_records,
                test_det_gt,
            )
        else:
            val_precision_iou50 = None
            val_recall_iou50 = None
            current_val_ap50_iou50 = None
            current_val_f1_iou50 = None
            test_precision_iou50 = None
            test_recall_iou50 = None
            current_test_ap50_iou50 = None
            current_test_f1_iou50 = None

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "best_val_rmse": best_val_rmse,
            "best_val_mae": best_val_mae,
            "best_val_loss": best_val_loss,
            "best_val_ap50_iou50": best_val_ap50_iou50,
            "best_val_f1_iou50": best_val_f1_iou50,
            "best_val_gated_rmse": best_val_gated_rmse,
            "best_val_gated_mae": best_val_gated_mae,
            "val_mae": current_val_mae,
            "val_rmse": current_val_rmse,
            "val_loss": avg_val_loss,
            "val_ap50_iou50": current_val_ap50_iou50,
            "val_f1_iou50": current_val_f1_iou50,
            "test_mae": current_test_mae,
            "test_rmse": current_test_rmse,
            "test_ap50_iou50": current_test_ap50_iou50,
            "test_f1_iou50": current_test_f1_iou50,
            "query_output_stride": query_output_stride,
            "use_semantic_anchor": use_semantic_anchor,
            "stride2_refinement": bool(getattr(args, "stride2_refinement", True)),
            "center_gaussian_loss_coef": float(
                getattr(args, "center_gaussian_loss_coef", 0.0)
            ),
            "num_prototypes": int(getattr(args, "num_prototypes", 4)),
            "mutual_adapter_layers": int(getattr(args, "mutual_adapter_layers", 1)),
            "decoupled_heads": bool(getattr(args, "decoupled_heads", True)),
            "config": dict(vars(args)),
        }
        os.makedirs(args.model_path, exist_ok=True)

        if (
            collect_eval_detection_metrics
            and current_val_ap50_iou50 is not None
            and current_val_ap50_iou50 > best_val_ap50_iou50
        ):
            best_val_ap50_iou50 = current_val_ap50_iou50
            best_val_ap50_iou50_epoch = True

        if (
            collect_eval_detection_metrics
            and current_val_f1_iou50 is not None
            and current_val_f1_iou50 > best_val_f1_iou50
        ):
            best_val_f1_iou50 = current_val_f1_iou50
            best_val_f1_iou50_epoch = True

        if current_val_rmse < best_val_rmse:
            best_val_rmse = current_val_rmse
            best_val_rmse_epoch = True

        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            best_val_mae_epoch = True

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_epoch = True

        detection_f1_gate_threshold = (
            0.0
            if best_val_f1_iou50 == float("-inf")
            else best_val_f1_iou50 * float(getattr(args, "detection_gate_ratio", 0.98))
        )
        detection_ap50_gate_threshold = (
            0.0
            if best_val_ap50_iou50 == float("-inf")
            else best_val_ap50_iou50 * float(getattr(args, "detection_gate_ratio", 0.98))
        )
        detection_gate_passed = (
            collect_eval_detection_metrics
            and current_val_f1_iou50 is not None
            and current_val_ap50_iou50 is not None
            and current_val_f1_iou50 >= detection_f1_gate_threshold
            and current_val_ap50_iou50 >= detection_ap50_gate_threshold
        )

        if detection_gate_passed and current_val_rmse < best_val_gated_rmse:
            best_val_gated_rmse = current_val_rmse
            best_val_gated_rmse_epoch = True

        if detection_gate_passed and current_val_mae < best_val_gated_mae:
            best_val_gated_mae = current_val_mae
            best_val_gated_mae_epoch = True

        checkpoint["best_val_rmse"] = best_val_rmse
        checkpoint["best_val_mae"] = best_val_mae
        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["best_val_ap50_iou50"] = best_val_ap50_iou50
        checkpoint["best_val_f1_iou50"] = best_val_f1_iou50
        checkpoint["best_val_gated_rmse"] = best_val_gated_rmse
        checkpoint["best_val_gated_mae"] = best_val_gated_mae
        if best_val_ap50_iou50_epoch:
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_best_val_ap50_iou50.pth"))
        if best_val_f1_iou50_epoch:
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_best_val_f1_iou50.pth"))
        if best_val_loss_epoch:
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_best_val_loss.pth"))
        if best_val_mae_epoch:
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_best_val_mae.pth"))
        if best_val_rmse_epoch:
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}_best_val_rmse.pth"))
            if not collect_eval_detection_metrics:
                torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}.pth"))
        if best_val_gated_rmse_epoch:
            torch.save(
                checkpoint,
                os.path.join(args.model_path, f"{args.model_name}_best_val_gated_rmse.pth"),
            )
            torch.save(checkpoint, os.path.join(args.model_path, f"{args.model_name}.pth"))
        if best_val_gated_mae_epoch:
            torch.save(
                checkpoint,
                os.path.join(args.model_path, f"{args.model_name}_best_val_gated_mae.pth"),
            )
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
            "val_precision_iou50": val_precision_iou50,
            "val_recall_iou50": val_recall_iou50,
            "val_ap50_iou50": current_val_ap50_iou50,
            "val_f1_iou50": current_val_f1_iou50,
            "test_mae": current_test_mae,
            "test_rmse": current_test_rmse,
            "test_precision_iou50": test_precision_iou50,
            "test_recall_iou50": test_recall_iou50,
            "test_ap50_iou50": current_test_ap50_iou50,
            "test_f1_iou50": current_test_f1_iou50,
            "detection_metrics_collected": collect_eval_detection_metrics,
            "query_output_stride": query_output_stride,
            "use_semantic_anchor": use_semantic_anchor,
            "stride2_refinement": bool(getattr(args, "stride2_refinement", True)),
            "center_gaussian_loss_coef": float(
                getattr(args, "center_gaussian_loss_coef", 0.0)
            ),
            "num_prototypes": int(getattr(args, "num_prototypes", 4)),
            "mutual_adapter_layers": int(getattr(args, "mutual_adapter_layers", 1)),
            "decoupled_heads": bool(getattr(args, "decoupled_heads", True)),
            "epoch_time_sec": end - start,
            "grad_accum_steps": grad_accum_steps,
            "effective_train_batch_size": args.batch_size * grad_accum_steps,
            "optimizer_steps": optimizer_steps,
            "best_val_rmse": best_val_rmse,
            "best_val_mae": best_val_mae,
            "best_val_loss": best_val_loss,
            "best_val_ap50_iou50": (
                None if best_val_ap50_iou50 == float("-inf") else best_val_ap50_iou50
            ),
            "best_val_f1_iou50": (
                None if best_val_f1_iou50 == float("-inf") else best_val_f1_iou50
            ),
            "best_val_gated_rmse": (
                None if best_val_gated_rmse == float("inf") else best_val_gated_rmse
            ),
            "best_val_gated_mae": (
                None if best_val_gated_mae == float("inf") else best_val_gated_mae
            ),
            "detection_gate_ratio": float(getattr(args, "detection_gate_ratio", 0.98)),
            "detection_gate_threshold": detection_f1_gate_threshold,
            "detection_f1_gate_threshold": detection_f1_gate_threshold,
            "detection_ap50_gate_threshold": detection_ap50_gate_threshold,
            "detection_gate_passed": detection_gate_passed,
            "best_val_rmse_epoch": best_val_rmse_epoch,
            "best_val_mae_epoch": best_val_mae_epoch,
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_val_ap50_iou50_epoch": best_val_ap50_iou50_epoch,
            "best_val_f1_iou50_epoch": best_val_f1_iou50_epoch,
            "best_val_gated_rmse_epoch": best_val_gated_rmse_epoch,
            "best_val_gated_mae_epoch": best_val_gated_mae_epoch,
            "train_seen": train_seen,
            "val_seen": val_seen,
            "test_seen": test_seen,
        }
        metrics.update(diagnostic_metrics(train_diag_stats, "train"))
        metrics.update(diagnostic_metrics(val_diag_stats, "val"))
        metrics.update(diagnostic_metrics(test_diag_stats, "test"))

        logger.info(
            "Epoch %d | train_loss=%.3f | val_loss=%.3f | train_mae=%.3f | "
            "val_mae=%.3f | val_rmse=%.2f | val_ap50=%s | val_f1_iou50=%s | "
            "test_mae=%.3f | test_rmse=%.2f | test_ap50=%s | test_f1_iou50=%s | "
            "time=%.3fs%s",
            epoch,
            metrics["train_loss"],
            metrics["val_loss"],
            metrics["train_mae"],
            metrics["val_mae"],
            metrics["val_rmse"],
            format_optional_metric(metrics["val_ap50_iou50"]),
            format_optional_metric(metrics["val_f1_iou50"]),
            metrics["test_mae"],
            metrics["test_rmse"],
            format_optional_metric(metrics["test_ap50_iou50"]),
            format_optional_metric(metrics["test_f1_iou50"]),
            metrics["epoch_time_sec"],
            " | best_rmse"
            if best_val_rmse_epoch
            else (
                " | best_mae"
                if best_val_mae_epoch
                else (
                    " | best_ap50"
                    if best_val_ap50_iou50_epoch
                    else (
                        " | best_f1"
                        if best_val_f1_iou50_epoch
                        else (
                            " | best_gated_rmse"
                            if best_val_gated_rmse_epoch
                            else (
                                " | best_gated_mae"
                                if best_val_gated_mae_epoch
                                else (" | best_loss" if best_val_loss_epoch else "")
                            )
                        )
                    )
                )
            ),
        )
        logger.info(
            "Epoch %d diagnostics | "
            "train pre/post/final=%d/%d/%d pred_err=%d dense=%.2f soft_nms=%.2f keep=%.3f verify_keep=%.3f "
            "thr=%.3f nms_iou=%.3f | "
            "val pre/post/final=%d/%d/%d pred_err=%d dense=%.2f soft_nms=%.2f keep=%.3f verify_keep=%.3f "
            "thr=%.3f nms_iou=%.3f | "
            "test pre/post/final=%d/%d/%d pred_err=%d dense=%.2f soft_nms=%.2f keep=%.3f verify_keep=%.3f "
            "thr=%.3f nms_iou=%.3f",
            epoch,
            metrics["train_preds_before_nms"],
            metrics["train_preds_after_nms"],
            metrics["train_preds_final"],
            metrics["train_pred_error"],
            metrics["train_dense_fraction"],
            metrics["train_soft_nms_fraction"],
            metrics["train_nms_keep_ratio"],
            metrics["train_verification_keep_ratio"],
            metrics["train_effective_threshold_mean"],
            metrics["train_effective_nms_iou_mean"],
            metrics["val_preds_before_nms"],
            metrics["val_preds_after_nms"],
            metrics["val_preds_final"],
            metrics["val_pred_error"],
            metrics["val_dense_fraction"],
            metrics["val_soft_nms_fraction"],
            metrics["val_nms_keep_ratio"],
            metrics["val_verification_keep_ratio"],
            metrics["val_effective_threshold_mean"],
            metrics["val_effective_nms_iou_mean"],
            metrics["test_preds_before_nms"],
            metrics["test_preds_after_nms"],
            metrics["test_preds_final"],
            metrics["test_pred_error"],
            metrics["test_dense_fraction"],
            metrics["test_soft_nms_fraction"],
            metrics["test_nms_keep_ratio"],
            metrics["test_verification_keep_ratio"],
            metrics["test_effective_threshold_mean"],
            metrics["test_effective_nms_iou_mean"],
        )
        write_jsonl(metrics_path, metrics)


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, type=str)
    pre_args, _ = pre_parser.parse_known_args()
    parser = get_argparser()
    parser.add_argument("--config", default=pre_args.config, type=str)
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    train(args)
