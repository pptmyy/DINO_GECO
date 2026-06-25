import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arg_parser import get_argparser
from infer import (
    boxes_to_original_coordinates,
    count_valid_gt_boxes,
    get_device,
    load_dgeco,
    postprocess_outputs,
)
from src.datasets.data import FSC147DATASET, pad_collate_test
from src.utils.detection_metrics import (
    average_precision_from_records,
    detection_ap_records,
    detection_summary,
    greedy_detection_counts,
)
from tools.diagnostics import SplitSummary, write_csv, write_json


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def valid_gt_boxes_norm(gt_bboxes_i: torch.Tensor, image_size: int) -> torch.Tensor:
    if gt_bboxes_i.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    gt_bboxes_i = gt_bboxes_i.reshape(-1, 4).float()
    valid_mask = torch.logical_not((gt_bboxes_i == 0).all(dim=1))
    return gt_bboxes_i[valid_mask] / float(image_size)


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, type=str)
    pre_args, _ = pre_parser.parse_known_args()

    parent = get_argparser()
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            parent.set_defaults(**json.load(f))

    parser = argparse.ArgumentParser(
        "Sweep DGECO postprocess and verification parameters",
        parents=[parent],
        conflict_handler="resolve",
    )
    parser.add_argument("--config", default=pre_args.config, type=str)
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to checkpoint. Defaults to model_path/model_name.pth.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs/eval_runs/postprocess_sweep", type=str)
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument("--max-combinations", default=256, type=int)
    parser.add_argument("--score-thresholds", default=None, type=str)
    parser.add_argument("--score-ratios", default=None, type=str)
    parser.add_argument("--nms-ious", default=None, type=str)
    parser.add_argument("--nms-methods", default=None, type=str)
    parser.add_argument("--soft-nms-sigmas", default=None, type=str)
    parser.add_argument("--soft-nms-score-thresholds", default=None, type=str)
    parser.add_argument("--threshold-modes", default=None, type=str)
    parser.add_argument("--score-quantiles", default=None, type=str)
    parser.add_argument("--verification-modes", default=None, type=str)
    parser.add_argument("--verification-thresholds", default=None, type=str)
    parser.add_argument("--verification-topks", default=None, type=str)
    parser.add_argument("--adaptive-sparse-score-ratios", default=None, type=str)
    parser.add_argument("--adaptive-dense-score-ratios", default=None, type=str)
    parser.add_argument("--adaptive-sparse-nms-ious", default=None, type=str)
    parser.add_argument("--adaptive-dense-nms-ious", default=None, type=str)
    parser.add_argument("--adaptive-dense-candidate-thresholds", default=None, type=str)
    parser.add_argument("--verification-filter-modes", default=None, type=str)
    parser.add_argument("--verification-score-gammas", default=None, type=str)
    parser.add_argument("--verification-hard-candidate-limits", default=None, type=str)
    parser.add_argument("--top-k-errors", default=50, type=int)
    parser.add_argument("--no-save-top-errors", action="store_true")
    return parser


def make_dataset(args):
    dataset = FSC147DATASET(
        args.data_path,
        args.image_size,
        split=args.split,
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
        training=False,
        allow_missing_coco=args.allow_missing_coco,
        exemplar_scale_mode=args.exemplar_scale_mode,
    )
    if args.split == "train":
        dataset.split = "test"
    return dataset


def build_combinations(args) -> List[Dict[str, object]]:
    score_thresholds = (
        parse_float_list(args.score_thresholds)
        if args.score_thresholds
        else [float(args.score_threshold)]
    )
    score_ratios = (
        parse_float_list(args.score_ratios)
        if args.score_ratios
        else [float(args.score_ratio)]
    )
    nms_ious = parse_float_list(args.nms_ious) if args.nms_ious else [float(args.nms_iou)]
    nms_methods = (
        parse_str_list(args.nms_methods)
        if args.nms_methods
        else [str(args.nms_method)]
    )
    soft_nms_sigmas = (
        parse_float_list(args.soft_nms_sigmas)
        if args.soft_nms_sigmas
        else [float(args.soft_nms_sigma)]
    )
    soft_nms_score_thresholds = (
        parse_float_list(args.soft_nms_score_thresholds)
        if args.soft_nms_score_thresholds
        else [float(args.soft_nms_score_threshold)]
    )
    threshold_modes = (
        parse_str_list(args.threshold_modes)
        if args.threshold_modes
        else [str(args.threshold_mode)]
    )
    score_quantiles = (
        parse_float_list(args.score_quantiles)
        if args.score_quantiles
        else [float(args.score_quantile)]
    )
    verification_modes = (
        parse_str_list(args.verification_modes)
        if args.verification_modes
        else [str(args.verification_mode)]
    )
    verification_thresholds = (
        parse_float_list(args.verification_thresholds)
        if args.verification_thresholds
        else [float(args.verification_threshold)]
    )
    verification_topks = (
        parse_int_list(args.verification_topks)
        if args.verification_topks
        else [int(args.verification_topk)]
    )
    adaptive_sparse_score_ratios = (
        parse_float_list(args.adaptive_sparse_score_ratios)
        if args.adaptive_sparse_score_ratios
        else [float(args.adaptive_sparse_score_ratio)]
    )
    adaptive_dense_score_ratios = (
        parse_float_list(args.adaptive_dense_score_ratios)
        if args.adaptive_dense_score_ratios
        else [float(args.adaptive_dense_score_ratio)]
    )
    adaptive_sparse_nms_ious = (
        parse_float_list(args.adaptive_sparse_nms_ious)
        if args.adaptive_sparse_nms_ious
        else [float(args.adaptive_sparse_nms_iou)]
    )
    adaptive_dense_nms_ious = (
        parse_float_list(args.adaptive_dense_nms_ious)
        if args.adaptive_dense_nms_ious
        else [float(args.adaptive_dense_nms_iou)]
    )
    adaptive_dense_candidate_thresholds = (
        parse_int_list(args.adaptive_dense_candidate_thresholds)
        if args.adaptive_dense_candidate_thresholds
        else [int(args.adaptive_dense_candidate_threshold)]
    )
    verification_filter_modes = (
        parse_str_list(args.verification_filter_modes)
        if args.verification_filter_modes
        else [str(args.verification_filter_mode)]
    )
    verification_score_gammas = (
        parse_float_list(args.verification_score_gammas)
        if args.verification_score_gammas
        else [float(args.verification_score_gamma)]
    )
    verification_hard_candidate_limits = (
        parse_int_list(args.verification_hard_candidate_limits)
        if args.verification_hard_candidate_limits
        else [int(args.verification_hard_candidate_limit)]
    )

    combos: List[Dict[str, object]] = []
    for threshold_mode in threshold_modes:
        quantiles = score_quantiles if threshold_mode == "quantile" else [score_quantiles[0]]
        if threshold_mode == "regime_adaptive":
            adaptive_values = list(itertools.product(
                adaptive_sparse_score_ratios,
                adaptive_dense_score_ratios,
                adaptive_sparse_nms_ious,
                adaptive_dense_nms_ious,
                adaptive_dense_candidate_thresholds,
            ))
        else:
            adaptive_values = [
                (
                    float(args.adaptive_sparse_score_ratio),
                    float(args.adaptive_dense_score_ratio),
                    float(args.adaptive_sparse_nms_iou),
                    float(args.adaptive_dense_nms_iou),
                    int(args.adaptive_dense_candidate_threshold),
                )
            ]
        for verification_mode in verification_modes:
            thresholds = verification_thresholds if verification_mode != "none" else [0.0]
            topks = verification_topks if verification_mode != "none" else [0]
            filter_modes = verification_filter_modes if verification_mode != "none" else ["hard"]
            score_gammas = verification_score_gammas if verification_mode != "none" else [0.0]
            hard_candidate_limits = (
                verification_hard_candidate_limits if verification_mode != "none" else [0]
            )
            for adaptive_value in adaptive_values:
                (
                    adaptive_sparse_score_ratio,
                    adaptive_dense_score_ratio,
                    adaptive_sparse_nms_iou,
                    adaptive_dense_nms_iou,
                    adaptive_dense_candidate_threshold,
                ) = adaptive_value
                for values in itertools.product(
                    score_thresholds,
                    score_ratios,
                    nms_ious,
                    nms_methods,
                    soft_nms_sigmas,
                    soft_nms_score_thresholds,
                    quantiles,
                    thresholds,
                    topks,
                    filter_modes,
                    score_gammas,
                    hard_candidate_limits,
                ):
                    (
                        score_threshold,
                        score_ratio,
                        nms_iou,
                        nms_method,
                        soft_nms_sigma,
                        soft_nms_score_threshold,
                        score_quantile,
                        verification_threshold,
                        verification_topk,
                        verification_filter_mode,
                        verification_score_gamma,
                        verification_hard_candidate_limit,
                    ) = values
                    combos.append(
                        {
                            "score_threshold": float(score_threshold),
                            "score_ratio": float(score_ratio),
                            "nms_iou": float(nms_iou),
                            "nms_method": str(nms_method),
                            "soft_nms_sigma": float(soft_nms_sigma),
                            "soft_nms_score_threshold": float(soft_nms_score_threshold),
                            "threshold_mode": threshold_mode,
                            "score_quantile": float(score_quantile),
                            "adaptive_sparse_score_ratio": float(adaptive_sparse_score_ratio),
                            "adaptive_dense_score_ratio": float(adaptive_dense_score_ratio),
                            "adaptive_sparse_nms_iou": float(adaptive_sparse_nms_iou),
                            "adaptive_dense_nms_iou": float(adaptive_dense_nms_iou),
                            "adaptive_dense_candidate_threshold": int(
                                adaptive_dense_candidate_threshold
                            ),
                            "verification_mode": verification_mode,
                            "verification_threshold": float(verification_threshold),
                            "verification_topk": int(verification_topk),
                            "verification_filter_mode": str(verification_filter_mode),
                            "verification_score_gamma": float(verification_score_gamma),
                            "verification_hard_candidate_limit": int(
                                verification_hard_candidate_limit
                            ),
                        }
                    )

    for idx, combo in enumerate(combos):
        combo["combo_id"] = idx

    if len(combos) > int(args.max_combinations):
        raise ValueError(
            f"Sweep has {len(combos)} combinations, above --max-combinations={args.max_combinations}. "
            "Narrow the grid or raise the limit intentionally."
        )
    return combos


def combo_row(combo: Dict[str, object], summary: SplitSummary, split: str) -> Dict[str, object]:
    row = {"combo_id": combo["combo_id"], "split": split}
    row.update(combo)
    row.update(summary.overall.to_dict())
    return row


def bin_rows(combo: Dict[str, object], summary: SplitSummary, split: str) -> List[Dict[str, object]]:
    rows = []
    for name, acc in summary.bins.items():
        row = {"combo_id": combo["combo_id"], "split": split, "count_bin": name}
        row.update(combo)
        row.update(acc.to_dict())
        rows.append(row)
    return rows


def pareto_front(rows: List[Dict[str, object]], max_items: int = 20) -> List[Dict[str, object]]:
    front: List[Dict[str, object]] = []
    best_rmse = float("inf")
    for row in sorted(rows, key=lambda item: (float(item["mae"]), float(item["rmse"]))):
        rmse = float(row["rmse"])
        if rmse < best_rmse:
            front.append(row)
            best_rmse = rmse
        if len(front) >= max_items:
            break
    return front


def top_error_rows(
    combos: List[Dict[str, object]],
    records_by_combo: Dict[int, List[Dict[str, object]]],
    top_k: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for combo in combos:
        combo_id = int(combo["combo_id"])
        records = sorted(
            records_by_combo.get(combo_id, []),
            key=lambda item: float(item["abs_error"]),
            reverse=True,
        )[: int(top_k)]
        for rank, record in enumerate(records, start=1):
            row = {"combo_id": combo_id, "rank": rank}
            row.update(combo)
            row.update(record)
            rows.append(row)
    return rows


def run_sweep(args) -> Dict[str, object]:
    combos = build_combinations(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    need_feature_similarity = any(
        combo["verification_mode"] == "feature_similarity" for combo in combos
    )
    original_verification_mode = args.verification_mode
    if need_feature_similarity:
        args.verification_mode = "feature_similarity"

    write_json(
        output_dir / "args.json",
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "args": vars(args),
            "original_verification_mode": original_verification_mode,
            "combinations": combos,
        },
    )

    device = get_device(args)
    model = load_dgeco(args, device)
    dataset = make_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate_test,
    )

    summaries = {int(combo["combo_id"]): SplitSummary() for combo in combos}
    records_by_combo = {int(combo["combo_id"]): [] for combo in combos}
    detection_totals = {
        int(combo["combo_id"]): {"tp": 0, "fp": 0, "fn": 0}
        for combo in combos
    }
    detection_ap_records_by_combo = {int(combo["combo_id"]): [] for combo in combos}
    detection_gt_counts = {int(combo["combo_id"]): 0 for combo in combos}
    seen = 0

    for batch in loader:
        img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh = batch
        img = img.to(device)
        bboxes = bboxes.to(device)

        with torch.inference_mode():
            outputs, _, _, _, _aux = model(img, bboxes)

        for idx in range(img.shape[0]):
            dataset_idx = int(ids[idx].item())
            image_name = dataset.image_names[dataset_idx]
            image_path = Path(dataset.data_path) / dataset.image_dir / image_name
            with Image.open(image_path) as pil_image:
                original_size = pil_image.size

            gt_count = count_valid_gt_boxes(gt_bboxes[idx])
            target_boxes_norm = valid_gt_boxes_norm(gt_bboxes[idx], args.image_size)
            density_sum_debug = float(density_map[idx].sum().item())
            exemplar_boxes = bboxes[idx] / float(args.image_size)

            for combo in combos:
                boxes_norm, scores, postprocess_stats, verification_stats = postprocess_outputs(
                    outputs[idx],
                    score_threshold=float(combo["score_threshold"]),
                    score_ratio=float(combo["score_ratio"]),
                    threshold_mode=str(combo["threshold_mode"]),
                    score_quantile=float(combo["score_quantile"]),
                    min_score_gap=args.min_score_gap,
                    pre_nms_topk=args.pre_nms_topk,
                    max_detections=args.max_detections,
                    nms_iou=float(combo["nms_iou"]),
                    nms_method=str(combo["nms_method"]),
                    soft_nms_sigma=float(combo["soft_nms_sigma"]),
                    soft_nms_score_threshold=float(combo["soft_nms_score_threshold"]),
                    min_box_area=args.min_box_area,
                    max_box_area=args.max_box_area,
                    adaptive_sparse_score_ratio=float(combo["adaptive_sparse_score_ratio"]),
                    adaptive_dense_score_ratio=float(combo["adaptive_dense_score_ratio"]),
                    adaptive_sparse_nms_iou=float(combo["adaptive_sparse_nms_iou"]),
                    adaptive_dense_nms_iou=float(combo["adaptive_dense_nms_iou"]),
                    adaptive_dense_candidate_threshold=int(
                        combo["adaptive_dense_candidate_threshold"]
                    ),
                    exemplar_boxes=exemplar_boxes,
                    verification_mode=str(combo["verification_mode"]),
                    verification_threshold=float(combo["verification_threshold"]),
                    verification_topk=int(combo["verification_topk"]),
                    verification_min_area_ratio=args.verification_min_area_ratio,
                    verification_max_area_ratio=args.verification_max_area_ratio,
                    verification_filter_mode=str(combo["verification_filter_mode"]),
                    verification_score_gamma=float(combo["verification_score_gamma"]),
                    verification_hard_candidate_limit=int(
                        combo["verification_hard_candidate_limit"]
                    ),
                    return_stats=True,
                )
                boxes_orig, _keep_original = boxes_to_original_coordinates(
                    boxes_norm.detach().cpu(),
                    image_size=args.image_size,
                    scaling_factor=float(scaling_factor[idx].item()),
                    padwh=padwh[idx].tolist(),
                    original_size=original_size,
                )
                pred_count = int(boxes_orig.shape[0])
                combo_id = int(combo["combo_id"])
                boxes_eval = boxes_norm.detach().cpu()[_keep_original]
                scores_eval = scores.detach().cpu()[_keep_original]
                tp, fp, fn = greedy_detection_counts(
                    boxes_eval,
                    target_boxes_norm,
                    iou_threshold=getattr(args, "val_iou_threshold", 0.5),
                )
                detection_totals[combo_id]["tp"] += tp
                detection_totals[combo_id]["fp"] += fp
                detection_totals[combo_id]["fn"] += fn
                detection_gt_counts[combo_id] += int(target_boxes_norm.shape[0])
                detection_ap_records_by_combo[combo_id].extend(
                    detection_ap_records(
                        boxes_eval,
                        scores_eval,
                        target_boxes_norm,
                        iou_threshold=getattr(args, "val_iou_threshold", 0.5),
                    )
                )
                postprocess_dict = postprocess_stats.to_dict()
                verification_dict = verification_stats.to_dict()
                summaries[combo_id].update(
                    gt_count=gt_count,
                    pred_count=pred_count,
                    postprocess=postprocess_dict,
                    verification=verification_dict,
                    density_sum_debug=density_sum_debug,
                )
                if not args.no_save_top_errors:
                    records_by_combo[combo_id].append(
                        {
                            "image": image_name,
                            "gt_count": gt_count,
                            "pred_count": pred_count,
                            "error": pred_count - gt_count,
                            "abs_error": abs(pred_count - gt_count),
                            "preds_before_nms": postprocess_dict.get("preds_before_nms"),
                            "preds_after_nms": postprocess_dict.get("preds_after_nms"),
                            "post_verify": verification_dict.get("kept_count"),
                            "effective_threshold": postprocess_dict.get("effective_threshold"),
                            "effective_score_ratio": postprocess_dict.get(
                                "effective_score_ratio"
                            ),
                            "effective_nms_iou": postprocess_dict.get("effective_nms_iou"),
                            "adaptive_regime": postprocess_dict.get("adaptive_regime"),
                            "nms_method": postprocess_dict.get("effective_nms_method"),
                            "verification_score_mean": verification_dict.get(
                                "verification_score_mean"
                            ),
                            "hard_filter_applied": verification_dict.get(
                                "hard_filter_applied"
                            ),
                            "density_sum_debug": density_sum_debug,
                        }
                    )

            seen += 1
            if args.max_images is not None and seen >= args.max_images:
                break

        if args.max_images is not None and seen >= args.max_images:
            break

    detection_rows = {
        combo_id: detection_summary(
            totals["tp"],
            totals["fp"],
            totals["fn"],
            ap50=average_precision_from_records(
                detection_ap_records_by_combo[combo_id],
                detection_gt_counts[combo_id],
            ),
        )
        for combo_id, totals in detection_totals.items()
    }
    sweep_rows = []
    for combo in combos:
        combo_id = int(combo["combo_id"])
        row = combo_row(combo, summaries[combo_id], args.split)
        row.update(detection_rows[combo_id])
        sweep_rows.append(row)
    sweep_rows.sort(key=lambda row: (float(row["mae"]), float(row["rmse"])))
    count_bin_rows: List[Dict[str, object]] = []
    for combo in combos:
        count_bin_rows.extend(bin_rows(combo, summaries[int(combo["combo_id"])], args.split))

    write_csv(output_dir / "postprocess_sweep.csv", sweep_rows)
    write_csv(output_dir / "summary_by_count_bin.csv", count_bin_rows)
    if not args.no_save_top_errors:
        write_csv(
            output_dir / "top_errors_by_combo.csv",
            top_error_rows(combos, records_by_combo, args.top_k_errors),
        )
    best_by_rmse = min(sweep_rows, key=lambda row: float(row["rmse"])) if sweep_rows else None
    best_detection_f1 = max(
        (float(row["f1_iou50"]) for row in sweep_rows),
        default=0.0,
    )
    best_detection_ap50 = max(
        (float(row["ap50"]) for row in sweep_rows),
        default=0.0,
    )
    gate_threshold = best_detection_f1 * float(getattr(args, "detection_gate_ratio", 0.98))
    ap50_gate_threshold = best_detection_ap50 * float(getattr(args, "detection_gate_ratio", 0.98))
    gated_rows = [
        row
        for row in sweep_rows
        if float(row["f1_iou50"]) >= gate_threshold
        and float(row["ap50"]) >= ap50_gate_threshold
    ]
    best_gated_by_mae = gated_rows[0] if gated_rows else None
    best_gated_by_rmse = (
        min(gated_rows, key=lambda row: float(row["rmse"])) if gated_rows else None
    )
    result = {
        "split": args.split,
        "checkpoint": args.checkpoint or str(Path(args.model_path) / f"{args.model_name}.pth"),
        "image_count": seen,
        "combination_count": len(combos),
        "best_by_mae": sweep_rows[0] if sweep_rows else None,
        "best_by_rmse": best_by_rmse,
        "best_detection_f1_iou50": best_detection_f1,
        "best_detection_ap50": best_detection_ap50,
        "detection_gate_ratio": float(getattr(args, "detection_gate_ratio", 0.98)),
        "detection_gate_threshold": gate_threshold,
        "detection_f1_gate_threshold": gate_threshold,
        "detection_ap50_gate_threshold": ap50_gate_threshold,
        "best_gated_by_mae": best_gated_by_mae,
        "best_gated_by_rmse": best_gated_by_rmse,
        "pareto_front": pareto_front(sweep_rows),
    }
    write_json(output_dir / "summary.json", result)
    coco_metrics = {
        "enabled": False,
        "reason": "postprocess sweep uses internal AP50/F1 records; COCO AP/AP75 was not run",
        "image_count": int(seen),
        "checkpoint": result["checkpoint"],
    }
    write_json(output_dir / "coco_metrics.json", coco_metrics)
    write_json(
        output_dir / "selection_report.json",
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "protocol": "P0",
            "selection_policy": (
                "First enforce the detection gate, then compare gated MAE/RMSE and "
                "count-bin signed error; keep ungated best rows for diagnostics only."
            ),
            "split": args.split,
            "checkpoint": result["checkpoint"],
            "protocol_config": {
                "model_name": args.model_name,
                "query_output_stride": int(getattr(args, "query_output_stride", 4)),
                "use_semantic_anchor": bool(getattr(args, "use_semantic_anchor", False)),
                "detection_gate_ratio": float(getattr(args, "detection_gate_ratio", 0.98)),
                "combination_count": len(combos),
            },
            "selected_rows": {
                "best_by_mae": result["best_by_mae"],
                "best_by_rmse": result["best_by_rmse"],
                "best_gated_by_mae": result["best_gated_by_mae"],
                "best_gated_by_rmse": result["best_gated_by_rmse"],
            },
            "detection_gate": {
                "best_detection_f1_iou50": result["best_detection_f1_iou50"],
                "best_detection_ap50": result["best_detection_ap50"],
                "detection_f1_gate_threshold": result["detection_f1_gate_threshold"],
                "detection_ap50_gate_threshold": result["detection_ap50_gate_threshold"],
            },
            "artifacts": {
                "args": str(output_dir / "args.json"),
                "summary": str(output_dir / "summary.json"),
                "summary_by_count_bin": str(output_dir / "summary_by_count_bin.csv"),
                "postprocess_sweep": str(output_dir / "postprocess_sweep.csv"),
                "top_abs_errors": str(output_dir / "top_errors_by_combo.csv"),
                "coco_metrics": str(output_dir / "coco_metrics.json"),
                "selection_report": str(output_dir / "selection_report.json"),
            },
        },
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run_sweep(args)
    best = result.get("best_by_mae") or {}
    print(
        "Saved sweep to %s | combinations %d | best combo %s MAE %.3f RMSE %.3f signed %.3f"
        % (
            args.output_dir,
            result["combination_count"],
            best.get("combo_id"),
            float(best.get("mae", 0.0)),
            float(best.get("rmse", 0.0)),
            float(best.get("signed_error", 0.0)),
        )
    )


if __name__ == "__main__":
    main()
