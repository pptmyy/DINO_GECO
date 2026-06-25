import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
    get_instances_path,
    load_dgeco,
    make_coco_detections,
    postprocess_outputs,
    run_coco_bbox_eval,
)
from src.datasets.data import FSC147DATASET, pad_collate_test
from tools.diagnostics import (
    SplitSummary,
    write_count_bin_csv,
    write_json,
    write_top_errors_csv,
)


CHECKPOINT_META_KEYS = (
    "epoch",
    "best_val_rmse",
    "best_val_mae",
    "best_val_loss",
    "best_val_ap50_iou50",
    "best_val_f1_iou50",
    "best_val_gated_rmse",
    "best_val_gated_mae",
    "val_mae",
    "val_rmse",
    "val_loss",
    "val_ap50_iou50",
    "val_f1_iou50",
    "test_mae",
    "test_rmse",
    "test_ap50_iou50",
    "test_f1_iou50",
    "query_output_stride",
    "use_semantic_anchor",
)


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, type=str)
    pre_args, _ = pre_parser.parse_known_args()

    parent = get_argparser()
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            parent.set_defaults(**json.load(f))

    parser = argparse.ArgumentParser(
        "Evaluate a DGECO checkpoint with per-image diagnostics",
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
    parser.add_argument("--output-dir", default="outputs/eval_runs/eval_checkpoint", type=str)
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument("--top-k-errors", default=50, type=int)
    parser.add_argument("--no-save-boxes", action="store_true")
    parser.add_argument("--coco-eval", action="store_true")
    parser.add_argument("--coco-category-id", default=1, type=int)
    parser.add_argument("--coco-max-dets", default=1000, type=int)
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


def build_selection_report(
    args,
    *,
    summary: Dict[str, object],
    model: torch.nn.Module,
    coco_metrics: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, object]:
    checkpoint = args.checkpoint or str(Path(args.model_path) / f"{args.model_name}.pth")
    checkpoint_meta = getattr(model, "_dgeco_checkpoint_meta", {}) or {}
    selected_checkpoint_meta = {
        key: checkpoint_meta.get(key)
        for key in CHECKPOINT_META_KEYS
        if key in checkpoint_meta
    }
    protocol_config = {
        "model_name": args.model_name,
        "query_output_stride": int(getattr(args, "query_output_stride", 4)),
        "use_semantic_anchor": bool(getattr(args, "use_semantic_anchor", False)),
        "score_threshold": args.score_threshold,
        "score_ratio": args.score_ratio,
        "threshold_mode": args.threshold_mode,
        "score_quantile": args.score_quantile,
        "pre_nms_topk": args.pre_nms_topk,
        "max_detections": args.max_detections,
        "nms_iou": args.nms_iou,
        "nms_method": args.nms_method,
        "verification_mode": args.verification_mode,
        "verification_threshold": args.verification_threshold,
        "verification_filter_mode": args.verification_filter_mode,
        "coco_eval": bool(args.coco_eval),
        "coco_max_dets": args.coco_max_dets,
    }
    artifacts = {
        "args": str(output_dir / "args.json"),
        "summary": str(output_dir / "summary.json"),
        "summary_by_count_bin": str(output_dir / "summary_by_count_bin.csv"),
        "top_abs_errors": str(output_dir / "top_abs_errors.csv"),
        "per_image_predictions": str(output_dir / "per_image_predictions.jsonl"),
        "coco_metrics": str(output_dir / "coco_metrics.json"),
        "selection_report": str(output_dir / "selection_report.json"),
    }
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": "P0",
        "selection_policy": (
            "Keep count-best, detection-best, loss-best, gated-count-best, and last "
            "checkpoints separate; compare checkpoints only under the recorded protocol."
        ),
        "split": args.split,
        "checkpoint": checkpoint,
        "checkpoint_metadata": selected_checkpoint_meta,
        "protocol_config": protocol_config,
        "evaluated_metrics": {
            "overall": summary.get("overall"),
            "count_bins": summary.get("count_bins"),
            "coco": coco_metrics,
        },
        "artifacts": artifacts,
    }


def evaluate(args) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "args.json",
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "args": vars(args),
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

    summary = SplitSummary()
    records: List[Dict[str, object]] = []
    coco_detections: List[Dict[str, object]] = []
    coco_eval_image_ids: List[int] = []
    pred_path = output_dir / "per_image_predictions.jsonl"

    seen = 0
    with pred_path.open("w", encoding="utf-8") as f:
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

                exemplar_boxes = bboxes[idx] / float(args.image_size)
                boxes_norm, scores, postprocess_stats, verification_stats = postprocess_outputs(
                    outputs[idx],
                    score_threshold=args.score_threshold,
                    score_ratio=args.score_ratio,
                    threshold_mode=args.threshold_mode,
                    score_quantile=args.score_quantile,
                    min_score_gap=args.min_score_gap,
                    pre_nms_topk=args.pre_nms_topk,
                    max_detections=args.max_detections,
                    nms_iou=args.nms_iou,
                    nms_method=args.nms_method,
                    soft_nms_sigma=args.soft_nms_sigma,
                    soft_nms_score_threshold=args.soft_nms_score_threshold,
                    min_box_area=args.min_box_area,
                    max_box_area=args.max_box_area,
                    adaptive_sparse_score_ratio=args.adaptive_sparse_score_ratio,
                    adaptive_dense_score_ratio=args.adaptive_dense_score_ratio,
                    adaptive_sparse_nms_iou=args.adaptive_sparse_nms_iou,
                    adaptive_dense_nms_iou=args.adaptive_dense_nms_iou,
                    adaptive_dense_candidate_threshold=args.adaptive_dense_candidate_threshold,
                    exemplar_boxes=exemplar_boxes,
                    verification_mode=args.verification_mode,
                    verification_threshold=args.verification_threshold,
                    verification_topk=args.verification_topk,
                    verification_min_area_ratio=args.verification_min_area_ratio,
                    verification_max_area_ratio=args.verification_max_area_ratio,
                    verification_filter_mode=args.verification_filter_mode,
                    verification_score_gamma=args.verification_score_gamma,
                    verification_hard_candidate_limit=args.verification_hard_candidate_limit,
                    return_stats=True,
                )
                boxes_orig, keep_original = boxes_to_original_coordinates(
                    boxes_norm.detach().cpu(),
                    image_size=args.image_size,
                    scaling_factor=float(scaling_factor[idx].item()),
                    padwh=padwh[idx].tolist(),
                    original_size=original_size,
                )
                scores_cpu = scores.detach().cpu()[keep_original]
                boxes_list = boxes_orig.tolist()
                scores_list = scores_cpu.tolist()

                gt_count = count_valid_gt_boxes(gt_bboxes[idx])
                pred_count = len(boxes_list)
                density_sum_debug = float(density_map[idx].sum().item())
                postprocess_dict = postprocess_stats.to_dict()
                verification_dict = verification_stats.to_dict()
                record = {
                    "image": image_name,
                    "image_index": dataset_idx,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "count": pred_count,
                    "error": pred_count - gt_count,
                    "abs_error": abs(pred_count - gt_count),
                    "gt_count_source": "gt_bboxes",
                    "density_sum_debug": density_sum_debug,
                    "postprocess": postprocess_dict,
                    "verification": verification_dict,
                }
                if not args.no_save_boxes:
                    record["boxes"] = boxes_list
                    record["scores"] = scores_list

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                summary.update(
                    gt_count=gt_count,
                    pred_count=pred_count,
                    postprocess=postprocess_dict,
                    verification=verification_dict,
                    density_sum_debug=density_sum_debug,
                )

                image_id = dataset.img_name_to_ori_id.get(image_name, dataset_idx)
                if args.coco_eval:
                    coco_eval_image_ids.append(int(image_id))
                    coco_detections.extend(
                        make_coco_detections(
                            image_id=image_id,
                            boxes_xyxy=boxes_list,
                            scores=scores_list,
                            category_id=args.coco_category_id,
                        )
                    )

                seen += 1
                if args.max_images is not None and seen >= args.max_images:
                    break

            if args.max_images is not None and seen >= args.max_images:
                break

    summary_dict = summary.to_dict()
    summary_dict["split"] = args.split
    summary_dict["checkpoint"] = args.checkpoint or str(Path(args.model_path) / f"{args.model_name}.pth")
    write_json(output_dir / "summary.json", summary_dict)
    write_count_bin_csv(output_dir / "summary_by_count_bin.csv", summary)
    write_top_errors_csv(output_dir / "top_abs_errors.csv", records, args.top_k_errors)

    coco_metrics: Dict[str, Any]
    if args.coco_eval:
        coco_metrics = run_coco_bbox_eval(
            instances_path=get_instances_path(dataset, args.split),
            detections=coco_detections,
            image_ids=coco_eval_image_ids,
            output_dir=output_dir,
            category_id=args.coco_category_id,
            max_dets=args.coco_max_dets,
        )
    else:
        coco_metrics = {
            "enabled": False,
            "reason": "--coco-eval was not set",
            "image_count": int(seen),
            "detection_count": int(len(coco_detections)),
            "category_id": int(args.coco_category_id),
            "requested_max_dets": int(args.coco_max_dets),
        }
        write_json(output_dir / "coco_metrics.json", coco_metrics)

    write_json(
        output_dir / "selection_report.json",
        build_selection_report(
            args,
            summary=summary_dict,
            model=model,
            coco_metrics=coco_metrics,
            output_dir=output_dir,
        ),
    )

    return summary_dict


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(args)
    overall = summary["overall"]
    print(
        "Saved diagnostics to %s | %s MAE %.3f RMSE %.3f signed %.3f"
        % (
            args.output_dir,
            args.split,
            overall["mae"],
            overall["rmse"],
            overall["signed_error"],
        )
    )


if __name__ == "__main__":
    main()
