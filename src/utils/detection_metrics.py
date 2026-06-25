from typing import Dict, Sequence, Tuple

import torch

from .box_ops import box_iou


def greedy_detection_counts(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    pred_count = int(pred_boxes.shape[0])
    target_count = int(target_boxes.shape[0])
    if pred_count == 0:
        return 0, 0, target_count
    if target_count == 0:
        return 0, pred_count, 0

    ious, _ = box_iou(pred_boxes.detach().float(), target_boxes.detach().float())
    tp = 0
    used_pred = set()
    used_gt = set()
    while ious.numel() > 0:
        max_value, flat_idx = torch.max(ious.reshape(-1), dim=0)
        if float(max_value.item()) < float(iou_threshold):
            break
        pred_idx = int(flat_idx.item() // ious.shape[1])
        gt_idx = int(flat_idx.item() % ious.shape[1])
        if pred_idx in used_pred or gt_idx in used_gt:
            ious[pred_idx, gt_idx] = -1.0
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        ious[pred_idx, :] = -1.0
        ious[:, gt_idx] = -1.0
        tp += 1
    fp = pred_count - tp
    fn = target_count - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = precision + recall
    f1 = 0.0 if denom <= 0 else 2.0 * precision * recall / denom
    return precision, recall, f1


def detection_ap_records(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> list[tuple[float, int]]:
    pred_count = int(pred_boxes.shape[0])
    target_count = int(target_boxes.shape[0])
    if pred_count == 0:
        return []

    scores = pred_scores.detach().reshape(-1).float()
    order = torch.argsort(scores, descending=True)
    records: list[tuple[float, int]] = []

    if target_count == 0:
        return [(float(scores[idx].item()), 0) for idx in order]

    boxes = pred_boxes.detach().float()[order]
    target_boxes = target_boxes.detach().float()
    ious, _ = box_iou(boxes, target_boxes)
    matched_gt = torch.zeros(target_count, dtype=torch.bool, device=ious.device)

    for rank, pred_idx in enumerate(order):
        sorted_iou, sorted_gt = torch.sort(ious[rank], descending=True)
        is_tp = False
        for best_iou, gt_idx in zip(sorted_iou, sorted_gt):
            if float(best_iou.item()) < float(iou_threshold):
                break
            if not bool(matched_gt[gt_idx].item()):
                matched_gt[gt_idx] = True
                is_tp = True
                break
        records.append((float(scores[pred_idx].item()), int(is_tp)))
    return records


def average_precision_from_records(
    records: Sequence[tuple[float, int]],
    target_count: int,
) -> float:
    if target_count <= 0 or not records:
        return 0.0

    ordered = sorted(records, key=lambda item: item[0], reverse=True)
    tp_cum = 0.0
    fp_cum = 0.0
    recalls = []
    precisions = []
    for _score, is_tp in ordered:
        if is_tp:
            tp_cum += 1.0
        else:
            fp_cum += 1.0
        recalls.append(tp_cum / float(target_count))
        precisions.append(tp_cum / max(tp_cum + fp_cum, 1.0))

    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])

    ap = 0.0
    for idx in range(len(mrec) - 1):
        if mrec[idx + 1] != mrec[idx]:
            ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]
    return float(ap)


def detection_summary(
    tp: int,
    fp: int,
    fn: int,
    *,
    ap50: float | None = None,
) -> Dict[str, object]:
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    summary: Dict[str, object] = {
        "det_tp_iou50": int(tp),
        "det_fp_iou50": int(fp),
        "det_fn_iou50": int(fn),
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": f1,
    }
    if ap50 is not None:
        summary["ap50"] = float(ap50)
    return summary
