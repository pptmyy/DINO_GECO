from dataclasses import asdict, dataclass
from typing import Dict, Tuple, Union

import torch
from torchvision import ops


@dataclass
class PostprocessStats:
    threshold_mode: str
    score_threshold: float
    score_ratio: float
    score_quantile: float
    min_score_gap: float
    effective_threshold: float
    score_min: float
    score_max: float
    score_mean: float
    preds_total: int
    preds_after_threshold: int
    preds_after_shape: int
    preds_after_area: int
    preds_before_nms: int
    preds_after_nms: int
    preds_final: int
    pre_nms_topk: int
    max_detections: int
    nms_iou: float
    min_box_area: float
    max_box_area: float
    adaptive_regime: str
    adaptive_candidate_count: int
    effective_score_ratio: float
    effective_nms_iou: float

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return asdict(self)


def _empty_stats(
    *,
    threshold_mode: str,
    score_threshold: float,
    score_ratio: float,
    score_quantile: float,
    min_score_gap: float,
    effective_threshold: float,
    score_min: float,
    score_max: float,
    score_mean: float,
    preds_total: int,
    preds_after_threshold: int = 0,
    preds_after_shape: int = 0,
    preds_after_area: int = 0,
    preds_before_nms: int = 0,
    preds_after_nms: int = 0,
    preds_final: int = 0,
    pre_nms_topk: int,
    max_detections: int,
    nms_iou: float,
    min_box_area: float,
    max_box_area: float,
    adaptive_regime: str = "none",
    adaptive_candidate_count: int = 0,
    effective_score_ratio: float = 0.0,
    effective_nms_iou: float = 0.0,
) -> PostprocessStats:
    effective_nms_iou = float(effective_nms_iou) if effective_nms_iou > 0 else float(nms_iou)
    effective_score_ratio = (
        float(effective_score_ratio) if effective_score_ratio > 0 else float(score_ratio)
    )
    return PostprocessStats(
        threshold_mode=threshold_mode,
        score_threshold=float(score_threshold),
        score_ratio=float(score_ratio),
        score_quantile=float(score_quantile),
        min_score_gap=float(min_score_gap),
        effective_threshold=float(effective_threshold),
        score_min=float(score_min),
        score_max=float(score_max),
        score_mean=float(score_mean),
        preds_total=int(preds_total),
        preds_after_threshold=int(preds_after_threshold),
        preds_after_shape=int(preds_after_shape),
        preds_after_area=int(preds_after_area),
        preds_before_nms=int(preds_before_nms),
        preds_after_nms=int(preds_after_nms),
        preds_final=int(preds_final),
        pre_nms_topk=int(pre_nms_topk),
        max_detections=int(max_detections),
        nms_iou=float(nms_iou),
        min_box_area=float(min_box_area),
        max_box_area=float(max_box_area),
        adaptive_regime=str(adaptive_regime),
        adaptive_candidate_count=int(adaptive_candidate_count),
        effective_score_ratio=effective_score_ratio,
        effective_nms_iou=effective_nms_iou,
    )


def _get_pred_boxes(output_i: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "pred_boxes" in output_i:
        return output_i["pred_boxes"]
    if "boxes" in output_i:
        return output_i["boxes"]
    raise KeyError("output_i must contain 'pred_boxes' or 'boxes'")


def _effective_threshold(
    scores: torch.Tensor,
    *,
    threshold_mode: str,
    score_threshold: float,
    score_ratio: float,
    score_quantile: float,
    min_score_gap: float,
) -> torch.Tensor:
    if threshold_mode == "static_ratio":
        threshold = scores.max() * float(score_ratio)
        threshold = torch.clamp(threshold, min=float(score_threshold), max=1.0)
    elif threshold_mode == "quantile":
        threshold = torch.quantile(scores, float(score_quantile))
        threshold = torch.clamp(threshold, min=float(score_threshold), max=1.0)
    else:
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode!r}")

    if min_score_gap > 0:
        threshold = torch.maximum(threshold, scores.max() - float(min_score_gap))
    return torch.clamp(threshold, min=0.0, max=1.0)


def _threshold_plan(
    scores: torch.Tensor,
    *,
    threshold_mode: str,
    score_threshold: float,
    score_ratio: float,
    score_quantile: float,
    min_score_gap: float,
    nms_iou: float,
    adaptive_sparse_score_ratio: float,
    adaptive_dense_score_ratio: float,
    adaptive_sparse_nms_iou: float,
    adaptive_dense_nms_iou: float,
    adaptive_dense_candidate_threshold: int,
) -> Tuple[torch.Tensor, float, str, int, float]:
    if threshold_mode != "regime_adaptive":
        threshold = _effective_threshold(
            scores,
            threshold_mode=threshold_mode,
            score_threshold=score_threshold,
            score_ratio=score_ratio,
            score_quantile=score_quantile,
            min_score_gap=min_score_gap,
        )
        return threshold, float(nms_iou), "none", 0, float(score_ratio)

    base_threshold = _effective_threshold(
        scores,
        threshold_mode="static_ratio",
        score_threshold=score_threshold,
        score_ratio=score_ratio,
        score_quantile=score_quantile,
        min_score_gap=0.0,
    )
    candidate_count = int((scores >= base_threshold).sum().item())
    if candidate_count >= int(adaptive_dense_candidate_threshold):
        regime = "dense"
        effective_score_ratio = float(adaptive_dense_score_ratio)
        effective_nms_iou = float(adaptive_dense_nms_iou)
    else:
        regime = "sparse"
        effective_score_ratio = float(adaptive_sparse_score_ratio)
        effective_nms_iou = float(adaptive_sparse_nms_iou)

    threshold = _effective_threshold(
        scores,
        threshold_mode="static_ratio",
        score_threshold=score_threshold,
        score_ratio=effective_score_ratio,
        score_quantile=score_quantile,
        min_score_gap=min_score_gap,
    )
    return threshold, effective_nms_iou, regime, candidate_count, effective_score_ratio


def filter_detections(
    output_i: Dict[str, torch.Tensor],
    *,
    score_threshold: float = 0.20,
    score_ratio: float = 0.50,
    threshold_mode: str = "static_ratio",
    score_quantile: float = 0.98,
    min_score_gap: float = 0.0,
    pre_nms_topk: int = 4096,
    max_detections: int = 4096,
    nms_iou: float = 0.30,
    min_box_area: float = 0.0,
    max_box_area: float = 0.0,
    adaptive_sparse_score_ratio: float = 0.50,
    adaptive_dense_score_ratio: float = 0.45,
    adaptive_sparse_nms_iou: float = 0.25,
    adaptive_dense_nms_iou: float = 0.25,
    adaptive_dense_candidate_threshold: int = 128,
    return_stats: bool = False,
    return_indices: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, PostprocessStats],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, PostprocessStats, torch.Tensor],
]:
    """Filter point-generated boxes with configurable thresholding and NMS."""
    def make_result(
        boxes_out: torch.Tensor,
        scores_out: torch.Tensor,
        stats_out: PostprocessStats,
        indices_out: torch.Tensor,
    ):
        result = [boxes_out, scores_out]
        if return_stats:
            result.append(stats_out)
        if return_indices:
            result.append(indices_out)
        return tuple(result)

    pred_boxes = _get_pred_boxes(output_i).reshape(-1, 4).float()
    logits = output_i["box_v"].reshape(-1).float()
    original_indices = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)

    if pred_boxes.numel() == 0 or logits.numel() == 0:
        boxes = pred_boxes.new_zeros((0, 4))
        scores = logits.new_zeros((0,))
        indices = original_indices.new_zeros((0,))
        stats = _empty_stats(
            threshold_mode=threshold_mode,
            score_threshold=score_threshold,
            score_ratio=score_ratio,
            score_quantile=score_quantile,
            min_score_gap=min_score_gap,
            effective_threshold=float(score_threshold),
            score_min=0.0,
            score_max=0.0,
            score_mean=0.0,
            preds_total=0,
            pre_nms_topk=pre_nms_topk,
            max_detections=max_detections,
            nms_iou=nms_iou,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            effective_score_ratio=score_ratio,
            effective_nms_iou=nms_iou,
        )
        return make_result(boxes, scores, stats, indices)

    scores = logits.sigmoid()
    (
        threshold,
        effective_nms_iou,
        adaptive_regime,
        adaptive_candidate_count,
        effective_score_ratio,
    ) = _threshold_plan(
        scores,
        threshold_mode=threshold_mode,
        score_threshold=score_threshold,
        score_ratio=score_ratio,
        score_quantile=score_quantile,
        min_score_gap=min_score_gap,
        nms_iou=nms_iou,
        adaptive_sparse_score_ratio=adaptive_sparse_score_ratio,
        adaptive_dense_score_ratio=adaptive_dense_score_ratio,
        adaptive_sparse_nms_iou=adaptive_sparse_nms_iou,
        adaptive_dense_nms_iou=adaptive_dense_nms_iou,
        adaptive_dense_candidate_threshold=adaptive_dense_candidate_threshold,
    )
    score_min = scores.min().item()
    score_max = scores.max().item()
    score_mean = scores.mean().item()
    preds_total = int(scores.numel())

    valid = scores >= threshold
    preds_after_threshold = int(valid.sum().item())
    if valid.sum() == 0:
        boxes = pred_boxes.new_zeros((0, 4))
        scores_out = scores.new_zeros((0,))
        indices = original_indices.new_zeros((0,))
        stats = _empty_stats(
            threshold_mode=threshold_mode,
            score_threshold=score_threshold,
            score_ratio=score_ratio,
            score_quantile=score_quantile,
            min_score_gap=min_score_gap,
            effective_threshold=threshold.item(),
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            preds_total=preds_total,
            preds_after_threshold=preds_after_threshold,
            pre_nms_topk=pre_nms_topk,
            max_detections=max_detections,
            nms_iou=nms_iou,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            adaptive_regime=adaptive_regime,
            adaptive_candidate_count=adaptive_candidate_count,
            effective_score_ratio=effective_score_ratio,
            effective_nms_iou=effective_nms_iou,
        )
        return make_result(boxes, scores_out, stats, indices)

    boxes = torch.clamp(pred_boxes[valid], 0, 1)
    scores = scores[valid]
    original_indices = original_indices[valid]

    keep_shape = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    preds_after_shape = int(keep_shape.sum().item())
    if keep_shape.sum() == 0:
        boxes_out = pred_boxes.new_zeros((0, 4))
        scores_out = scores.new_zeros((0,))
        indices = original_indices.new_zeros((0,))
        stats = _empty_stats(
            threshold_mode=threshold_mode,
            score_threshold=score_threshold,
            score_ratio=score_ratio,
            score_quantile=score_quantile,
            min_score_gap=min_score_gap,
            effective_threshold=threshold.item(),
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            preds_total=preds_total,
            preds_after_threshold=preds_after_threshold,
            preds_after_shape=preds_after_shape,
            pre_nms_topk=pre_nms_topk,
            max_detections=max_detections,
            nms_iou=nms_iou,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            adaptive_regime=adaptive_regime,
            adaptive_candidate_count=adaptive_candidate_count,
            effective_score_ratio=effective_score_ratio,
            effective_nms_iou=effective_nms_iou,
        )
        return make_result(boxes_out, scores_out, stats, indices)

    boxes = boxes[keep_shape]
    scores = scores[keep_shape]
    original_indices = original_indices[keep_shape]

    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    areas = widths * heights
    keep_area = areas >= float(min_box_area)
    if max_box_area > 0:
        keep_area = keep_area & (areas <= float(max_box_area))
    preds_after_area = int(keep_area.sum().item())
    if keep_area.sum() == 0:
        boxes_out = pred_boxes.new_zeros((0, 4))
        scores_out = scores.new_zeros((0,))
        indices = original_indices.new_zeros((0,))
        stats = _empty_stats(
            threshold_mode=threshold_mode,
            score_threshold=score_threshold,
            score_ratio=score_ratio,
            score_quantile=score_quantile,
            min_score_gap=min_score_gap,
            effective_threshold=threshold.item(),
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            preds_total=preds_total,
            preds_after_threshold=preds_after_threshold,
            preds_after_shape=preds_after_shape,
            preds_after_area=preds_after_area,
            pre_nms_topk=pre_nms_topk,
            max_detections=max_detections,
            nms_iou=nms_iou,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            adaptive_regime=adaptive_regime,
            adaptive_candidate_count=adaptive_candidate_count,
            effective_score_ratio=effective_score_ratio,
            effective_nms_iou=effective_nms_iou,
        )
        return make_result(boxes_out, scores_out, stats, indices)

    boxes = boxes[keep_area]
    scores = scores[keep_area]
    original_indices = original_indices[keep_area]

    if pre_nms_topk > 0 and scores.numel() > pre_nms_topk:
        topk_scores, topk_idx = torch.topk(scores, k=int(pre_nms_topk), largest=True)
        boxes = boxes[topk_idx]
        scores = topk_scores
        original_indices = original_indices[topk_idx]

    preds_before_nms = int(scores.numel())
    keep = ops.nms(boxes, scores, float(effective_nms_iou))
    if max_detections > 0:
        keep = keep[: int(max_detections)]

    boxes_out = boxes[keep]
    scores_out = scores[keep]
    indices_out = original_indices[keep]
    preds_after_nms = int(keep.numel())
    stats = _empty_stats(
        threshold_mode=threshold_mode,
        score_threshold=score_threshold,
        score_ratio=score_ratio,
        score_quantile=score_quantile,
        min_score_gap=min_score_gap,
        effective_threshold=threshold.item(),
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        preds_total=preds_total,
        preds_after_threshold=preds_after_threshold,
        preds_after_shape=preds_after_shape,
        preds_after_area=preds_after_area,
        preds_before_nms=preds_before_nms,
        preds_after_nms=preds_after_nms,
        preds_final=int(boxes_out.shape[0]),
        pre_nms_topk=pre_nms_topk,
        max_detections=max_detections,
        nms_iou=nms_iou,
        min_box_area=min_box_area,
        max_box_area=max_box_area,
        adaptive_regime=adaptive_regime,
        adaptive_candidate_count=adaptive_candidate_count,
        effective_score_ratio=effective_score_ratio,
        effective_nms_iou=effective_nms_iou,
    )
    return make_result(boxes_out, scores_out, stats, indices_out)
