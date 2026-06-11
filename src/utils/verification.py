from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Union

import torch


@dataclass
class VerificationStats:
    mode: str
    enabled: bool
    filter_mode: str
    threshold: float
    score_gamma: float
    hard_filter_applied: bool
    candidate_count: int
    kept_count: int
    filtered_count: int
    verification_score_min: float
    verification_score_max: float
    verification_score_mean: float

    def to_dict(self) -> Dict[str, Union[str, bool, float, int]]:
        return asdict(self)


def _stats(
    *,
    mode: str,
    enabled: bool,
    filter_mode: str,
    threshold: float,
    score_gamma: float,
    hard_filter_applied: bool,
    candidate_count: int,
    kept_count: int,
    verification_scores: Optional[torch.Tensor] = None,
) -> VerificationStats:
    if verification_scores is None or verification_scores.numel() == 0:
        score_min = 0.0
        score_max = 0.0
        score_mean = 0.0
    else:
        score_min = float(verification_scores.min().item())
        score_max = float(verification_scores.max().item())
        score_mean = float(verification_scores.mean().item())

    return VerificationStats(
        mode=mode,
        enabled=bool(enabled),
        filter_mode=str(filter_mode),
        threshold=float(threshold),
        score_gamma=float(score_gamma),
        hard_filter_applied=bool(hard_filter_applied),
        candidate_count=int(candidate_count),
        kept_count=int(kept_count),
        filtered_count=int(candidate_count - kept_count),
        verification_score_min=score_min,
        verification_score_max=score_max,
        verification_score_mean=score_mean,
    )


def _valid_boxes(boxes: torch.Tensor) -> torch.Tensor:
    if boxes is None or boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = boxes.reshape(-1, 4).float()
    valid = torch.logical_not((boxes == 0).all(dim=1))
    boxes = boxes[valid]
    if boxes.numel() == 0:
        return boxes
    boxes = torch.clamp(boxes, 0, 1)
    keep_shape = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep_shape]


def _valid_box_mask(boxes: torch.Tensor) -> torch.Tensor:
    if boxes is None or boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    boxes = boxes.reshape(-1, 4).float()
    non_empty = torch.logical_not((boxes == 0).all(dim=1))
    boxes = torch.clamp(boxes, 0, 1)
    keep_shape = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return non_empty & keep_shape


def _geometry_scores(boxes: torch.Tensor, exemplar_boxes: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(boxes.dtype).eps
    box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(eps)
    exemplar_wh = (exemplar_boxes[:, 2:] - exemplar_boxes[:, :2]).clamp_min(eps)

    box_area = box_wh[:, 0] * box_wh[:, 1]
    exemplar_area = exemplar_wh[:, 0] * exemplar_wh[:, 1]
    box_aspect = box_wh[:, 0] / box_wh[:, 1]
    exemplar_aspect = exemplar_wh[:, 0] / exemplar_wh[:, 1]

    reference_area = torch.median(exemplar_area).clamp_min(eps)
    reference_aspect = torch.median(exemplar_aspect).clamp_min(eps)

    area_similarity = torch.exp(-torch.abs(torch.log((box_area / reference_area).clamp_min(eps))))
    aspect_similarity = torch.exp(
        -torch.abs(torch.log((box_aspect / reference_aspect).clamp_min(eps)))
    )
    return 0.7 * area_similarity + 0.3 * aspect_similarity


def _feature_scores(candidate_features: torch.Tensor, exemplar_features: torch.Tensor) -> torch.Tensor:
    candidate_features = candidate_features.float()
    exemplar_features = exemplar_features.float()
    candidate_features = torch.nn.functional.normalize(candidate_features, dim=-1)
    exemplar_features = torch.nn.functional.normalize(exemplar_features, dim=-1)
    similarity = candidate_features @ exemplar_features.transpose(-1, -2)
    return similarity.max(dim=-1).values


def verify_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    exemplar_boxes: Optional[torch.Tensor] = None,
    candidate_features: Optional[torch.Tensor] = None,
    exemplar_features: Optional[torch.Tensor] = None,
    mode: str = "none",
    threshold: float = 0.0,
    topk: int = 0,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 0.0,
    filter_mode: str = "hard",
    score_gamma: float = 0.0,
    hard_candidate_limit: int = 0,
    return_stats: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, VerificationStats]]:
    """Optional detect-and-verify stage after NMS.

    The built-in verifiers are intentionally lightweight: exemplar_geometry checks
    size/aspect consistency, while feature_similarity filters by cosine similarity
    between candidate features and exemplar ROI features.
    """
    candidate_count = int(boxes.shape[0])
    if mode == "none" or candidate_count == 0:
        stats = _stats(
            mode=mode,
            enabled=False,
            filter_mode=filter_mode,
            threshold=threshold,
            score_gamma=score_gamma,
            hard_filter_applied=False,
            candidate_count=candidate_count,
            kept_count=candidate_count,
        )
        if return_stats:
            return boxes, scores, stats
        return boxes, scores

    if mode not in {"exemplar_geometry", "feature_similarity"}:
        raise ValueError(f"Unsupported verification mode: {mode!r}")
    if filter_mode not in {"hard", "soft", "sparse_hard"}:
        raise ValueError(f"Unsupported verification filter_mode: {filter_mode!r}")

    if mode == "feature_similarity":
        if candidate_features is None or exemplar_features is None:
            stats = _stats(
                mode=mode,
                enabled=False,
                filter_mode=filter_mode,
                threshold=threshold,
                score_gamma=score_gamma,
                hard_filter_applied=False,
                candidate_count=candidate_count,
                kept_count=candidate_count,
            )
            if return_stats:
                return boxes, scores, stats
            return boxes, scores

        candidate_features = candidate_features.reshape(candidate_count, -1).to(
            device=boxes.device,
            dtype=boxes.dtype,
        )
        exemplar_features = exemplar_features.reshape(-1, candidate_features.shape[-1]).to(
            device=boxes.device,
            dtype=boxes.dtype,
        )
        if exemplar_boxes is not None and exemplar_boxes.numel() > 0:
            valid_mask = _valid_box_mask(exemplar_boxes.to(device=boxes.device))
            if valid_mask.numel() == exemplar_features.shape[0]:
                exemplar_features = exemplar_features[valid_mask]
        if exemplar_features.numel() == 0:
            stats = _stats(
                mode=mode,
                enabled=False,
                filter_mode=filter_mode,
                threshold=threshold,
                score_gamma=score_gamma,
                hard_filter_applied=False,
                candidate_count=candidate_count,
                kept_count=candidate_count,
            )
            if return_stats:
                return boxes, scores, stats
            return boxes, scores
        verification_scores = _feature_scores(candidate_features, exemplar_features)
        exemplar_boxes_valid = (
            _valid_boxes(exemplar_boxes.to(device=boxes.device, dtype=boxes.dtype))
            if exemplar_boxes is not None
            else None
        )
    elif exemplar_boxes is None:
        stats = _stats(
            mode=mode,
            enabled=False,
            filter_mode=filter_mode,
            threshold=threshold,
            score_gamma=score_gamma,
            hard_filter_applied=False,
            candidate_count=candidate_count,
            kept_count=candidate_count,
        )
        if return_stats:
            return boxes, scores, stats
        return boxes, scores

    else:
        exemplar_boxes_valid = _valid_boxes(exemplar_boxes.to(device=boxes.device, dtype=boxes.dtype))
        if exemplar_boxes_valid.numel() == 0:
            stats = _stats(
                mode=mode,
                enabled=False,
                filter_mode=filter_mode,
                threshold=threshold,
                score_gamma=score_gamma,
                hard_filter_applied=False,
                candidate_count=candidate_count,
                kept_count=candidate_count,
            )
            if return_stats:
                return boxes, scores, stats
            return boxes, scores
        verification_scores = _geometry_scores(boxes, exemplar_boxes_valid)

    if score_gamma > 0:
        score_weights = verification_scores.clamp(min=0.0, max=1.0).pow(float(score_gamma))
        scores = scores * score_weights

    keep = torch.ones_like(verification_scores, dtype=torch.bool)
    hard_filter_applied = False
    if threshold > 0 and filter_mode == "hard":
        hard_filter_applied = True
    elif threshold > 0 and filter_mode == "sparse_hard":
        hard_filter_applied = int(hard_candidate_limit) > 0 and candidate_count <= int(
            hard_candidate_limit
        )

    if hard_filter_applied:
        keep = keep & (verification_scores >= float(threshold))

    if (min_area_ratio > 0 or max_area_ratio > 0) and exemplar_boxes_valid is not None:
        eps = torch.finfo(boxes.dtype).eps
        box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(eps)
        exemplar_wh = (exemplar_boxes_valid[:, 2:] - exemplar_boxes_valid[:, :2]).clamp_min(eps)
        box_area = box_wh[:, 0] * box_wh[:, 1]
        reference_area = torch.median(exemplar_wh[:, 0] * exemplar_wh[:, 1]).clamp_min(eps)
        area_ratio = box_area / reference_area
        if min_area_ratio > 0:
            keep = keep & (area_ratio >= float(min_area_ratio))
        if max_area_ratio > 0:
            keep = keep & (area_ratio <= float(max_area_ratio))

    keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
    if topk > 0 and keep_idx.numel() > int(topk):
        ranking_scores = scores[keep_idx]
        _, order = torch.topk(ranking_scores, k=int(topk), largest=True)
        keep_idx = keep_idx[order]

    boxes_out = boxes[keep_idx]
    scores_out = scores[keep_idx]
    stats = _stats(
        mode=mode,
        enabled=True,
        filter_mode=filter_mode,
        threshold=threshold,
        score_gamma=score_gamma,
        hard_filter_applied=hard_filter_applied,
        candidate_count=candidate_count,
        kept_count=int(boxes_out.shape[0]),
        verification_scores=verification_scores,
    )
    if return_stats:
        return boxes_out, scores_out, stats
    return boxes_out, scores_out
