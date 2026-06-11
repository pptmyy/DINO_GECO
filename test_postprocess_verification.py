import torch

from arg_parser import get_argparser
from infer import count_valid_gt_boxes
from src.utils.postprocess import filter_detections
from src.utils.verification import verify_detections


def test_filter_detections_static_ratio_returns_stats():
    output = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 0.1, 0.1],
                [0.2, 0.2, 0.3, 0.3],
                [0.4, 0.4, 0.5, 0.5],
                [0.6, 0.6, 0.7, 0.7],
            ]
        ),
        "box_v": torch.tensor([5.0, 4.0, 0.0, -5.0]),
    }

    boxes, scores, stats = filter_detections(
        output,
        score_threshold=0.1,
        score_ratio=0.5,
        nms_iou=0.5,
        return_stats=True,
    )

    assert boxes.shape == (3, 4)
    assert scores.shape == (3,)
    assert stats.threshold_mode == "static_ratio"
    assert stats.preds_total == 4
    assert stats.preds_after_threshold == 3
    assert stats.preds_after_nms == 3


def test_filter_detections_can_return_original_indices():
    output = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 0.1, 0.1],
                [0.2, 0.2, 0.3, 0.3],
                [0.4, 0.4, 0.5, 0.5],
                [0.6, 0.6, 0.7, 0.7],
            ]
        ),
        "box_v": torch.tensor([5.0, 4.0, 0.0, -5.0]),
    }

    boxes, _, _, indices = filter_detections(
        output,
        score_threshold=0.1,
        score_ratio=0.5,
        return_stats=True,
        return_indices=True,
    )

    assert boxes.shape == (3, 4)
    assert indices.tolist() == [0, 1, 2]


def test_filter_detections_quantile_mode_keeps_high_score_tail():
    output = {
        "pred_boxes": torch.tensor(
            [
                [0.0, 0.0, 0.1, 0.1],
                [0.2, 0.2, 0.3, 0.3],
                [0.4, 0.4, 0.5, 0.5],
                [0.6, 0.6, 0.7, 0.7],
            ]
        ),
        "box_v": torch.tensor([0.0, 1.0, 2.0, 3.0]),
    }

    boxes, _, stats = filter_detections(
        output,
        score_threshold=0.0,
        threshold_mode="quantile",
        score_quantile=0.75,
        nms_iou=0.5,
        return_stats=True,
    )

    assert boxes.shape[0] == 1
    assert stats.threshold_mode == "quantile"
    assert 0.0 < stats.effective_threshold <= 1.0


def test_filter_detections_regime_adaptive_uses_candidate_count():
    output = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 0.1, 0.1],
                [0.2, 0.2, 0.3, 0.3],
                [0.4, 0.4, 0.5, 0.5],
                [0.6, 0.6, 0.7, 0.7],
                [0.8, 0.8, 0.9, 0.9],
            ]
        ),
        "box_v": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
    }

    boxes, _, stats = filter_detections(
        output,
        score_threshold=0.0,
        score_ratio=0.5,
        threshold_mode="regime_adaptive",
        adaptive_dense_candidate_threshold=2,
        adaptive_dense_score_ratio=0.9,
        adaptive_dense_nms_iou=0.2,
        return_stats=True,
    )

    assert boxes.shape[0] == 3
    assert stats.adaptive_regime == "dense"
    assert stats.adaptive_candidate_count == 5
    assert stats.effective_score_ratio == 0.9
    assert stats.effective_nms_iou == 0.2


def test_verify_detections_exemplar_geometry_filters_mismatched_boxes():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.6, 0.6],
            [0.0, 0.0, 0.1, 0.4],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    exemplar_boxes = torch.tensor([[0.2, 0.2, 0.3, 0.3]])

    verified_boxes, verified_scores, stats = verify_detections(
        boxes,
        scores,
        exemplar_boxes=exemplar_boxes,
        mode="exemplar_geometry",
        threshold=0.75,
        return_stats=True,
    )

    assert verified_boxes.shape == (1, 4)
    assert torch.allclose(verified_scores, torch.tensor([0.9]))
    assert stats.enabled is True
    assert stats.candidate_count == 3
    assert stats.kept_count == 1
    assert stats.filtered_count == 2


def test_verify_detections_feature_similarity_filters_by_cosine():
    boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]])
    scores = torch.tensor([0.9, 0.8])
    candidate_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    exemplar_features = torch.tensor([[1.0, 0.0]])

    verified_boxes, verified_scores, stats = verify_detections(
        boxes,
        scores,
        candidate_features=candidate_features,
        exemplar_features=exemplar_features,
        mode="feature_similarity",
        threshold=0.8,
        return_stats=True,
    )

    assert verified_boxes.shape == (1, 4)
    assert torch.allclose(verified_scores, torch.tensor([0.9]))
    assert stats.enabled is True
    assert stats.kept_count == 1


def test_verify_detections_soft_mode_reweights_without_filtering():
    boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]])
    scores = torch.tensor([0.9, 0.8])
    candidate_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    exemplar_features = torch.tensor([[1.0, 0.0]])

    verified_boxes, verified_scores, stats = verify_detections(
        boxes,
        scores,
        candidate_features=candidate_features,
        exemplar_features=exemplar_features,
        mode="feature_similarity",
        threshold=0.8,
        filter_mode="soft",
        score_gamma=1.0,
        return_stats=True,
    )

    assert verified_boxes.shape == (2, 4)
    assert torch.allclose(verified_scores, torch.tensor([0.9, 0.0]))
    assert stats.filter_mode == "soft"
    assert stats.hard_filter_applied is False
    assert stats.kept_count == 2


def test_parser_exposes_postprocess_and_verification_options():
    args = get_argparser().parse_args([])

    assert args.threshold_mode == "static_ratio"
    assert args.score_quantile == 0.98
    assert args.adaptive_dense_score_ratio == 0.45
    assert args.verification_mode == "none"
    assert args.verification_filter_mode == "hard"
    assert args.verification_topk == 0

    args = get_argparser().parse_args(
        [
            "--threshold-mode",
            "regime_adaptive",
            "--verification-mode",
            "feature_similarity",
            "--verification-filter-mode",
            "soft",
        ]
    )
    assert args.threshold_mode == "regime_adaptive"
    assert args.verification_mode == "feature_similarity"
    assert args.verification_filter_mode == "soft"


def test_infer_gt_count_uses_valid_gt_boxes_not_density_sum():
    gt_bboxes = torch.tensor(
        [
            [10.0, 10.0, 20.0, 20.0],
            [0.0, 0.0, 0.0, 0.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )

    assert count_valid_gt_boxes(gt_bboxes) == 2
