import torch

from arg_parser import get_argparser
from infer import count_valid_gt_boxes
from src.models.scale_query_aggregator import ScaleAwareQueryAggregator
from src.utils.detection_metrics import (
    average_precision_from_records,
    detection_ap_records,
)
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


def test_filter_detections_dense_soft_switches_to_soft_nms_in_dense_regime():
    output = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 0.4, 0.4],
                [0.02, 0.02, 0.42, 0.42],
                [0.04, 0.04, 0.44, 0.44],
            ]
        ),
        "box_v": torch.tensor([5.0, 4.9, 4.8]),
    }

    boxes, _, stats = filter_detections(
        output,
        score_threshold=0.0,
        score_ratio=0.1,
        threshold_mode="regime_adaptive",
        nms_iou=0.3,
        nms_method="dense_soft",
        soft_nms_score_threshold=0.001,
        adaptive_dense_candidate_threshold=1,
        adaptive_dense_nms_iou=0.3,
        return_stats=True,
    )

    assert stats.adaptive_regime == "dense"
    assert stats.effective_nms_method == "soft"
    assert boxes.shape[0] > 1


def test_filter_detections_dense_soft_switches_with_static_threshold():
    output = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 0.4, 0.4],
                [0.02, 0.02, 0.42, 0.42],
                [0.04, 0.04, 0.44, 0.44],
            ]
        ),
        "box_v": torch.tensor([5.0, 4.9, 4.8]),
    }

    boxes, _, stats = filter_detections(
        output,
        score_threshold=0.0,
        score_ratio=0.1,
        threshold_mode="static_ratio",
        nms_iou=0.3,
        nms_method="dense_soft",
        soft_nms_score_threshold=0.001,
        adaptive_dense_candidate_threshold=1,
        adaptive_dense_nms_iou=0.3,
        return_stats=True,
    )

    assert stats.adaptive_regime == "dense"
    assert stats.effective_nms_method == "soft"
    assert boxes.shape[0] > 1


def test_scale_query_aggregator_outputs_stride_4_main_and_stride_2_refinement():
    torch.manual_seed(0)
    aggregator = ScaleAwareQueryAggregator(channels=8, output_stride=4)
    q_out, q_refine = aggregator(
        q3=torch.randn(2, 8, 4, 4),
        q2=torch.randn(2, 8, 8, 8),
        q1=torch.randn(2, 8, 16, 16),
        prototype_embeddings=torch.randn(2, 3, 8),
        hq_prototypes=[torch.randn(2, 3, 8), torch.randn(2, 3, 8)],
    )

    assert q_out.shape == (2, 8, 16, 16)
    assert q_refine.shape == (2, 8, 32, 32)


def test_detection_ap_records_are_score_ordered_and_match_once():
    pred_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.2, 0.2],
            [0.0, 0.0, 0.2, 0.2],
            [0.5, 0.5, 0.7, 0.7],
        ]
    )
    pred_scores = torch.tensor([0.8, 0.9, 0.7])
    target_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2]])

    records = detection_ap_records(pred_boxes, pred_scores, target_boxes, iou_threshold=0.5)
    ap50 = average_precision_from_records(records, target_count=1)

    assert [round(score, 1) for score, _ in records] == [0.9, 0.8, 0.7]
    assert [is_tp for _, is_tp in records] == [1, 0, 0]
    assert ap50 == 1.0


def test_detection_ap_records_falls_back_to_unmatched_gt():
    pred_boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.05, 0.05, 1.05, 1.05],
        ]
    )
    pred_scores = torch.tensor([0.9, 0.8])
    target_boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],
        ]
    )

    records = detection_ap_records(pred_boxes, pred_scores, target_boxes, iou_threshold=0.5)

    assert [is_tp for _, is_tp in records] == [1, 1]
    assert average_precision_from_records(records, target_count=2) == 1.0


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


def test_verify_detections_feature_similarity_filters_padded_exemplar_tokens():
    boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]])
    scores = torch.tensor([0.9, 0.8])
    candidate_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    exemplar_boxes = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0],
            [0.3, 0.3, 0.4, 0.4],
        ]
    )
    exemplar_features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    verified_boxes, verified_scores, stats = verify_detections(
        boxes,
        scores,
        exemplar_boxes=exemplar_boxes,
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


def test_verify_detections_soft_mode_reweights_then_filters_scores():
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

    assert verified_boxes.shape == (1, 4)
    assert torch.allclose(verified_scores, torch.tensor([0.9]))
    assert stats.filter_mode == "soft"
    assert stats.hard_filter_applied is False
    assert stats.soft_filter_applied is True
    assert stats.kept_count == 1


def test_parser_exposes_postprocess_and_verification_options():
    args = get_argparser().parse_args([])

    assert args.threshold_mode == "static_ratio"
    assert args.score_quantile == 0.98
    assert args.pre_nms_topk == 16000
    assert args.max_detections == 16000
    assert args.nms_method == "dense_soft"
    assert args.soft_nms_sigma == 0.5
    assert args.soft_nms_score_threshold == 0.001
    assert args.adaptive_dense_score_ratio == 0.35
    assert args.adaptive_dense_nms_iou == 0.45
    assert args.detection_metric_interval == 1
    assert args.detection_gate_ratio == 0.98
    assert args.use_semantic_anchor is False
    assert args.query_output_stride == 4
    assert args.stride2_refinement is True
    assert args.center_gaussian_head is True
    assert args.center_gaussian_loss_coef == 1.0
    assert args.num_prototypes == 4
    assert args.use_background_token is True
    assert args.decoupled_heads is True
    assert not hasattr(args, "share_scale_shape_embedding")
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
            "--query-output-stride",
            "4",
            "--use-semantic-anchor",
        ]
    )
    assert args.threshold_mode == "regime_adaptive"
    assert args.verification_mode == "feature_similarity"
    assert args.verification_filter_mode == "soft"
    assert args.query_output_stride == 4
    assert args.use_semantic_anchor is True


def test_infer_gt_count_uses_valid_gt_boxes_not_density_sum():
    gt_bboxes = torch.tensor(
        [
            [10.0, 10.0, 20.0, 20.0],
            [0.0, 0.0, 0.0, 0.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )

    assert count_valid_gt_boxes(gt_bboxes) == 2
