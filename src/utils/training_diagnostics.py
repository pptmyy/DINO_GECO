DIAGNOSTIC_SUM_KEYS = (
    "targets",
    "preds_total",
    "preds_before_nms",
    "preds_after_nms",
    "preds_after_verify",
    "preds_final",
    "effective_threshold_sum",
    "verification_filtered",
    "postprocess_samples",
    "adaptive_dense_samples",
    "adaptive_sparse_samples",
    "adaptive_none_samples",
    "nms_soft_samples",
    "nms_hard_samples",
    "adaptive_candidate_count_sum",
    "effective_score_ratio_sum",
    "effective_nms_iou_sum",
    "score_mean_sum",
    "verification_enabled_samples",
    "verification_score_mean_sum",
)


def empty_loss_stats():
    return {
        "loss_total": 0.0,
        "loss_giou": 0.0,
        "loss_bbox": 0.0,
        "loss_ce": 0.0,
        "loss_center_gaussian": 0.0,
        "aux_loss_total": 0.0,
        "aux_loss_giou": 0.0,
        "aux_loss_bbox": 0.0,
        "aux_loss_ce": 0.0,
        "mask_sum": 0.0,
        "aux_mask_sum": 0.0,
        "aux_positive_sum": 0.0,
        "aux_active_samples": 0,
        "aux_weight_sum": 0.0,
        "center_gaussian_positive_sum": 0.0,
        "center_gaussian_weight_sum": 0.0,
        "positive_sum": 0.0,
        "targets": 0,
        "preds_total": 0,
        "preds_before_nms": 0,
        "preds_after_nms": 0,
        "preds_after_verify": 0,
        "preds_final": 0,
        "effective_threshold_sum": 0.0,
        "effective_threshold_mean": 0.0,
        "verification_filtered": 0,
        "postprocess_samples": 0,
        "adaptive_dense_samples": 0,
        "adaptive_sparse_samples": 0,
        "adaptive_none_samples": 0,
        "nms_soft_samples": 0,
        "nms_hard_samples": 0,
        "adaptive_candidate_count_sum": 0.0,
        "adaptive_candidate_count_mean": 0.0,
        "effective_score_ratio_sum": 0.0,
        "effective_score_ratio_mean": 0.0,
        "effective_nms_iou_sum": 0.0,
        "effective_nms_iou_mean": 0.0,
        "score_mean_sum": 0.0,
        "score_mean": 0.0,
        "verification_enabled_samples": 0,
        "verification_score_mean_sum": 0.0,
        "verification_score_mean": 0.0,
        "det_tp_iou50": 0,
        "det_fp_iou50": 0,
        "det_fn_iou50": 0,
        "det_gt_iou50": 0,
        "det_ap_records_iou50": [],
    }


def add_loss_stats(stats, losses, *, prefix="", scale=1.0):
    for key in ("loss_total", "loss_giou", "loss_bbox", "loss_ce"):
        if key in losses:
            stats[f"{prefix}{key}"] += (losses[key] * scale).detach().item()
    if "loss_center_gaussian" in losses:
        stats[f"{prefix}loss_center_gaussian"] += (
            losses["loss_center_gaussian"] * scale
        ).detach().item()


def average_loss_stats(stats, batch_size):
    if batch_size <= 0:
        return stats
    for key in (
        "loss_total",
        "loss_giou",
        "loss_bbox",
        "loss_ce",
        "loss_center_gaussian",
        "aux_loss_total",
        "aux_loss_giou",
        "aux_loss_bbox",
        "aux_loss_ce",
    ):
        stats[key] /= batch_size

    postprocess_samples = max(int(stats["postprocess_samples"]), 1)
    verification_samples = max(int(stats["verification_enabled_samples"]), 1)
    stats["effective_threshold_mean"] = stats["effective_threshold_sum"] / postprocess_samples
    stats["adaptive_candidate_count_mean"] = (
        stats["adaptive_candidate_count_sum"] / postprocess_samples
    )
    stats["effective_score_ratio_mean"] = (
        stats["effective_score_ratio_sum"] / postprocess_samples
    )
    stats["effective_nms_iou_mean"] = stats["effective_nms_iou_sum"] / postprocess_samples
    stats["score_mean"] = stats["score_mean_sum"] / postprocess_samples
    stats["verification_score_mean"] = (
        stats["verification_score_mean_sum"] / verification_samples
    )
    return stats


def add_postprocess_diagnostics(stats, postprocess_stats, verification_stats, final_count):
    stats["preds_total"] += postprocess_stats.preds_total
    stats["preds_before_nms"] += postprocess_stats.preds_before_nms
    stats["preds_after_nms"] += postprocess_stats.preds_after_nms
    stats["preds_after_verify"] += verification_stats.kept_count
    stats["preds_final"] += int(final_count)
    stats["effective_threshold_sum"] += postprocess_stats.effective_threshold
    stats["verification_filtered"] += verification_stats.filtered_count
    stats["postprocess_samples"] += 1
    stats["adaptive_candidate_count_sum"] += postprocess_stats.adaptive_candidate_count
    stats["effective_score_ratio_sum"] += postprocess_stats.effective_score_ratio
    stats["effective_nms_iou_sum"] += postprocess_stats.effective_nms_iou
    stats["score_mean_sum"] += postprocess_stats.score_mean

    if postprocess_stats.adaptive_regime == "dense":
        stats["adaptive_dense_samples"] += 1
    elif postprocess_stats.adaptive_regime == "sparse":
        stats["adaptive_sparse_samples"] += 1
    else:
        stats["adaptive_none_samples"] += 1

    if postprocess_stats.effective_nms_method == "soft":
        stats["nms_soft_samples"] += 1
    elif postprocess_stats.effective_nms_method == "hard":
        stats["nms_hard_samples"] += 1

    if verification_stats.enabled:
        stats["verification_enabled_samples"] += 1
        stats["verification_score_mean_sum"] += verification_stats.verification_score_mean


def merge_diagnostic_stats(total, update):
    for key in DIAGNOSTIC_SUM_KEYS:
        total[key] += update.get(key, 0)
    return total


def diagnostic_metrics(stats, prefix):
    samples = max(int(stats.get("postprocess_samples", 0)), 1)
    before_nms = max(int(stats.get("preds_before_nms", 0)), 1)
    after_nms = max(int(stats.get("preds_after_nms", 0)), 1)
    targets = int(stats.get("targets", 0))
    preds_final = int(stats.get("preds_final", 0))
    return {
        f"{prefix}_targets": targets,
        f"{prefix}_preds_total": int(stats.get("preds_total", 0)),
        f"{prefix}_preds_before_nms": int(stats.get("preds_before_nms", 0)),
        f"{prefix}_preds_after_nms": int(stats.get("preds_after_nms", 0)),
        f"{prefix}_preds_final": preds_final,
        f"{prefix}_pred_error": preds_final - targets,
        f"{prefix}_preds_per_gt": preds_final / max(targets, 1),
        f"{prefix}_nms_keep_ratio": stats.get("preds_after_nms", 0) / before_nms,
        f"{prefix}_verification_keep_ratio": stats.get("preds_after_verify", 0)
        / after_nms,
        f"{prefix}_effective_threshold_mean": stats.get("effective_threshold_sum", 0.0)
        / samples,
        f"{prefix}_effective_score_ratio_mean": stats.get(
            "effective_score_ratio_sum",
            0.0,
        )
        / samples,
        f"{prefix}_effective_nms_iou_mean": stats.get("effective_nms_iou_sum", 0.0)
        / samples,
        f"{prefix}_score_mean": stats.get("score_mean_sum", 0.0) / samples,
        f"{prefix}_adaptive_candidate_count_mean": stats.get(
            "adaptive_candidate_count_sum",
            0.0,
        )
        / samples,
        f"{prefix}_dense_fraction": stats.get("adaptive_dense_samples", 0) / samples,
        f"{prefix}_sparse_fraction": stats.get("adaptive_sparse_samples", 0) / samples,
        f"{prefix}_soft_nms_fraction": stats.get("nms_soft_samples", 0) / samples,
        f"{prefix}_hard_nms_fraction": stats.get("nms_hard_samples", 0) / samples,
        f"{prefix}_verification_filtered": int(stats.get("verification_filtered", 0)),
        f"{prefix}_verification_enabled_samples": int(
            stats.get("verification_enabled_samples", 0)
        ),
        f"{prefix}_verification_score_mean": stats.get(
            "verification_score_mean_sum",
            0.0,
        )
        / max(int(stats.get("verification_enabled_samples", 0)), 1),
    }


def format_optional_metric(value, precision=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"
