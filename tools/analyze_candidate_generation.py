#!/usr/bin/env python
"""Compare candidate generation statistics from eval per-image JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


COUNT_BINS = (
    ("7-20", 0, 20),
    ("20-50", 20, 50),
    ("50-100", 50, 100),
    ("100-300", 100, 300),
    (">300", 300, None),
)


def count_bin(gt_count: int) -> str:
    for name, _lower, upper in COUNT_BINS:
        if upper is None or gt_count <= upper:
            return name
    return COUNT_BINS[-1][0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {"overall": records}
    for name, _lo, _hi in COUNT_BINS:
        groups[name] = [r for r in records if count_bin(int(r["gt_count"])) == name]

    out: dict[str, dict[str, float]] = {}
    for name, rows in groups.items():
        n = len(rows)
        denom = max(n, 1)

        gt = [safe_float(r.get("gt_count")) for r in rows]
        pred = [safe_float(r.get("pred_count")) for r in rows]
        err = [p - g for p, g in zip(pred, gt)]
        pp = [r.get("postprocess") or {} for r in rows]
        scores = [safe_float(s) for r in rows for s in (r.get("scores") or [])]

        preds_total = [safe_float(p.get("preds_total")) for p in pp]
        after_threshold = [safe_float(p.get("preds_after_threshold")) for p in pp]
        before_nms = [safe_float(p.get("preds_before_nms")) for p in pp]
        after_nms = [safe_float(p.get("preds_after_nms")) for p in pp]
        threshold_keep = [
            a / t if t > 0 else 0.0 for a, t in zip(after_threshold, preds_total)
        ]
        nms_keep = [a / b if b > 0 else 0.0 for a, b in zip(after_nms, before_nms)]

        out[name] = {
            "n": float(n),
            "gt_avg": sum(gt) / denom,
            "pred_avg": sum(pred) / denom,
            "mae": sum(abs(e) for e in err) / denom,
            "rmse": math.sqrt(sum(e * e for e in err) / denom) if n else 0.0,
            "signed_error": sum(err) / denom,
            "over_count": float(sum(1 for e in err if e > 0)),
            "under_count": float(sum(1 for e in err if e < 0)),
            "preds_total_avg": sum(preds_total) / denom,
            "after_threshold_avg": sum(after_threshold) / denom,
            "before_nms_avg": sum(before_nms) / denom,
            "after_nms_avg": sum(after_nms) / denom,
            "threshold_keep_ratio_avg": sum(threshold_keep) / denom,
            "nms_keep_ratio_avg": sum(nms_keep) / denom,
            "pre_nms_per_gt": (sum(before_nms) / max(sum(gt), 1.0)) if n else 0.0,
            "post_nms_per_gt": (sum(after_nms) / max(sum(gt), 1.0)) if n else 0.0,
            "score_min_avg": sum(safe_float(p.get("score_min")) for p in pp) / denom,
            "score_mean_avg": sum(safe_float(p.get("score_mean")) for p in pp) / denom,
            "score_max_avg": sum(safe_float(p.get("score_max")) for p in pp) / denom,
            "effective_threshold_avg": sum(
                safe_float(p.get("effective_threshold")) for p in pp
            )
            / denom,
            "final_score_p25": quantile(scores, 0.25),
            "final_score_p50": quantile(scores, 0.50),
            "final_score_p75": quantile(scores, 0.75),
            "final_score_p90": quantile(scores, 0.90),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(label: str, summary: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for bin_name in ["overall", *(name for name, _lo, _hi in COUNT_BINS)]:
        row: dict[str, Any] = {"run": label, "count_bin": bin_name}
        row.update(summary[bin_name])
        rows.append(row)
    return rows


def delta_rows(
    baseline: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    keys = [
        "gt_avg",
        "pred_avg",
        "mae",
        "rmse",
        "signed_error",
        "preds_total_avg",
        "after_threshold_avg",
        "before_nms_avg",
        "after_nms_avg",
        "threshold_keep_ratio_avg",
        "nms_keep_ratio_avg",
        "pre_nms_per_gt",
        "post_nms_per_gt",
        "score_mean_avg",
        "score_max_avg",
        "effective_threshold_avg",
        "final_score_p50",
        "final_score_p90",
    ]
    rows = []
    for bin_name in ["overall", *(name for name, _lo, _hi in COUNT_BINS)]:
        row: dict[str, Any] = {"count_bin": bin_name, "n": candidate[bin_name]["n"]}
        for key in keys:
            base = baseline[bin_name].get(key, 0.0)
            cand = candidate[bin_name].get(key, 0.0)
            row[f"baseline_{key}"] = base
            row[f"candidate_{key}"] = cand
            row[f"delta_{key}"] = cand - base
        rows.append(row)
    return rows


def focus_rows(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    focus_images: list[str],
) -> list[dict[str, Any]]:
    baseline = {str(r.get("image")): r for r in baseline_records}
    candidate = {str(r.get("image")): r for r in candidate_records}
    rows: list[dict[str, Any]] = []
    for image in focus_images:
        for label, record in (("baseline", baseline.get(image)), ("candidate", candidate.get(image))):
            if record is None:
                rows.append({"image": image, "run": label, "missing": True})
                continue
            pp = record.get("postprocess") or {}
            scores = [safe_float(s) for s in (record.get("scores") or [])]
            row = {
                "image": image,
                "run": label,
                "count_bin": count_bin(int(record["gt_count"])),
                "gt_count": record.get("gt_count"),
                "pred_count": record.get("pred_count"),
                "error": record.get("error"),
                "abs_error": record.get("abs_error"),
                "density_sum_debug": record.get("density_sum_debug"),
                "preds_total": pp.get("preds_total"),
                "score_min": pp.get("score_min"),
                "score_mean": pp.get("score_mean"),
                "score_max": pp.get("score_max"),
                "effective_threshold": pp.get("effective_threshold"),
                "threshold_over_score_max": safe_float(pp.get("effective_threshold"))
                / max(safe_float(pp.get("score_max")), 1e-12),
                "preds_after_threshold": pp.get("preds_after_threshold"),
                "preds_before_nms": pp.get("preds_before_nms"),
                "preds_after_nms": pp.get("preds_after_nms"),
                "threshold_keep_ratio": safe_float(pp.get("preds_after_threshold"))
                / max(safe_float(pp.get("preds_total")), 1e-12),
                "nms_keep_ratio": safe_float(pp.get("preds_after_nms"))
                / max(safe_float(pp.get("preds_before_nms")), 1e-12),
                "adaptive_regime": pp.get("adaptive_regime"),
                "effective_nms_method": pp.get("effective_nms_method"),
                "effective_nms_iou": pp.get("effective_nms_iou"),
                "final_score_p50": quantile(scores, 0.50),
                "final_score_p90": quantile(scores, 0.90),
                "top5_scores": ";".join(f"{s:.4f}" for s in sorted(scores, reverse=True)[:5]),
            }
            rows.append(row)
    return rows


def write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    focus: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    delta_by_bin = {row["count_bin"]: row for row in deltas}
    focus_candidate = [r for r in focus if r.get("run") == "candidate"]
    focus_baseline = [r for r in focus if r.get("run") == "baseline"]
    base_by_img = {r["image"]: r for r in focus_baseline}

    lines = [
        "# Candidate Generation Diagnostics",
        "",
        f"Baseline: `{baseline_label}`",
        f"Candidate: `{candidate_label}`",
        "",
        "## Count-Bin Delta",
        "",
        "| bin | delta MAE | delta RMSE | delta signed | delta pre-NMS/GT | delta post-NMS/GT | delta score mean | delta score max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bin_name in ["overall", *(name for name, _lo, _hi in COUNT_BINS)]:
        row = delta_by_bin[bin_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    bin_name,
                    fmt(row["delta_mae"]),
                    fmt(row["delta_rmse"]),
                    fmt(row["delta_signed_error"]),
                    fmt(row["delta_pre_nms_per_gt"]),
                    fmt(row["delta_post_nms_per_gt"]),
                    fmt(row["delta_score_mean_avg"]),
                    fmt(row["delta_score_max_avg"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Focus Images",
            "",
            "| image | gt | baseline pred/error/pre/post | candidate pred/error/pre/post | candidate threshold | candidate score mean/max |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for cand in focus_candidate:
        image = cand["image"]
        base = base_by_img.get(image, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    image,
                    fmt(cand.get("gt_count")),
                    f"{fmt(base.get('pred_count'))}/{fmt(base.get('error'))}/{fmt(base.get('preds_before_nms'))}/{fmt(base.get('preds_after_nms'))}",
                    f"{fmt(cand.get('pred_count'))}/{fmt(cand.get('error'))}/{fmt(cand.get('preds_before_nms'))}/{fmt(cand.get('preds_after_nms'))}",
                    fmt(cand.get("effective_threshold")),
                    f"{fmt(cand.get('score_mean'))}/{fmt(cand.get('score_max'))}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Existing per-image files contain raw score min/max/mean, not full raw score arrays. Final post-NMS score quantiles are reported in CSV outputs.",
            "- If candidate high-count pre-NMS/GT is below baseline while NMS keep ratio is near 1.0, the bottleneck is candidate generation before NMS, not NMS suppression.",
            "- If low-count bins improve while >300 worsens, the model/threshold path is trading low-count false positives for high-count recall.",
            "",
            "Artifacts:",
            "- `candidate_generation_summary.csv`",
            "- `candidate_generation_delta.csv`",
            "- `focus_image_diagnostics.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--focus-images",
        nargs="+",
        default=["935.jpg", "7656.jpg", "1956.jpg", "949.jpg", "4106.jpg"],
    )
    args = parser.parse_args()

    baseline_records = read_jsonl(args.baseline)
    candidate_records = read_jsonl(args.candidate)

    baseline_summary = summarize(baseline_records)
    candidate_summary = summarize(candidate_records)

    summary_rows = flatten_summary(args.baseline_label, baseline_summary)
    summary_rows.extend(flatten_summary(args.candidate_label, candidate_summary))
    deltas = delta_rows(baseline_summary, candidate_summary)
    focus = focus_rows(baseline_records, candidate_records, args.focus_images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_generation_summary.csv", summary_rows)
    write_csv(args.output_dir / "candidate_generation_delta.csv", deltas)
    write_csv(args.output_dir / "focus_image_diagnostics.csv", focus)
    write_report(
        args.output_dir / "diagnostic_report.md",
        summary_rows=summary_rows,
        deltas=deltas,
        focus=focus,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )

    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
