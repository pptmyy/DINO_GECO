import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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


@dataclass
class SummaryAccumulator:
    n: int = 0
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0
    signed_error_sum: float = 0.0
    gt_sum: float = 0.0
    pred_sum: float = 0.0
    over_count: int = 0
    under_count: int = 0
    equal_count: int = 0
    pre_nms_sum: float = 0.0
    post_nms_sum: float = 0.0
    post_verify_sum: float = 0.0
    threshold_sum: float = 0.0
    density_abs_gap_sum: float = 0.0

    def update(
        self,
        *,
        gt_count: int,
        pred_count: int,
        postprocess: Optional[Dict[str, object]] = None,
        verification: Optional[Dict[str, object]] = None,
        density_sum_debug: Optional[float] = None,
    ) -> None:
        err = float(pred_count) - float(gt_count)
        self.n += 1
        self.abs_error_sum += abs(err)
        self.sq_error_sum += err * err
        self.signed_error_sum += err
        self.gt_sum += float(gt_count)
        self.pred_sum += float(pred_count)
        if err > 0:
            self.over_count += 1
        elif err < 0:
            self.under_count += 1
        else:
            self.equal_count += 1

        postprocess = postprocess or {}
        verification = verification or {}
        self.pre_nms_sum += float(postprocess.get("preds_before_nms", 0.0) or 0.0)
        self.post_nms_sum += float(postprocess.get("preds_after_nms", 0.0) or 0.0)
        self.post_verify_sum += float(verification.get("kept_count", pred_count) or 0.0)
        self.threshold_sum += float(postprocess.get("effective_threshold", 0.0) or 0.0)
        if density_sum_debug is not None:
            self.density_abs_gap_sum += abs(float(density_sum_debug) - float(gt_count))

    def to_dict(self, prefix: str = "") -> Dict[str, object]:
        denom = max(self.n, 1)
        return {
            f"{prefix}n": self.n,
            f"{prefix}gt_avg": self.gt_sum / denom,
            f"{prefix}pred_avg": self.pred_sum / denom,
            f"{prefix}mae": self.abs_error_sum / denom,
            f"{prefix}rmse": (self.sq_error_sum / denom) ** 0.5,
            f"{prefix}signed_error": self.signed_error_sum / denom,
            f"{prefix}over_count": self.over_count,
            f"{prefix}under_count": self.under_count,
            f"{prefix}equal_count": self.equal_count,
            f"{prefix}avg_pre_nms": self.pre_nms_sum / denom,
            f"{prefix}avg_post_nms": self.post_nms_sum / denom,
            f"{prefix}avg_post_verify": self.post_verify_sum / denom,
            f"{prefix}avg_effective_threshold": self.threshold_sum / denom,
            f"{prefix}density_abs_gap": self.density_abs_gap_sum / denom,
        }


@dataclass
class SplitSummary:
    overall: SummaryAccumulator = field(default_factory=SummaryAccumulator)
    bins: Dict[str, SummaryAccumulator] = field(
        default_factory=lambda: {name: SummaryAccumulator() for name, _lo, _hi in COUNT_BINS}
    )

    def update(
        self,
        *,
        gt_count: int,
        pred_count: int,
        postprocess: Optional[Dict[str, object]] = None,
        verification: Optional[Dict[str, object]] = None,
        density_sum_debug: Optional[float] = None,
    ) -> None:
        self.overall.update(
            gt_count=gt_count,
            pred_count=pred_count,
            postprocess=postprocess,
            verification=verification,
            density_sum_debug=density_sum_debug,
        )
        self.bins[count_bin(gt_count)].update(
            gt_count=gt_count,
            pred_count=pred_count,
            postprocess=postprocess,
            verification=verification,
            density_sum_debug=density_sum_debug,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "count_bins": {name: acc.to_dict() for name, acc in self.bins.items()},
        }


def json_default(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=json_default)


def write_count_bin_csv(path: Path, summary: SplitSummary) -> None:
    rows = []
    for name, acc in summary.bins.items():
        row = {"count_bin": name}
        row.update(acc.to_dict())
        rows.append(row)
    write_csv(path, rows)


def write_top_errors_csv(path: Path, records: Iterable[Dict[str, object]], top_k: int) -> None:
    rows: List[Dict[str, object]] = []
    for record in records:
        gt_count = int(record["gt_count"])
        pred_count = int(record["pred_count"])
        rows.append(
            {
                "image": record.get("image"),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "error": pred_count - gt_count,
                "abs_error": abs(pred_count - gt_count),
                "density_sum_debug": record.get("density_sum_debug"),
                "preds_before_nms": (record.get("postprocess") or {}).get("preds_before_nms"),
                "preds_after_nms": (record.get("postprocess") or {}).get("preds_after_nms"),
                "effective_threshold": (record.get("postprocess") or {}).get("effective_threshold"),
            }
        )
    rows.sort(key=lambda row: float(row["abs_error"]), reverse=True)
    write_csv(path, rows[:top_k])


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
