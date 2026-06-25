from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.models.backbones.dinov3_adapter import DINOv3Adapter
from src.utils.box_ops import boxes_with_scores


VALID_SCALES = ("c1", "c2", "c3")


def parse_scale_list(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        scales = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        scales = tuple(str(item).strip() for item in value if str(item).strip())
    invalid = [scale for scale in scales if scale not in VALID_SCALES]
    if invalid:
        raise ValueError(f"Unsupported scales: {invalid}. Valid scales: {VALID_SCALES}")
    if not scales:
        raise ValueError("At least one training scale is required")
    return scales


def parse_scale_float_map(value: str, scales: Sequence[str], default: float) -> Dict[str, float]:
    result = {scale: float(default) for scale in scales}
    if not value:
        return result
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw_number = item.split(":", maxsplit=1)
        key = key.strip()
        if key not in scales:
            raise ValueError(f"Unknown scale {key!r}; expected one of {tuple(scales)}")
        result[key] = float(raw_number)
    return result


def parse_scale_int_map(value: str, scales: Sequence[str], default: int) -> Dict[str, int]:
    result = {scale: int(default) for scale in scales}
    if not value:
        return result
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw_number = item.split(":", maxsplit=1)
        key = key.strip()
        if key not in scales:
            raise ValueError(f"Unknown scale {key!r}; expected one of {tuple(scales)}")
        result[key] = int(raw_number)
    return result


class AdapterDetectionHead(nn.Module):
    """Small class-agnostic dense detection head used only for COCO warm-up."""

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels or channels)
        self.tower = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32 if hidden_channels % 32 == 0 else 1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32 if hidden_channels % 32 == 0 else 1, hidden_channels),
            nn.GELU(),
        )
        self.objectness = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        self.bbox = nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.tower(feature)
        return self.objectness(feature), self.bbox(feature).sigmoid()


class CocoAdapterPretrainModel(nn.Module):
    """DINOv3Adapter plus temporary class-agnostic heads for COCO pretraining."""

    def __init__(
        self,
        *,
        image_size: int,
        dinov3_model_size: str,
        dinov3_pretrained_weights: str | None,
        train_scales: Sequence[str] = VALID_SCALES,
        max_candidates_per_scale: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.train_scales = parse_scale_list(train_scales)
        self.max_candidates_per_scale = {
            scale: int((max_candidates_per_scale or {}).get(scale, 4096))
            for scale in self.train_scales
        }
        self.backbone = DINOv3Adapter(
            image_size=image_size,
            model_size=dinov3_model_size,
            patch_size=16,
            out_feature_indexes=[2, 5, 8, 11],
            freeze_encoder=True,
            pretrained_weights=dinov3_pretrained_weights,
        )
        channels = self.backbone.dinov3.embed_dim
        self.heads = nn.ModuleDict(
            {scale: AdapterDetectionHead(channels) for scale in self.train_scales}
        )

    @staticmethod
    def _scale_features(features: Mapping[str, object]) -> Dict[str, Tensor]:
        fpn = features["backbone_fpn"]
        return {
            "c1": fpn[0],
            "c2": fpn[1],
            "c3": features["vision_features"],
        }

    @staticmethod
    def _limit_candidates(
        outputs: list[Dict[str, Tensor]],
        ref_points: Tensor,
        max_candidates: int,
    ) -> tuple[list[Dict[str, Tensor]], Tensor]:
        if max_candidates <= 0:
            return outputs, ref_points

        limited_outputs: list[Dict[str, Tensor]] = []
        limited_points = ref_points.new_zeros((ref_points.shape[0], max_candidates, 2))
        for idx, output in enumerate(outputs):
            logits = output["box_v"].reshape(-1)
            num_candidates = int(logits.numel())
            if num_candidates == 0:
                limited_outputs.append(output)
                continue

            points = ref_points[idx, :num_candidates]
            keep_count = min(num_candidates, int(max_candidates))
            keep = torch.arange(keep_count, device=logits.device)
            boxes = output["pred_boxes"].reshape(-1, 4)[keep]
            kept_logits = logits[keep]
            limited_outputs.append(
                {
                    "pred_boxes": boxes.unsqueeze(0),
                    "boxes": boxes.unsqueeze(0),
                    "box_v": kept_logits.unsqueeze(0),
                }
            )
            limited_points[idx, :keep_count] = points[keep]

        return limited_outputs, limited_points

    def forward(self, images: Tensor) -> Dict[str, Dict[str, object]]:
        features = self.backbone(images)
        scale_features = self._scale_features(features)
        predictions: Dict[str, Dict[str, object]] = {}
        for scale in self.train_scales:
            centerness, offsets = self.heads[scale](scale_features[scale])
            outputs, ref_points = boxes_with_scores(
                centerness,
                offsets,
                sort=True,
                validate=False,
            )
            outputs, ref_points = self._limit_candidates(
                outputs,
                ref_points,
                self.max_candidates_per_scale[scale],
            )
            predictions[scale] = {
                "outputs": outputs,
                "ref_points": ref_points,
                "centerness": centerness,
            }
        return predictions


def merge_scale_outputs(
    predictions: Mapping[str, Mapping[str, object]],
    sample_idx: int,
) -> Dict[str, Tensor]:
    boxes = []
    logits = []
    for scale_prediction in predictions.values():
        output_i = scale_prediction["outputs"][sample_idx]
        boxes.append(output_i["pred_boxes"].reshape(-1, 4))
        logits.append(output_i["box_v"].reshape(-1))
    if not boxes:
        device = next(iter(predictions.values()))["centerness"].device
        return {
            "pred_boxes": torch.zeros((0, 4), device=device),
            "boxes": torch.zeros((0, 4), device=device),
            "box_v": torch.zeros((0,), device=device),
        }
    merged_boxes = torch.cat(boxes, dim=0)
    merged_logits = torch.cat(logits, dim=0)
    return {
        "pred_boxes": merged_boxes,
        "boxes": merged_boxes,
        "box_v": merged_logits,
    }
