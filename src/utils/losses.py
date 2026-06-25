import torch
import torch.nn.functional as F
from torch import nn

from src.utils import box_ops


def quality_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Focal-style objectness loss for soft IoU/center-quality targets."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    scale = (targets - prob).abs().pow(gamma)
    if alpha >= 0:
        alpha_t = torch.where(targets > 0, alpha, 1 - alpha)
        ce_loss = alpha_t * ce_loss
    return ce_loss * scale


class SetCriterion(nn.Module):
    """Single-image detection criterion used by the current training loop."""

    CENTER_RADIUS = 1
    CENTER_NEIGHBOR_TARGET = 0.5

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        focal_alpha=0.25,
        center_gaussian_sigma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.center_gaussian_sigma = float(center_gaussian_sigma)

    def loss_boxes(self, outputs, targets, indices, num_boxes, centerness, centerness_gt, mask):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices, outputs["pred_boxes"].device)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if src_boxes.numel() == 0 or target_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def ce_loss(self, outputs, targets, indices, num_boxes, centerness, centerness_gt, mask):
        if mask.sum() == 0:
            return {"loss_ce": centerness.sum() * 0.0}

        loss_ce = quality_focal_loss(
            centerness[mask > 0],
            centerness_gt[mask > 0],
            alpha=self.focal_alpha,
            gamma=2.0,
        )
        return {"loss_ce": loss_ce.mean()}

    def _get_src_permutation_idx(self, indices, device):
        if len(indices) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty
        batch_idx = torch.cat(
            [torch.full_like(src, i, device=device) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src.to(device) for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, centerness, centerness_gt, mask):
        loss_map = {
            "bboxes": self.loss_boxes,
            "ce": self.ce_loss,
        }
        if loss not in loss_map:
            raise ValueError(f"Unsupported loss: {loss}")
        return loss_map[loss](
            outputs,
            targets,
            indices,
            num_boxes,
            centerness,
            centerness_gt,
            mask,
        )

    def _mark_centers(self, centerness_gt, mask, boxes):
        if boxes.numel() == 0:
            return
        grid_h, grid_w = centerness_gt.shape[-2:]
        scaled = boxes * boxes.new_tensor([grid_w, grid_h, grid_w, grid_h])
        y = torch.clamp(((scaled[:, 3] + scaled[:, 1]) / 2).long(), min=0, max=grid_h - 1)
        x = torch.clamp(((scaled[:, 2] + scaled[:, 0]) / 2).long(), min=0, max=grid_w - 1)
        for dy in range(-self.CENTER_RADIUS, self.CENTER_RADIUS + 1):
            for dx in range(-self.CENTER_RADIUS, self.CENTER_RADIUS + 1):
                yy = torch.clamp(y + dy, min=0, max=grid_h - 1)
                xx = torch.clamp(x + dx, min=0, max=grid_w - 1)
                target = 1.0 if dy == 0 and dx == 0 else self.CENTER_NEIGHBOR_TARGET
                current = centerness_gt[0, yy, xx]
                target_tensor = torch.full_like(current, target)
                centerness_gt[0, yy, xx] = torch.maximum(current, target_tensor)
                mask[0, yy, xx] = 1

    def _matched_quality_targets(self, outputs, targets, indices):
        if len(indices) == 0:
            device = outputs["pred_boxes"].device
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=outputs["pred_boxes"].dtype, device=device),
            )

        device = outputs["pred_boxes"].device
        matched_pred_idx, matched_gt_idx = indices[0]
        matched_pred_idx = matched_pred_idx.to(device=device, dtype=torch.long)
        matched_gt_idx = matched_gt_idx.to(device=device, dtype=torch.long)
        if matched_pred_idx.numel() == 0:
            return matched_pred_idx, outputs["pred_boxes"].new_empty(0)

        pred_boxes = outputs["pred_boxes"][0, matched_pred_idx]
        target_boxes = targets[0]["boxes"].to(device=device)[matched_gt_idx]
        iou, _ = box_ops.box_iou(pred_boxes, target_boxes)
        qualities = torch.diag(iou).clamp(min=0.0, max=1.0)
        return matched_pred_idx, qualities

    def generate_centerness_gt(self, indices, fn_idx, fp_idx, outputs, targets, centerness, ref_points):
        centerness_gt = torch.zeros_like(centerness)
        mask = torch.zeros_like(centerness)

        point_locs = ref_points if ref_points.shape[-1] == 2 else ref_points.permute(1, 0)
        point_locs = point_locs.to(device=centerness.device, dtype=torch.long)

        fp_idx = torch.as_tensor(fp_idx, dtype=torch.long, device=centerness.device)
        if fp_idx.numel() > 0 and point_locs.numel() > 0:
            fp_idx = fp_idx[fp_idx < point_locs.shape[0]]
            fp_locs = point_locs[fp_idx]
            if fp_locs.numel() > 0:
                mask[0, fp_locs[:, 0], fp_locs[:, 1]] = 1

        matched_pred_idx, matched_quality = self._matched_quality_targets(outputs, targets, indices)
        matched_pred_idx = matched_pred_idx.to(centerness.device)
        matched_quality = matched_quality.to(centerness.device)
        if matched_pred_idx.numel() > 0 and point_locs.numel() > 0:
            valid = matched_pred_idx < point_locs.shape[0]
            matched_pred_idx = matched_pred_idx[valid]
            matched_quality = matched_quality[valid]
            tp_locs = point_locs[matched_pred_idx]
            if tp_locs.numel() > 0:
                current = centerness_gt[0, tp_locs[:, 0], tp_locs[:, 1]]
                centerness_gt[0, tp_locs[:, 0], tp_locs[:, 1]] = torch.maximum(
                    current,
                    matched_quality,
                )
                mask[0, tp_locs[:, 0], tp_locs[:, 1]] = 1

        target_boxes = targets[0]["boxes"]
        fn_idx = fn_idx[0].to(target_boxes.device) if isinstance(fn_idx, list) else torch.as_tensor(fn_idx, dtype=torch.long, device=target_boxes.device)
        if fn_idx.numel() > 0:
            self._mark_centers(centerness_gt, mask, target_boxes[fn_idx])

        # Always supervise a small center neighborhood to improve recall on missed GTs.
        self._mark_centers(centerness_gt, mask, target_boxes)

        return centerness_gt, mask

    def weighted_total(self, losses):
        total = None
        for key, value in losses.items():
            if key in self.weight_dict:
                weighted = value * self.weight_dict[key]
                total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No weighted losses were produced")
        return total

    def generate_gaussian_center_gt(self, targets, center_heatmap):
        if center_heatmap.dim() == 4:
            center_heatmap = center_heatmap.squeeze(0)
        target = torch.zeros_like(center_heatmap)
        boxes = targets[0]["boxes"].to(device=center_heatmap.device, dtype=center_heatmap.dtype)
        if boxes.numel() == 0:
            return target

        _, grid_h, grid_w = center_heatmap.shape
        yy, xx = torch.meshgrid(
            torch.arange(grid_h, device=center_heatmap.device, dtype=center_heatmap.dtype),
            torch.arange(grid_w, device=center_heatmap.device, dtype=center_heatmap.dtype),
            indexing="ij",
        )
        sigma = max(self.center_gaussian_sigma, 1e-6)
        for box in boxes:
            center_x = ((box[0] + box[2]) * 0.5 * grid_w).clamp(0, grid_w - 1)
            center_y = ((box[1] + box[3]) * 0.5 * grid_h).clamp(0, grid_h - 1)
            gaussian = torch.exp(
                -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma)
            )
            target[0] = torch.maximum(target[0], gaussian)
        return target

    def center_gaussian_loss(self, targets, center_heatmap):
        if center_heatmap is None:
            raise ValueError("center_gaussian_loss requires a center_heatmap tensor")
        target = self.generate_gaussian_center_gt(targets, center_heatmap)
        loss = (center_heatmap.sigmoid() - target).pow(2)
        loss = (loss * (1.0 + 4.0 * target)).mean()
        return {
            "loss_center_gaussian": loss,
            "center_gaussian_positive_sum": (target > 0.01).sum().detach(),
        }

    def forward(self, outputs, targets, centerness, ref_points):
        indices, fn_idx, fp_idx = self.matcher(outputs, targets, ref_points=ref_points)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(float(num_boxes), 1.0)

        centerness_gt, mask = self.generate_centerness_gt(
            indices,
            fn_idx,
            fp_idx,
            outputs,
            targets,
            centerness,
            ref_points,
        )

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    indices,
                    num_boxes,
                    centerness,
                    centerness_gt,
                    mask,
                )
            )

        losses["loss_total"] = self.weighted_total(losses)
        losses["mask_sum"] = mask.sum().detach()
        losses["positive_sum"] = (centerness_gt > 0).sum().detach()
        losses["num_boxes"] = torch.as_tensor(num_boxes, device=centerness.device)
        return losses
