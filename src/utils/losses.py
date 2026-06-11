import torch
import torch.nn.functional as F
from torch import nn

from src.utils import box_ops


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


class SetCriterion(nn.Module):
    """Single-image detection criterion used by the current training loop."""

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

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

        loss_ce = sigmoid_focal_loss(
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
        centerness_gt[0, y, x] = 1
        mask[0, y, x] = 1

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

        matched_pred_idx = indices[0][0].to(centerness.device) if indices else torch.empty(0, dtype=torch.long, device=centerness.device)
        if matched_pred_idx.numel() > 0 and point_locs.numel() > 0:
            matched_pred_idx = matched_pred_idx[matched_pred_idx < point_locs.shape[0]]
            tp_locs = point_locs[matched_pred_idx]
            if tp_locs.numel() > 0:
                centerness_gt[0, tp_locs[:, 0], tp_locs[:, 1]] = 1
                mask[0, tp_locs[:, 0], tp_locs[:, 1]] = 1

        target_boxes = targets[0]["boxes"]
        fn_idx = fn_idx[0].to(target_boxes.device) if isinstance(fn_idx, list) else torch.as_tensor(fn_idx, dtype=torch.long, device=target_boxes.device)
        if fn_idx.numel() > 0:
            self._mark_centers(centerness_gt, mask, target_boxes[fn_idx])

        # If duplicate centers or weak matching left some GT without positive supervision,
        # add all GT centers while keeping existing FP negatives.
        if centerness_gt.sum() < target_boxes.shape[0]:
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

    def forward(self, outputs, targets, centerness, ref_points):
        indices, fn_idx, fp_idx = self.matcher(outputs, targets)
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
        losses["positive_sum"] = centerness_gt.sum().detach()
        losses["num_boxes"] = torch.as_tensor(num_boxes, device=centerness.device)
        return losses
