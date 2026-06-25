# Building Hungarian Matcher
# Borrow code from AnchorDETR
# We replace bounding box matching with point location matching
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.utils.box_ops import generalized_box_iou, box_iou

class PointLossHungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @staticmethod
    def _box_centers(boxes):
        return 0.5 * (boxes[:, :2] + boxes[:, 2:])

    def _fallback_match_zero_iou_gt(self, out_bbox, tgt_bbox, matched_pred, matched_gt, zero_iou_gt):
        if zero_iou_gt.numel() == 0:
            return matched_pred, matched_gt

        keep = ~torch.isin(matched_gt, zero_iou_gt)
        final_pred = matched_pred[keep].clone()
        final_gt = matched_gt[keep].clone()
        used_pred = set(final_pred.cpu().tolist())

        pred_centers = self._box_centers(out_bbox)
        gt_centers = self._box_centers(tgt_bbox)
        distances = torch.cdist(pred_centers, gt_centers[zero_iou_gt], p=2)

        for local_gt_idx, gt_idx in enumerate(zero_iou_gt):
            order = torch.argsort(distances[:, local_gt_idx])
            chosen = None
            for pred_idx in order.cpu().tolist():
                if pred_idx not in used_pred:
                    chosen = pred_idx
                    break
            if chosen is None:
                continue
            used_pred.add(chosen)
            final_pred = torch.cat(
                [final_pred, torch.as_tensor([chosen], dtype=torch.int64, device=out_bbox.device)]
            )
            final_gt = torch.cat([final_gt, gt_idx.reshape(1).to(device=out_bbox.device)])

        return final_pred, final_gt

    def forward(self, outputs, targets, ref_points=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "box_v": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["box_v"].shape[:2]
            if bs != 1:
                raise ValueError("PointLossHungarianMatcher currently expects a single image per criterion call")

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["box_v"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

           # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            if tgt_bbox.numel() == 0 or out_bbox.numel() == 0:
                empty_pred = torch.empty(0, dtype=torch.int64, device=out_bbox.device)
                empty_gt = torch.empty(0, dtype=torch.int64, device=out_bbox.device)
                non_matched_pred = np.arange(out_bbox.shape[0])
                return [(empty_pred, empty_gt)], [empty_gt], non_matched_pred

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_class = -out_prob[:, None]

            # Compute the giou cost betwen boxes
            iou, unions = box_iou(out_bbox, tgt_bbox)
            cost_giou = - generalized_box_iou(out_bbox, tgt_bbox)
            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            device = out_bbox.device
            matched_pred = torch.as_tensor(indices[0][0], dtype=torch.int64, device=device)
            matched_gt = torch.as_tensor(indices[0][1], dtype=torch.int64, device=device)
            zero_iou_gt = torch.where(iou.max(dim=0)[0] == 0)[0].to(device=device)
            matched_pred, matched_gt = self._fallback_match_zero_iou_gt(
                out_bbox,
                tgt_bbox,
                matched_pred,
                matched_gt,
                zero_iou_gt,
            )
            all_gt = torch.arange(tgt_bbox.shape[0], dtype=torch.int64, device=device)
            if matched_gt.numel() > 0:
                gt_unmatched_mask = ~torch.isin(all_gt, matched_gt)
                pred_unmatched_mask = ~torch.isin(
                    torch.arange(out_bbox.shape[0], dtype=torch.int64, device=device),
                    matched_pred,
                )
            else:
                gt_unmatched_mask = torch.ones_like(all_gt, dtype=torch.bool)
                pred_unmatched_mask = torch.ones(out_bbox.shape[0], dtype=torch.bool, device=device)
            non_mathced_gt_bbox_idx = [all_gt[gt_unmatched_mask]]
            ind0 = matched_pred
            ind1 = matched_gt
            non_mathced_pred_bbox_idx = \
                torch.arange(out_bbox.shape[0], dtype=torch.int64, device=device)[pred_unmatched_mask].cpu().numpy()

            match_indexes = [
                (
                    ind0.to(dtype=torch.int64, device=device),
                    ind1.to(dtype=torch.int64, device=device),
                )
            ]
            return match_indexes, non_mathced_gt_bbox_idx, non_mathced_pred_bbox_idx

def build_matcher(args):
    return PointLossHungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)

