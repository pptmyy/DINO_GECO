from typing import Tuple, Union
from torch.nn import functional as F

import torch
from torch import nn
from torchvision.ops import roi_align
from .backbones.dinov3_adapter import DINOv3Adapter
from src.utils.box_ops import boxes_with_scores
from .query_generator import QueryGenerator
from .prompt_encoder import PromptEncoder


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DGECO(nn.Module):
    """
 
    2. x -> DINOv3Adapter

    """

    def __init__(
        self,
        *,
        image_size: int = 1024,
        # DINOv3 参数
        dinov3_model_size: str = "base",
        dinov3_patch_size: int = 16,
        dinov3_out_feature_indexes: Tuple[int, int, int, int] = (2, 5, 8, 11),
        dinov3_freeze_encoder: bool = True,
        dinov3_pretrained_weights: Union[str, None] = None,
        dinov3_conv_inplane: int = 16,
    
        #网络训练的数量
        num_objects:int,
        kernel_dim:int,
        zero_shot:bool,
        training:bool,
        reduction:int,
        query_output_stride: int = 4,
        use_semantic_anchor: bool = False,
        return_candidate_features: bool = False,
        stride2_refinement: bool = True,
        center_gaussian_head: bool = True,
        num_prototypes: int = 4,
        prototype_pred_topk: int = 32,
        prototype_pred_score_threshold: float = 0.5,
        prototype_ema_momentum: float = 0.9,
        mutual_adapter_layers: int = 1,
        decoupled_heads: bool = True,

    ):
        super().__init__()

        


        self.backbone = DINOv3Adapter(
            image_size=image_size,
            model_size=dinov3_model_size,
            patch_size=dinov3_patch_size,
            out_feature_indexes=list(dinov3_out_feature_indexes),
            freeze_encoder=dinov3_freeze_encoder,
            pretrained_weights=dinov3_pretrained_weights,
            conv_inplane=dinov3_conv_inplane,
            mutual_adapter_layers=mutual_adapter_layers,
        )
        self.dinov3_adapter = self.backbone
        self.validate = not training

        self.emb_dim = self.backbone.dinov3.embed_dim

        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.pretrain = False
        self.num_objects = num_objects
        self.query_output_stride = int(query_output_stride)
        self.use_semantic_anchor = bool(use_semantic_anchor)
        self.return_candidate_features = return_candidate_features
        self.stride2_refinement = bool(stride2_refinement)
        self.center_gaussian_head_enabled = bool(center_gaussian_head)
        self.num_prototypes = int(num_prototypes)
        self.prototype_pred_topk = int(prototype_pred_topk)
        self.prototype_pred_score_threshold = float(prototype_pred_score_threshold)
        self.prototype_ema_momentum = float(prototype_ema_momentum)
        self.decoupled_heads = bool(decoupled_heads)

        self.class_embed = nn.Linear(self.emb_dim, 1)
        self.bbox_embed = MLP(self.emb_dim, self.emb_dim, 4, 3)
        if not self.pretrain:
            self.class_embed_aux = nn.Linear(self.emb_dim, 1)
            self.bbox_embed_aux = MLP(self.emb_dim, self.emb_dim, 4, 3)        
        self.proto_sem = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
        )
        self.proto_geo = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
        )
        self.center_heatmap_head = nn.Sequential(
            nn.Conv2d(self.emb_dim, self.emb_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.emb_dim, 1, kernel_size=1),
        )
        
        self.dino_prompt_encoder = PromptEncoder(
            embed_dim=self.emb_dim,
            image_embedding_size=(
                self.image_size // self.reduction,
                self.image_size // self.reduction,
            ),
            input_image_size=(self.image_size,self.image_size),
            mask_in_chans=16,
        )

        geometry_dim = 9
        self.prototype_geometry = nn.Sequential(
            nn.Linear(geometry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.emb_dim),
            nn.ReLU(),
        )

        self.adapt_features = QueryGenerator(
            transformer_dim= self.emb_dim,
            num_prototype_attn_steps = 3,
            num_image_attn_steps=2,
            output_stride=self.query_output_stride,
            num_prototypes=self.num_prototypes,
            prototype_ema_momentum=self.prototype_ema_momentum,
        )

        if self.validate:
            from .box_corr import Box_correction
            self.box_correction = Box_correction(self.reduction,self.image_size ,self.emb_dim)

    def _roi_align_exemplars(self, feature, bboxes_roi, bs, num_objects):
        spatial_scale = float(feature.shape[-1]) / float(self.image_size)
        return roi_align(
            feature,
            boxes=bboxes_roi,
            output_size=self.kernel_dim,
            spatial_scale=spatial_scale,
            aligned=True,
        ).permute(0, 2, 3, 1).reshape(
            bs,
            num_objects * self.kernel_dim**2,
            self.emb_dim,
        )

    def _attach_verification_features(self, outputs, ref_points, candidate_tokens, exemplar_tokens, height, width):
        feature_map = candidate_tokens.reshape(candidate_tokens.shape[0], height, width, self.emb_dim)
        for idx, output_i in enumerate(outputs):
            num_candidates = int(output_i["box_v"].reshape(-1).shape[0])
            if num_candidates == 0:
                output_i["candidate_features"] = feature_map.new_zeros((0, self.emb_dim))
            else:
                points = ref_points[idx, :num_candidates].long()
                output_i["candidate_features"] = feature_map[idx, points[:, 0], points[:, 1]].detach()
            output_i["exemplar_features"] = exemplar_tokens[idx].detach()

    def _pool_to_k(self, tokens, num_tokens):
        if tokens.numel() == 0:
            return tokens.new_zeros((num_tokens, self.emb_dim))
        return F.adaptive_avg_pool1d(
            tokens.transpose(0, 1).unsqueeze(0),
            num_tokens,
        ).squeeze(0).transpose(0, 1)

    def _episode_memory_prototypes(self, feature, prototype_embeddings):
        if self.num_prototypes <= 0 or self.prototype_pred_topk <= 0:
            return None
        tokens = feature.flatten(2).transpose(1, 2)
        base_context = prototype_embeddings.mean(dim=1)
        token_norm = F.normalize(tokens, dim=-1)
        context_norm = F.normalize(base_context, dim=-1)
        scores = torch.einsum("bnc,bc->bn", token_norm, context_norm)
        topk = min(self.prototype_pred_topk, tokens.shape[1])
        top_scores, top_indices = torch.topk(scores, k=topk, dim=1)

        memories = []
        for batch_idx in range(tokens.shape[0]):
            selected = tokens[batch_idx, top_indices[batch_idx]]
            keep = top_scores[batch_idx] >= self.prototype_pred_score_threshold
            if keep.any():
                selected = selected[keep]
            memories.append(self._pool_to_k(selected.detach(), self.num_prototypes))
        return torch.stack(memories, dim=0)

    def _prototype_contexts(self, exemplars, geometry_base, semantic_anchor):
        sem_context = exemplars.mean(dim=1)
        if self.use_semantic_anchor and semantic_anchor is not None:
            sem_context = 0.5 * (sem_context + semantic_anchor)
        geo_context = geometry_base.mean(dim=1)
        return sem_context, geo_context

    def _predict_dense(self, feature, class_head, bbox_head, sem_context, geo_context):
        bs, _, height, width = feature.shape
        if self.decoupled_heads:
            sem_feature = feature + self.proto_sem(sem_context).view(bs, self.emb_dim, 1, 1)
            geo_feature = feature + self.proto_geo(geo_context).view(bs, self.emb_dim, 1, 1)
        else:
            sem_feature = feature
            geo_feature = feature

        sem_tokens = sem_feature.flatten(2).transpose(1, 2)
        geo_tokens = geo_feature.flatten(2).transpose(1, 2)
        centerness = class_head(sem_tokens).view(bs, height, width, 1).permute(0, 3, 1, 2)
        boxes = bbox_head(geo_tokens).sigmoid().view(bs, height, width, 4).permute(0, 3, 1, 2)
        return centerness, boxes, sem_tokens, height, width

    def forward(self,x,bboxes):
        if bboxes.dim() != 3 or bboxes.shape[-1] != 4:
            raise ValueError(
                f"DGECO expects bboxes with shape [B, num_objects, 4], got {tuple(bboxes.shape)}"
            )
        
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects

  
        feats = self.backbone(x, bboxes)
        src = feats['vision_features']

        bs, _, _, _ = src.shape
        
        bboxes_roi = torch.cat([
            torch.arange(
                bs,requires_grad=False
            ).to(bboxes.device).repeat_interleave(num_objects).reshape(-1,1),
            bboxes.flatten(0,1),
        ],dim= 1)

        #self.kernel_dim = 1

        exemplars = self._roi_align_exemplars(src, bboxes_roi, bs, num_objects)

        l1 = feats['backbone_fpn'][0]
        l2 = feats['backbone_fpn'][1]

        exemplars_l1 = self._roi_align_exemplars(l1, bboxes_roi, bs, num_objects)
        exemplars_l2 = self._roi_align_exemplars(l2, bboxes_roi, bs, num_objects)

        box_w = (bboxes[:, :, 2] - bboxes[:, :, 0]).clamp(min=1.0)
        box_h = (bboxes[:, :, 3] - bboxes[:, :, 1]).clamp(min=1.0)
        box_area = box_w * box_h
        image_area = float(self.image_size * self.image_size)
        geometry = torch.stack(
            [
                box_w / self.image_size,
                box_h / self.image_size,
                box_area / image_area,
                box_w / box_h,
                box_h / box_w,
                torch.log(box_w) / torch.log(box_w.new_tensor(float(self.image_size))),
                torch.log(box_h) / torch.log(box_h.new_tensor(float(self.image_size))),
                torch.sqrt(box_area) / self.image_size,
                torch.log(box_area) / torch.log(box_area.new_tensor(image_area)),
            ],
            dim=-1,
        )
        geometry_base = self.prototype_geometry(geometry)
        shape = geometry_base.reshape(bs, -1, self.emb_dim)

        prototype_embeddings = torch.cat([exemplars, shape], dim=1)
        prototype_embeddings_l1 = torch.cat([exemplars_l1, shape], dim=1)
        prototype_embeddings_l2 = torch.cat([exemplars_l2, shape], dim=1)
        hq_prototype_embeddings = [prototype_embeddings_l1, prototype_embeddings_l2]
        semantic_anchor = feats.get("semantic_anchor")
        sem_context, geo_context = self._prototype_contexts(
            exemplars,
            geometry_base,
            semantic_anchor,
        )
        prototype_memory = self._episode_memory_prototypes(src, prototype_embeddings)


        adapt_kwargs = {
            "image_embeddings": src,
            "image_pe": self.dino_prompt_encoder.get_dense_pe(),
            "prototype_embeddings": prototype_embeddings,
            "hq_features": feats['backbone_fpn'],
            "hq_prototypes": hq_prototype_embeddings,
            "hq_pos": feats['vision_pos_enc'],
            "prototype_memory": prototype_memory,
        }
        if self.use_semantic_anchor:
            adapt_kwargs["semantic_context"] = semantic_anchor
        adapted_f, refinement_f = self.adapt_features(**adapt_kwargs)
        if not self.stride2_refinement:
            refinement_f = adapted_f

        centerness, outputs_coord, main_tokens, main_h, main_w = self._predict_dense(
            adapted_f,
            self.class_embed,
            self.bbox_embed,
            sem_context,
            geo_context,
        )
        outputs, ref_points = boxes_with_scores(centerness, outputs_coord, sort=False, validate=self.validate)
        if self.return_candidate_features and not self.validate:
            self._attach_verification_features(
                outputs,
                ref_points,
                main_tokens,
                exemplars,
                main_h,
                main_w,
            )

        if not self.pretrain:
            centerness_refine, outputs_coord_refine, _, _, _ = self._predict_dense(
                refinement_f,
                self.class_embed_aux,
                self.bbox_embed_aux,
                sem_context,
                geo_context,
            )
            outputs_refine, ref_points_refine = boxes_with_scores(
                centerness_refine,
                outputs_coord_refine,
                sort=False,
                validate=self.validate,
            )
            center_heatmap = (
                self.center_heatmap_head(refinement_f)
                if self.center_gaussian_head_enabled
                else None
            )

        if self.validate:
            outputs, masks = self.box_correction(feats, outputs, x)

        else:
            for i in range(len(outputs)):
                outputs[i]["scores"] = outputs[i]["box_v"]

        if self.pretrain:
            return outputs, ref_points, centerness, outputs_coord
        else:
            return outputs, ref_points, centerness, outputs_coord, {
                "refine_outputs": outputs_refine,
                "refine_ref_points": ref_points_refine,
                "refine_centerness": centerness_refine,
                "refine_boxes": outputs_coord_refine,
                "center_heatmap": center_heatmap,
            }
        



def build_model(args):

    return DGECO(
        image_size=args.image_size,
        dinov3_model_size= args.dinov3_model_size,
        dinov3_patch_size= 16,
        dinov3_pretrained_weights=args.dinov3_pretrained_weights,
        training= args.training,
        num_objects=args.num_objects,
        kernel_dim=args.kernel_dim,
        zero_shot=args.zero_shot,
        reduction=args.reduction,
        query_output_stride=getattr(args, "query_output_stride", 4),
        use_semantic_anchor=getattr(args, "use_semantic_anchor", False),
        return_candidate_features=getattr(args, "verification_mode", "none") == "feature_similarity",
        stride2_refinement=getattr(args, "stride2_refinement", True),
        center_gaussian_head=getattr(args, "center_gaussian_head", True),
        num_prototypes=getattr(args, "num_prototypes", 4),
        prototype_pred_topk=getattr(args, "prototype_pred_topk", 32),
        prototype_pred_score_threshold=getattr(args, "prototype_pred_score_threshold", 0.5),
        prototype_ema_momentum=getattr(args, "prototype_ema_momentum", 0.9),
        mutual_adapter_layers=getattr(args, "mutual_adapter_layers", 1),
        decoupled_heads=getattr(args, "decoupled_heads", True),
    )
