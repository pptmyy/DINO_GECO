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
        return_candidate_features: bool = False,

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
        self.return_candidate_features = return_candidate_features

        self.class_embed = nn.Linear(self.emb_dim, 1)
        self.bbox_embed = MLP(self.emb_dim, self.emb_dim, 4, 3)
        if not self.pretrain:
            self.class_embed_aux = nn.Linear(self.emb_dim, 1)
            self.bbox_embed_aux = MLP(self.emb_dim, self.emb_dim, 4, 3)        
        
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
        self.prototype_geometry_src = nn.Linear(self.emb_dim, self.emb_dim)
        self.prototype_geometry_l1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.prototype_geometry_l2 = nn.Linear(self.emb_dim, self.emb_dim)

        self.adapt_features = QueryGenerator(
            transformer_dim= self.emb_dim,
            num_prototype_attn_steps = 3,
            num_image_attn_steps=2,
        )

        if self.validate:
            from .box_corr import Box_correction
            self.box_correction = Box_correction(self.reduction,self.image_size ,self.emb_dim)

    def _attach_verification_features(self, outputs, ref_points, candidate_tokens, exemplar_tokens, height, width):
        feature_map = candidate_tokens.view(candidate_tokens.shape[0], height, width, self.emb_dim)
        for idx, output_i in enumerate(outputs):
            num_candidates = int(output_i["box_v"].reshape(-1).shape[0])
            if num_candidates == 0:
                output_i["candidate_features"] = feature_map.new_zeros((0, self.emb_dim))
            else:
                points = ref_points[idx, :num_candidates].long()
                output_i["candidate_features"] = feature_map[idx, points[:, 0], points[:, 1]].detach()
            output_i["exemplar_features"] = exemplar_tokens[idx].detach()

    def forward(self,x,bboxes):
        if bboxes.dim() != 3 or bboxes.shape[-1] != 4:
            raise ValueError(
                f"DGECO expects bboxes with shape [B, num_objects, 4], got {tuple(bboxes.shape)}"
            )
        
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects

  
        feats = self.backbone(x)
        src = feats['vision_features']

        bs,c,w,h = src.shape

        current_reduction = self.image_size / float(w)
        
        bboxes_roi = torch.cat([
            torch.arange(
                bs,requires_grad=False
            ).to(bboxes.device).repeat_interleave(num_objects).reshape(-1,1),
            bboxes.flatten(0,1),
        ],dim= 1)

        #self.kernel_dim = 1

        exemplars = roi_align(
            src,
            boxes = bboxes_roi, output_size=self.kernel_dim,
            spatial_scale = 1.0 / current_reduction, aligned = True
        ).permute(0,2,3,1).reshape(bs,num_objects*self.kernel_dim**2,self.emb_dim)

        l1 = feats['backbone_fpn'][0]
        l2 = feats['backbone_fpn'][1]

        exemplars_l1 = roi_align(
            l1,
            boxes = bboxes_roi, output_size = self.kernel_dim,
            spatial_scale = 1.0 / current_reduction * 2 * 2, aligned = True
        ).permute(0,2,3,1).reshape(bs,num_objects * self.kernel_dim**2,self.emb_dim)

        exemplars_l2 = roi_align(
            l2,
            boxes=bboxes_roi, output_size=self.kernel_dim,
            spatial_scale=1.0 / current_reduction * 2, aligned=True
        ).permute(0, 2, 3, 1).reshape(bs, num_objects * self.kernel_dim ** 2, self.emb_dim)

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
        shape = self.prototype_geometry_src(geometry_base).reshape(bs, -1, self.emb_dim)
        shape_l1 = self.prototype_geometry_l1(geometry_base).reshape(bs, -1, self.emb_dim)
        shape_l2 = self.prototype_geometry_l2(geometry_base).reshape(bs, -1, self.emb_dim)

        prototype_embeddings = torch.cat([exemplars, shape], dim=1)
        prototype_embeddings_l1 = torch.cat([exemplars_l1, shape_l1], dim=1)
        prototype_embeddings_l2 = torch.cat([exemplars_l2, shape_l2], dim=1)
        hq_prototype_embeddings = [prototype_embeddings_l1, prototype_embeddings_l2]


        adapted_f, adapted_f_aux = self.adapt_features(
            image_embeddings=src,
            image_pe=self.dino_prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=feats['backbone_fpn'],
            hq_prototypes=hq_prototype_embeddings,
            hq_pos=feats['vision_pos_enc'],
        )

        # Predict class [fg, bg] and l,r,t,b
        bs, c, w, h = adapted_f.shape
        adapted_f = adapted_f.view(bs, self.emb_dim, -1).permute(0, 2, 1)
        centerness = self.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
        outputs_coord = self.bbox_embed(adapted_f).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        outputs, ref_points = boxes_with_scores(centerness, outputs_coord, sort=False, validate=self.validate)
        if self.return_candidate_features and not self.validate:
            self._attach_verification_features(outputs, ref_points, adapted_f, exemplars, w, h)

        if not self.pretrain:
            adapted_f_aux = adapted_f_aux.view(bs, self.emb_dim, -1).permute(0, 2, 1)
            centerness_aux = self.class_embed_aux(adapted_f_aux).view(bs, w, h, 1).permute(0, 3, 1, 2)
            outputs_coord_aux = self.bbox_embed_aux(adapted_f_aux).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
            outputs_aux, ref_points_aux = boxes_with_scores(centerness_aux, outputs_coord_aux, sort=False, validate=self.validate)

        if self.validate:
            outputs, masks = self.box_correction(feats, outputs, x)

        else:
            for i in range(len(outputs)):
                outputs[i]["scores"] = outputs[i]["box_v"]

        if self.pretrain:
            return outputs, ref_points, centerness, outputs_coord
        else:
            return outputs, ref_points, centerness, outputs_coord, (
                outputs_aux,
                ref_points_aux,
                centerness_aux,
                outputs_coord_aux,
            )
        



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
        return_candidate_features=getattr(args, "verification_mode", "none") == "feature_similarity",
    )
