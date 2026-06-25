from typing import Tuple
from typing import Optional

import torch

from torch import nn

from .scale_query_aggregator import ScaleAwareQueryAggregator
from .transformer import PrototypeAttentionBlock

from .DeformableDETR.models.ops.modules.ms_deform_attn import MSDeformAttn

class QueryGenerator(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            num_prototype_attn_steps: int,
            num_image_attn_steps: int,
            output_stride: int = 4,
            num_prototypes: int = 4,
            prototype_ema_momentum: float = 0.9,

    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.image_attention = nn.ModuleList()
        self.image_attention_l1 = nn.ModuleList()
        self.image_attention_l2 = nn.ModuleList()

        self.prototype_attention = nn.ModuleList()
        self.prototype_attention_l1 = nn.ModuleList()
        self.prototype_attention_l2 = nn.ModuleList()

        for _ in range(num_prototype_attn_steps):
            self.prototype_attention.append(
                PrototypeAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                )
            )
            self.prototype_attention_l1.append(
                PrototypeAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                )
            )
            self.prototype_attention_l2.append(
                PrototypeAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                )
            )

        for _ in range(num_image_attn_steps):
            self.image_attention.append(MSDeformAttn(
                d_model=transformer_dim, n_levels=1, n_heads=8, n_points=8))

            self.image_attention_l1.append(MSDeformAttn(
                d_model=transformer_dim, n_levels=1, n_heads=8, n_points=8))

            self.image_attention_l2.append(MSDeformAttn(
                d_model=transformer_dim, n_levels=1, n_heads=8, n_points=8))

        self.scale_query_aggregator = ScaleAwareQueryAggregator(
            transformer_dim,
            output_stride=output_stride,
            num_prototypes=num_prototypes,
            prototype_ema_momentum=prototype_ema_momentum,
        )

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    @staticmethod
    def _deformable_geometry(height: int, width: int, device):
        spatial_shapes = torch.tensor(
            [[height, width]],
            dtype=torch.long,
            device=device,
        )
        valid_ratios = torch.ones((1, 2), dtype=torch.float32, device=device)
        level_start_index = torch.zeros((1,), dtype=torch.long, device=device)
        reference_points = QueryGenerator.get_reference_points(
            spatial_shapes,
            valid_ratios,
            device=device,
        )
        return spatial_shapes, level_start_index, reference_points

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device='cpu'):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            prototype_embeddings: torch.Tensor,
            hq_features: torch.Tensor,
            hq_prototypes: torch.Tensor,
            hq_pos: torch.Tensor,
            semantic_context: Optional[torch.Tensor] = None,
            prototype_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        """
        if len(hq_features) < 2 or len(hq_prototypes) < 2 or len(hq_pos) < 2:
            raise ValueError("QueryGenerator expects l1/l2 high-resolution features and prototypes")
        b, c, h, w = image_embeddings.shape
        _, _, h1, w1 = hq_features[0].shape
        _, _, h2, w2 = hq_features[1].shape
        spatial_shapes, level_start_index, reference_points = self._deformable_geometry(
            h,
            w,
            image_embeddings.device,
        )
        spatial_shapes1, level_start_index1, reference_points1 = self._deformable_geometry(
            h1,
            w1,
            image_embeddings.device,
        )
        spatial_shapes2, level_start_index2, reference_points2 = self._deformable_geometry(
            h2,
            w2,
            image_embeddings.device,
        )
        image_pe = torch.repeat_interleave(image_pe, image_embeddings.shape[0], dim=0) # pe 是 position_encode
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        src = image_embeddings


        hq_features_l1_pos = hq_pos[0].flatten(2).permute(0, 2, 1)
        hq_features_l2_pos = hq_pos[1].flatten(2).permute(0, 2, 1)

        hq_features_l1 = hq_features[0].flatten(2).permute(0, 2, 1)
        hq_features_l2 = hq_features[1].flatten(2).permute(0, 2, 1)

        for layer in self.prototype_attention:
            src = layer(image_f=src, prototypes=prototype_embeddings)

        for layer in self.prototype_attention_l1:
            hq_features_l1 = layer(image_f=hq_features_l1, prototypes=hq_prototypes[0])

        for layer in self.prototype_attention_l2:
            hq_features_l2 = layer(image_f=hq_features_l2, prototypes=hq_prototypes[1])

        for layer in self.image_attention:
            src = layer((src + image_pe), reference_points, src, spatial_shapes, level_start_index)

        for layer in self.image_attention_l1:
            hq_features_l1 = layer(
                (hq_features_l1 + hq_features_l1_pos),
                reference_points1,
                hq_features_l1,
                spatial_shapes1,
                level_start_index1,
            )

        for layer in self.image_attention_l2:
            hq_features_l2 = layer(
                (hq_features_l2 + hq_features_l2_pos),
                reference_points2,
                hq_features_l2,
                spatial_shapes2,
                level_start_index2,
            )

        src = src.transpose(1, 2).reshape(b, c, h, w)
        hq_features_l2 = hq_features_l2.transpose(1, 2).view(b, c, h2, w2)
        hq_features_l1 = hq_features_l1.transpose(1, 2).view(b, c, h1, w1)

        src, src_aux = self.scale_query_aggregator(
            q3=src,
            q2=hq_features_l2,
            q1=hq_features_l1,
            prototype_embeddings=prototype_embeddings,
            hq_prototypes=hq_prototypes,
            semantic_context=semantic_context,
            prototype_memory=prototype_memory,
        )

        return src, src_aux
