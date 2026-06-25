from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .regression_head import UpsamplingLayer


def _make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ScaleFusionBlock(nn.Module):
    """Fuse an upsampled semantic query into a higher-resolution local query."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.high_gate = nn.Conv2d(channels, 1, kernel_size=1)
        self.local_gate = nn.Conv2d(channels, 1, kernel_size=1)
        self.context_gate = nn.Linear(channels, channels)
        self.norm = _make_group_norm(channels)

    @staticmethod
    def _best_prototype_gate(query: Tensor, prototypes: Tensor) -> Tensor:
        query = F.normalize(query, dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        similarity = torch.einsum("bchw,bkc->bkhw", query, prototypes)
        best_similarity = similarity.max(dim=1, keepdim=True).values
        return torch.sigmoid(best_similarity * 2.0)

    def forward(self, high_query: Tensor, local_query: Tensor, context: Tensor) -> Tensor:
        if high_query.shape[-2:] != local_query.shape[-2:]:
            high_query = F.interpolate(
                high_query,
                size=local_query.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        spatial_gate = torch.sigmoid(self.high_gate(high_query) + self.local_gate(local_query))
        if context.dim() == 3:
            spatial_gate = spatial_gate * self._best_prototype_gate(high_query + local_query, context)
            channel_context = context.mean(dim=1)
        else:
            channel_context = context
        channel_gate = torch.sigmoid(self.context_gate(channel_context)).view(
            channel_context.shape[0],
            channel_context.shape[1],
            1,
            1,
        )
        fused = local_query + spatial_gate * channel_gate * high_query
        return F.gelu(self.norm(fused))


class ScaleAwareQueryAggregator(nn.Module):
    """Gradually aggregate exemplar-specific queries from semantic to detail scales."""

    def __init__(
        self,
        channels: int,
        output_stride: int = 4,
        num_prototypes: int = 4,
        prototype_ema_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        if output_stride not in {2, 4, 8}:
            raise ValueError(f"output_stride must be one of 2, 4, 8; got {output_stride}")
        self.output_stride = int(output_stride)
        self.num_prototypes = int(num_prototypes)
        self.prototype_ema_momentum = float(prototype_ema_momentum)
        self.up_c3_to_c2 = UpsamplingLayer(channels, channels)
        self.up_c2_to_c1 = UpsamplingLayer(channels, channels)
        self.up_c1_to_out = UpsamplingLayer(channels, channels)
        self.fuse_c2 = ScaleFusionBlock(channels)
        self.fuse_c1 = ScaleFusionBlock(channels)
        self.semantic_context = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
        )

    @staticmethod
    def _prototype_context(prototypes: Tensor) -> Tensor:
        if prototypes.numel() == 0:
            return prototypes.new_zeros((prototypes.shape[0], prototypes.shape[-1]))
        return prototypes.mean(dim=1)

    def _pool_prototypes(self, prototypes: Tensor) -> Tensor:
        if prototypes.numel() == 0:
            return prototypes.new_zeros(
                (prototypes.shape[0], self.num_prototypes, prototypes.shape[-1])
            )
        pooled = F.adaptive_avg_pool1d(
            prototypes.transpose(1, 2),
            self.num_prototypes,
        ).transpose(1, 2)
        return pooled

    def _blend_memory(self, base: Tensor, memory: Tensor | None) -> Tensor:
        if memory is None:
            return base
        if memory.shape[1] != base.shape[1]:
            memory = F.adaptive_avg_pool1d(
                memory.transpose(1, 2),
                base.shape[1],
            ).transpose(1, 2)
        momentum = max(0.0, min(1.0, self.prototype_ema_momentum))
        return momentum * base + (1.0 - momentum) * memory.detach()

    def forward(
        self,
        *,
        q3: Tensor,
        q2: Tensor,
        q1: Tensor,
        prototype_embeddings: Tensor,
        hq_prototypes: Sequence[Tensor],
        semantic_context: Tensor | None = None,
        prototype_memory: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(hq_prototypes) < 2:
            raise ValueError("ScaleAwareQueryAggregator expects l1/l2 prototype tensors")

        context_base = self._prototype_context(prototype_embeddings)
        if semantic_context is not None:
            context_base = 0.5 * (context_base + self.semantic_context(semantic_context))
        proto_l2 = self._blend_memory(self._pool_prototypes(hq_prototypes[1]), prototype_memory)
        proto_l1 = self._blend_memory(self._pool_prototypes(hq_prototypes[0]), prototype_memory)
        context_l2 = 0.5 * (proto_l2 + context_base.unsqueeze(1))
        context_l1 = 0.5 * (proto_l1 + context_base.unsqueeze(1))

        q2 = self.fuse_c2(self.up_c3_to_c2(q3), q2, context_l2)
        q1 = self.fuse_c1(self.up_c2_to_c1(q2), q1, context_l1)

        if self.output_stride == 8:
            q1_to_q2 = F.interpolate(
                q1,
                size=q2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            q_out = 0.5 * (q2 + q1_to_q2)
        elif self.output_stride == 4:
            q_out = q1
        else:
            q_out = self.up_c1_to_out(q1)
        q_refine = self.up_c1_to_out(q1)
        return q_out, q_refine
