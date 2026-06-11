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

    def forward(self, high_query: Tensor, local_query: Tensor, context: Tensor) -> Tensor:
        if high_query.shape[-2:] != local_query.shape[-2:]:
            high_query = F.interpolate(
                high_query,
                size=local_query.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        spatial_gate = torch.sigmoid(self.high_gate(high_query) + self.local_gate(local_query))
        channel_gate = torch.sigmoid(self.context_gate(context)).view(
            context.shape[0],
            context.shape[1],
            1,
            1,
        )
        fused = local_query + spatial_gate * channel_gate * high_query
        return F.gelu(self.norm(fused))


class ScaleAwareQueryAggregator(nn.Module):
    """Gradually aggregate exemplar-specific queries from semantic to detail scales."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up_c3_to_c2 = UpsamplingLayer(channels, channels)
        self.up_c2_to_c1 = UpsamplingLayer(channels, channels)
        self.up_c1_to_out = UpsamplingLayer(channels, channels)
        self.up_aux = UpsamplingLayer(channels, channels)
        self.fuse_c2 = ScaleFusionBlock(channels)
        self.fuse_c1 = ScaleFusionBlock(channels)

    @staticmethod
    def _prototype_context(prototypes: Tensor) -> Tensor:
        if prototypes.numel() == 0:
            return prototypes.new_zeros((prototypes.shape[0], prototypes.shape[-1]))
        return prototypes.mean(dim=1)

    def forward(
        self,
        *,
        q3: Tensor,
        q2: Tensor,
        q1: Tensor,
        prototype_embeddings: Tensor,
        hq_prototypes: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if len(hq_prototypes) < 2:
            raise ValueError("ScaleAwareQueryAggregator expects l1/l2 prototype tensors")

        context_base = self._prototype_context(prototype_embeddings)
        context_l2 = 0.5 * (context_base + self._prototype_context(hq_prototypes[1]))
        context_l1 = 0.5 * (context_base + self._prototype_context(hq_prototypes[0]))

        q2 = self.fuse_c2(self.up_c3_to_c2(q3), q2, context_l2)
        q1 = self.fuse_c1(self.up_c2_to_c1(q2), q1, context_l1)

        q_out = self.up_c1_to_out(q1)
        q_aux = self.up_aux(q1)
        return q_out, q_aux
