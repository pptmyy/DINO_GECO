from contextlib import nullcontext
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3.vision_transformer import DinoVisionTransformer
from src.models.position_encoding import PositionEmbeddingSine
from src.models.transformer import Attention


def _make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                _make_group_norm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )

        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                _make_group_norm(2 * inplanes),
            ]
        )

        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                _make_group_norm(4 * inplanes),
            ]
        )

        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                _make_group_norm(4 * inplanes),
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        输入:
            x: [B, 3, H, W]

        输出:
            c1: [B, conv_inplane,     H/4,  W/4]
            c2: [B, conv_inplane*2,   H/8,  W/8]
            c3: [B, conv_inplane*4,   H/16, W/16]
            c4: [B, conv_inplane*4,   H/32, W/32]
        """
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        return c1, c2, c3, c4


class MutualAwareTokenAdapter(nn.Module):
    """Bidirectional image/exemplar interaction before ViT tokens become FPN maps."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.image_to_proto = Attention(embed_dim, num_heads)
        self.proto_to_image = Attention(embed_dim, num_heads)
        self.image_norm = nn.LayerNorm(embed_dim)
        self.proto_norm = nn.LayerNorm(embed_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _box_tokens(
        patch_tokens: torch.Tensor,
        bboxes: torch.Tensor | None,
        grid_h: int,
        grid_w: int,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        bsz, _, channels = patch_tokens.shape
        if bboxes is None or bboxes.numel() == 0:
            return patch_tokens.mean(dim=1, keepdim=True)

        token_map = patch_tokens.reshape(bsz, grid_h, grid_w, channels)
        bboxes = bboxes.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        tokens_per_batch = []
        for batch_idx in range(bsz):
            sample_tokens = []
            for box in bboxes[batch_idx]:
                if bool((box <= 0).all()):
                    sample_tokens.append(token_map[batch_idx].reshape(-1, channels).mean(dim=0))
                    continue
                x1, y1, x2, y2 = box
                ix1 = int(torch.floor(x1 * grid_w / max(float(image_w), 1.0)).item())
                ix2 = int(torch.ceil(x2 * grid_w / max(float(image_w), 1.0)).item())
                iy1 = int(torch.floor(y1 * grid_h / max(float(image_h), 1.0)).item())
                iy2 = int(torch.ceil(y2 * grid_h / max(float(image_h), 1.0)).item())
                ix1 = max(0, min(ix1, grid_w - 1))
                ix2 = max(ix1 + 1, min(ix2, grid_w))
                iy1 = max(0, min(iy1, grid_h - 1))
                iy2 = max(iy1 + 1, min(iy2, grid_h))
                region = token_map[batch_idx, iy1:iy2, ix1:ix2].reshape(-1, channels)
                sample_tokens.append(region.mean(dim=0))
            if sample_tokens:
                tokens_per_batch.append(torch.stack(sample_tokens, dim=0))
            else:
                tokens_per_batch.append(token_map[batch_idx].reshape(-1, channels).mean(dim=0)[None])
        return torch.stack(tokens_per_batch, dim=0)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        bboxes: torch.Tensor | None,
        grid_h: int,
        grid_w: int,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        proto_tokens = self._box_tokens(
            patch_tokens,
            bboxes,
            grid_h,
            grid_w,
            image_h,
            image_w,
        )

        proto_tokens = self.proto_norm(
            proto_tokens
            + self.residual_scale * self.proto_to_image(proto_tokens, patch_tokens, patch_tokens)
        )
        patch_tokens = self.image_norm(
            patch_tokens
            + self.residual_scale * self.image_to_proto(patch_tokens, proto_tokens, proto_tokens)
        )
        return patch_tokens


class DINOv3Adapter(nn.Module):
    def __init__(
        self,
        image_size,
        model_size,
        patch_size,
        out_feature_indexes,
        freeze_encoder=True,
        pretrained_weights=None,
        conv_inplane=16,
        mutual_adapter_layers=1,
    ):
        super().__init__()

        self.out_feature_indexes = out_feature_indexes
        self.image_size = image_size
        self.patch_size = patch_size

        dinov3_configs = {
            "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "ffn_ratio": 4},
            "base":  {"embed_dim": 768, "depth": 12, "num_heads": 12, "ffn_ratio": 4},
            "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "ffn_ratio": 4},
        }

        if model_size not in dinov3_configs:
            raise ValueError(
                f"Unsupported DINOv3 size: {model_size}. "
                f"Available sizes: {list(dinov3_configs.keys())}"
            )

        self.dinov3 = DinoVisionTransformer(
            image_size=self.image_size,
            patch_size=self.patch_size,
            **dinov3_configs[model_size],
        )

        embed_dim = self.dinov3.embed_dim
        hidden_dim = self.dinov3.embed_dim

        if pretrained_weights is not None:
            print(f"Loading pretrained weights from {pretrained_weights}")
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.dinov3.load_state_dict(state_dict, strict=False)

        if freeze_encoder:
            for param in self.dinov3.parameters():
                param.requires_grad = False

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)

        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        ])

        self.norms = nn.ModuleList([
            _make_group_norm(hidden_dim),
            _make_group_norm(hidden_dim),
            _make_group_norm(hidden_dim),
            _make_group_norm(hidden_dim),
        ])


        self.pos_encode = PositionEmbeddingSine(num_pos_feats=embed_dim)
        num_heads = dinov3_configs[model_size]["num_heads"]
        self.mutual_adapters = nn.ModuleList(
            [
                MutualAwareTokenAdapter(
                    embed_dim,
                    num_heads,
                )
                for _ in range(int(mutual_adapter_layers))
            ]
        )


    def forward(
        self,
        x: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) :
        """
        输入:
            x: 普通图像张量
               [B, 3, H, W]

        输出:
            c1, c2, c3, c4，均为普通 torch.Tensor

            c1: [B, embed_dim, H/4,  W/4]
            c2: [B, embed_dim, H/8,  W/8]
            c3: [B, embed_dim, H/16, W/16]
            c4: [B, embed_dim, H/32, W/32]
        """

        if x.dim() != 4:
            raise ValueError(
                f"DINOv3Adapter expects a 4D tensor [B, 3, H, W], "
                f"but got shape {tuple(x.shape)}"
            )

        if x.shape[1] != 3:
            raise ValueError(
                f"DINOv3Adapter expects RGB input with 3 channels, "
                f"but got {x.shape[1]} channels"
            )

        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        bs, C, h, w = x.shape

        # DINOv3 主干直接接收普通 Tensor
        dinov3_context = (
            torch.no_grad()
            if not any(param.requires_grad for param in self.dinov3.parameters())
            else nullcontext()
        )
        with dinov3_context:
            all_layers = self.dinov3.get_intermediate_layers(
                x,
                n=self.out_feature_indexes,
                return_class_token=True,
            )

        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        class_tokens = []
        num_scales = len(all_layers) - 2

        for i, sem_feat in enumerate(all_layers):
            feat, class_token = sem_feat
            for adapter in self.mutual_adapters:
                feat = adapter(feat, bboxes, H_c, W_c, x.shape[2], x.shape[3])
            class_tokens.append(class_token)

            # ViT token -> feature map
            # feat: [B, N, C] -> [B, C, H/16, W/16]
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()

            resize_H = int(H_c * 2 ** (num_scales - i))
            resize_W = int(W_c * 2 ** (num_scales - i))

            sem_feat = F.interpolate(
                sem_feat,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )

            sem_feats.append(sem_feat)

        # detail 分支直接接收普通 Tensor
        detail_feats = self.sta(x)

        fused_feats = []
        for sem_feat, detail_feat in zip(sem_feats, detail_feats):
            fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))

        c1 = self.norms[0](self.convs[0](fused_feats[0]))
        c2 = self.norms[1](self.convs[1](fused_feats[1]))
        c3 = self.norms[2](self.convs[2](fused_feats[2]))
        c4 = self.norms[3](self.convs[3](fused_feats[3]))

        c1_pos_encode = self.pos_encode(c1)
        c2_pos_encode = self.pos_encode(c2)
        c3_pos_encode = self.pos_encode(c3)

        backbone_fpn = [c1,c2,c3]

        vision_features = c3
        
        vision_pos_enc = [c1_pos_encode,c2_pos_encode,c3_pos_encode]



        return {
            'vision_features':vision_features,
            'backbone_fpn':backbone_fpn,
            'vision_pos_enc':vision_pos_enc,
            'semantic_anchor': class_tokens[-1] if class_tokens else c4.flatten(2).mean(dim=-1),
        }
