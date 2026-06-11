# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from typing import Any,Optional,Tuple

import numpy as np
import torch 

from torch import nn


def init_x_y(end_x:int,end_y:int):
    #生成网格，0~(end_x*end_y-1)
    t = torch.arange(end_x*end_y,dtype=torch.float32)
    
    t_x = (t % end_x).float()
    
    t_y = torch.div(t,end_x,rounding_mode='floor').float()

    return t_x,t_y

def compute_axial_cis(dim:int,end_x:int,end_y:int,theta:float=10000.0):

    freqs_x = 1.0 / (theta ** (torch.arange(0,dim,4)[: (dim//4)].float() /dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x,t_y = init_x_y(end_x,end_y)
    freqs_x = torch.outer(t_x,freqs_x)
    freqs_y = torch.outer(t_y,freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x),freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y),freqs_y)

    return torch.cat([freqs_cis_x,freqs_cis_y],dim=-1)

def reshape_for_broadcast(freqs_cis:torch.Tensor,x:torch.Tensor):

    ndim = x.ndim

    assert 0 <= 1 <ndim
    assert freqs_cis.shape == (x.shape[-2],x.shape[-1])
    
    shape  = [d if i >= ndim-2 else 1 for i,d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
 

 
class PositionEmbeddingSine(nn.Module):
    """
    基于正弦函数的位置编码模块,是《Attention Is All You Need》中位置编码的改进版本，
    推广到二维图像场景（同时编码 x、y 两个方向）。
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize:bool = True,
        scale: Optional[float]=None,
    ):
        super().__init__()
        
        #入参检测
        assert num_pos_feats % 2 == 0 , "Expecting even model width"

        self.num_pos_feats = num_pos_feats // 2
        #控制不同频率分量的基底,默认为10000
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        
        self.scale = scale
        self.cache = {}
    def _encode_xy(self,x,y):
        """
        对一维坐标序列 x、y 分别计算正弦位置编码。
        输入坐标应已归一化（值域 [0, 1]）。
        返回形状均为 (N, num_pos_feats) 的编码张量。
        """
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        
        x_embed = x*self.scale
        y_embed = y*self.scale

        # 构造频率向量：dim_t[i] = temperature^2(2*(i//2) / num_pos_feats)
        # 偶数维用sin,奇数维用cos

        dim_t = torch.arange(self.num_pos_feats,dtype=torch.float32,device=x.device)
        dim_t = self.temperature ** (2*(dim_t//2)/self.num_pos_feats)

        #将坐标广播到每一个频率维度上,得到(N,num_pos_feats)相位矩阵
        pos_x = x_embed[:,None] / dim_t
        pos_y = y_embed[:,None] / dim_t

        # 偶数列取 sin，奇数列取 cos，交错拼接后展平
        # stack 后形状：(N, num_pos_feats/2, 2) → flatten → (N, num_pos_feats)
        pos_x = torch.stack(
            (pos_x[:,0:2].sin(),pos_x[:,1:2].cos()),dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:,0:2].sin(),pos_y[:,1:2].cos()),dim=2
        ).flatten(1)

        return pos_x,pos_y

    def encode_boxes(self,x,y,w,h):

        """
        对检测框进行位置编码
        输入:归一化的中心点坐标(x,y)和宽高(w,h)形状为(N,)
        输出:(N,2*num_pos_feats+2)的编码向量
        """
        pos_x,pos_y = self._encode_xy(x,y)

        pos = torch.cat((pos_y,pos_x,h[:,None],w[:,None]),dim=1)

        return pos
    encode = encode_boxes

    def encode_points(self,x,y,labels):
        """
        对点提示(point prompt)进行位置编码
        输入 归一化x,y以对应labels,形状(B,N)
        输出 (B,N,2*num_pos_feat+1)的编码张量,最后一维附加了原始label标量
        """

        (bx,nx),(by,ny),(bl,nl) = x.shape,y.shape,labels.shape

        assert bx == by and nx == ny and bx == bl and nx == nl

        #将batch 和 点数展平后统一编码，恢复batch
        pos_x,pos_y = self._encode_xy(x.flatten(),y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)

        # 拼接 y 编码、x 编码与 label，label 直接作为最后一维附加
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos 
   
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        为输入特征图生成二维正弦位置编码。
        输入:x 形状为 (B, C, H, W) 的特征图（仅用其空间尺寸）。
        输出:形状为 (B, 2*num_pos_feats, H, W) 的位置编码张量。
        """
        # 以 (H, W) 为 key 缓存位置编码 相同尺寸无需重复计算
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            # 直接复制缓存结果到当前 batch
            return self.cache[cache_key].to(device=x.device, dtype=x.dtype)[None].repeat(x.shape[0], 1, 1, 1)

        # 生成 y 方向坐标网格：值域 [1, H] 形状 (B, H, W)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        # 生成 x 方向坐标网格：值域 [1, W] 形状 (B, H, W)
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6  # 防止除零
            # 用最大坐标值归一化 再缩放到 [0, scale]
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 构造频率向量，形状 (num_pos_feats,)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 将坐标网格广播到频率维度 形状 (B, H, W, num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # 偶数维取 sin 奇数维取cos 交错拼接后展平频率维度
        # stack 后：(B, H, W, num_pos_feats/2, 2) → flatten(3) → (B, H, W, num_pos_feats)
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # 拼接 y 和 x 编码，并调整维度顺序为 (B, C, H, W)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # 缓存第一个样本的编码（batch 内各样本编码相同）
        self.cache[cache_key] = pos[0]
        return pos
    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
