import math
from typing import Tuple, Type

import torch
from torch import Tensor
from torch import nn


class Attention(nn.Module):

    def __init__(
        self,
        embedding_dim: int, 
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads

        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim,self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim,self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim,self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim,embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:

        b,n,c = x.shape

        x = x.reshape(b,n,num_heads,c//num_heads)

        return x.transpose(1,2)
    
    def _recombine_heads(self, x: Tensor) -> Tensor:

        b, n_heads, n_tokens, c_per_head = x.shape

        x = x.transpose(1,2)

        return x.reshape(b,n_tokens,n_heads*c_per_head)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class PrototypeAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        """
        """
        super().__init__()
        self.cross_attention =  Attention(embedding_dim, num_heads)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_f: Tensor, prototypes: Tensor,
    ) -> Tensor:

        image_f = image_f + self.cross_attention(q=image_f,
                                          k=prototypes,
                                          v=prototypes)
        image_f = self.norm(image_f)
        return image_f

class SelfCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim:int,
        num_heads:int 
    ):
        super().__init__()

        self.self_attention = Attention(embedding_dim,num_heads)
        self.cross_attention = Attention(embedding_dim,num_heads)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_f: Tensor, adapted_image_f: Tensor, pos_enc: Tensor,
    ) -> Tensor:
        adapted_image_f  = adapted_image_f+ self.self_attention(q=adapted_image_f+pos_enc,
                                         k=adapted_image_f+pos_enc,
                                         v=adapted_image_f+pos_enc)
        adapted_image_f = self.norm1(adapted_image_f)
        adapted_image_f = adapted_image_f + self.cross_attention(q=adapted_image_f+pos_enc,
                                          k=image_f+pos_enc,
                                          v=image_f+pos_enc)
        adapted_image_f = self.norm2(adapted_image_f)
        return adapted_image_f


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn = Attention(
            embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:


        queries = self.self_attn(q=queries, k=queries, v=queries)
        queries = self.norm1(queries)

        # Cross attention block
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        return queries, keys
