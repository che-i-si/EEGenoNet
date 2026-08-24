import math
import torch
from torch import nn
from einops import rearrange
from typing import Sequence, Literal

# %% BASE
def get_sincos_pos_embed(dim: int, seq_len: int, cls_token: bool = False):
    if cls_token:
        pe = torch.zeros(seq_len + 1, dim)
        position = torch.arange(0, seq_len + 1, dtype=torch.float).unsqueeze(1)
    else:
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    return pe


class Transpose(nn.Module):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.dims)


class Interpolation(nn.Module):
    def __init__(self, size: Sequence[int],
                 mode:Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact']):
        super(Interpolation, self).__init__()
        self.size = tuple([*size])
        self.mode = mode
    def forward(self, x):
        return nn.functional.interpolate(x, self.size, mode=self.mode)


class GELUTanh(nn.Module):
    def __init__(self):
        super(GELUTanh, self).__init__()
    def forward(self, x):
        return torch.where(x < 0, nn.functional.gelu(x), nn.functional.tanh(1.25*x))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.block(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None
        self.value_gradients = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def save_value_gradients(self, value_gradients):
        self.value_gradients = value_gradients

    def get_value_gradients(self):
        return self.value_gradients

    def forward(self, x, register_hook=False):
        b, n, _, h = *x.shape, self.num_heads
        q, k, v = rearrange(self.qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=h)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum("bhij,bhjd->bhid", attn, v)

        if register_hook:
            self.save_attention_map(attn)
            v.register_hook(self.save_value_gradients)
            attn.register_hook(self.save_attn_gradients)

        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False):
        x = x + self.attn(self.norm1(x), register_hook=register_hook)
        x = x + self.mlp(self.norm2(x))
        return x


# %% Biaxial Information Embedding Block
class Embedding(nn.Module):
    def __init__(self, embdim:int, nhead:int, N:int, seq_len:int, dropout:float):
        super().__init__()
        self.embdim = embdim
        self.seq_len = seq_len
        self.N = N

        self.temporal_block = Block(
            dim=self.embdim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout,
            attn_drop=dropout,
        )

        self.spatial_block = Block(
            dim=self.embdim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout,
            attn_drop=dropout,
        )

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len + 1, embdim), requires_grad=False
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.N + 1, self.embdim),
            requires_grad=False,
        )

        self.temporal_token = nn.Parameter(torch.zeros(1, 1, self.embdim))
        self.spatial_token = nn.Parameter(torch.zeros(1, 1, self.embdim))

        self.initialize_weights()

    def initialize_weights(self):
        temporal_pos_embed = get_sincos_pos_embed(
            dim=self.embdim, seq_len=self.seq_len, cls_token=True
        )
        self.temporal_pos_embed.data.copy_(temporal_pos_embed)

        spatial_pos_embed = get_sincos_pos_embed(
            dim=self.embdim,
            seq_len=self.N,
            cls_token=True,
        )
        self.spatial_pos_embed.data.copy_(spatial_pos_embed)

        torch.nn.init.normal_(self.temporal_token, std=0.02)
        torch.nn.init.normal_(self.spatial_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, register_hook=False):
        """

        Parameters
        ----------
        x: torch.Tensor
            shape (B, d, N, T')
        register_hook: bool


        Returns
        -------
        torch.Tensor
            shape (B x (N+1) x d x (T'+1))
        """
        B, d, N, T = x.shape
        # Temporal position embedding & block
        x = x.permute((0, 2, 3, 1)).reshape(B*N, T, d)  # BN x T' x d
        x = x + self.temporal_pos_embed[:, 1:, :]
        token = self.temporal_token + self.temporal_pos_embed[:, :1, :]
        token = token.expand(B * N, -1, -1)             # token: (BN x 1 x d)
        x = torch.cat((token, x), dim=1)        # BN x (1+T') x d
        x = self.temporal_block(x, register_hook=register_hook)
        x = x.reshape(B, N, -1, self.embdim)  # B x N x 1+T' x D
        x = x.transpose(1, 2)  # B x T'+1 x N x D

        # Spatial position embedding & block
        x = x.reshape(B * (1+T), N, d)  # B(1+T) x N x d
        x = x + self.spatial_pos_embed[:, 1:, :]
        token = self.spatial_token + self.spatial_pos_embed[:, :1, :]
        token = token.expand(B * (1+T), -1, -1)         # token: (B(1+T) x 1 x d)
        x = torch.cat((token, x), dim=1)         # B(1+T') x (1+N) x d
        x = self.spatial_block(x, register_hook=register_hook)
        x = x.reshape(B, T+1, N + 1, self.embdim)  # B x T'+1 x N+1 x D
        x = x.permute(0, 2, 3, 1)  # B x (N+1) x d x (T'+1)
        del token

        return x

# %% Former Blocks
class TemporalSpatialEncoder(nn.Module):
    def __init__(self, embdim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.embdim = embdim

        self.temporal_block = Block(
            dim=self.embdim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout,
            attn_drop=dropout,
        )

        self.spatial_block = Block(
            dim=self.embdim,
            num_heads=nhead,
            mlp_ratio=1.0,
            qkv_bias=False,
            drop=dropout,
            attn_drop=dropout,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, register_hook=False):
        """

        Parameters
        ----------
        x: torch.Tensor
            shape (B x (N+1) x d x (T'+1))
        register_hook: bool

        Returns
        -------
        torch.Tensor
            shape (B x (N+1) x d x (T'+1))
        """
        B, N, d, T = x.shape        # B x (N+1) x d x (T'+1)

        # Temporal Block
        x = x.reshape(B * N, d, T)  # B(N+1) x d x (T'+1)
        x = x.transpose(1, 2)  # B(N+1) x (T'+1) x d
        x = self.temporal_block(x, register_hook=register_hook)
        x = x.reshape(B, N, T, d)  # B x (N+1) x (T'+1) x d
        x = x.transpose(1, 2)  # B x (T'+1) x (N+1) x d

        # Spatial Block
        x = x.reshape(B * T, N, d)  # B(T'+1) x (N+1) x d
        x = self.spatial_block(x, register_hook=register_hook)
        x = x.reshape(B, T, N, d)  # B x (T'+1) x (N+1) x d
        x = x.permute(0, 2, 3, 1)  # B x (N+1) x d x (T'+1)

        return x
