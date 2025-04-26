# Our model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NO TE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class AffectTransformer(AffectTransformer): 
    def __init__(self, max_num_frame=5000, num_classes=7, global_pool='token', meta_dim=248, meta_input_dim=4, use_attn=True):
        super().__init__(max_num_frame=max_num_frame, num_classes=num_classes, global_pool=global_pool) 
        self.fc = nn.Linear(self.embed_dim + meta_dim, num_classes)
        self.embedding_m = nn.Linear(meta_input_dim, meta_dim)
        self.attention = nn.Linear(meta_input_dim, self.embed_dim)
        self.layernorm = nn.LayerNorm(self.embed_dim)

        self.use_attn = use_attn
        if use_attn:
            print(self.use_attn)
            self.scale_cross = self.embed_dim ** -0.5
            self.kv_cross = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=True)
            self.q_cross = nn.Linear(meta_dim, self.embed_dim, bias=True)
            self.attn_drop_cross = nn.Dropout(0.)
            self.proj_cross = nn.Linear(self.embed_dim, self.embed_dim)
            self.proj_drop_cross = nn.Dropout(0.)
            self.norm_cross = nn.LayerNorm(self.embed_dim)
            mlp_hidden_dim = int(self.embed_dim * 2)
            self.mlp_cross = Mlp(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=GELU, drop=0)

    def forward(self, face, affect, meta):
        meta = self.embedding_m(meta)

        affect = self.affect_compress(affect)
        x = torch.cat([face, affect], dim=2) # [b, f, d]

        b, _, _ = x.shape
        x = self.forward_features(x)

        # attention
        if self.use_attn:
            B, N, C = x.shape
            kv = self.kv_cross(x).reshape(B, N, 2, C).permute(2, 0, 1, 3)
            k, v = kv[0], kv[1]
            q = self.q_cross(meta.repeat(1, N, 1)).reshape(B, N, C) # repeat meta for N
            attn = (q @ k.transpose(-2, -1)) * self.scale_cross
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop_cross(attn)
            attn_x = (attn @ v).reshape(B, N, C)
            attn_x = self.proj_cross(attn_x)
            attn = self.proj_drop_cross(attn_x)
            x = x + attn_x
            x = x + self.mlp_cross(self.norm_cross(x))

        # global pool
        x = x[:, self.num_prefix_tokens:, :].mean(dim=1) if self.global_pool == 'avg' else x[:, 0, :]
        x = self.layernorm(x)

        x = self.fc(torch.cat([x, meta], dim=1))

        return x, x

