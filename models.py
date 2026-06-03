# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
from diffusion.DiT_blocks import TimestepEmbedder
import random

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # print("DiTBlock ", x.shape, y.shape)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PatchEmbed(nn.Module):
    """ 1D "image" to Patch Embedding
    """
    def __init__(self, sequence_size=224, patch_size=1, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (sequence_size // patch_size)

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class YOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv1d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv1d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv1d(in_features, in_features, kernel_size=1, )

        self.initialize_weights()
    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

    def initialize_weights(self):
        nn.init.constant_(self.conv1.weight, 0)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.weight, 0)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.projector.weight, 0)
        nn.init.constant_(self.projector.bias, 0)

class YEmbed(nn.Module):
    """ 1D "image" to Patch Embedding
    """
    def __init__(self, sequence_size=224, patch_size=1, in_chans=1, embed_dim=768, uncond_prob=None):
        super().__init__()
        num_patches = (sequence_size // patch_size)


        self.num_patches = num_patches
        self.uncond_prob = uncond_prob

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.proj1 = nn.Linear(embed_dim, 64)
        self.yop = YOp(64)
        self.proj2 = nn.Linear(64, embed_dim)

        self.nonlinear = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self.norm = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.ones(embed_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(embed_dim))
        self.initialize_weights()

        self.register_buffer("y_embedding", nn.Parameter(torch.randn(in_chans, sequence_size) / sequence_size ** 0.5))

    def token_drop(self, x, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(x.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        x = torch.where(drop_ids[:, None, None], self.y_embedding, x)
        return x

    def forward(self, x, train, force_drop_ids=None):
        B, C, L = x.shape
        if train:
            assert x.shape[1:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            x = self.token_drop(x, force_drop_ids)


        x = self.proj(x).transpose(1, 2)
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        proj1 = self.proj1(x).transpose(1, 2)
        proj1 = self.yop(proj1).transpose(1, 2)
        nonlinear = self.nonlinear(proj1)
        nonlinear = self.dropout(nonlinear)
        proj2 = self.proj2(nonlinear)

        return proj2 + identity

    def initialize_weights(self):
        nn.init.constant_(self.proj1.weight, 0)
        nn.init.constant_(self.proj1.bias, 0)
        nn.init.constant_(self.proj2.weight, 0)
        nn.init.constant_(self.proj2.bias, 0)



class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=96,
        patch_size=1,
        in_channels=1,
        exo_channels=10,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.exo_embedder = YEmbed(input_size, patch_size, exo_channels, hidden_size, class_dropout_prob)


        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv1d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # exo patched embedder
        w = self.exo_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.exo_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)



    def forward(self, x, t, exo_c=None):
        """
        Forward pass of DiT.
        x: (N, C, L) tensor of input Sequence
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D),
        t = self.t_embedder(t)                   # (N, D)
        c = t
        exo_c = self.exo_embedder(exo_c, self.training) if exo_c is not None else 0
        x = x + exo_c
        x += c.unsqueeze(1).repeat(1, x.shape[1], 1)  # (N, T, D)


        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size * out_channels)
        x = x.transpose(1, 2)                     # (N, patch_size * out_channels, T)
        return x

    def forward_with_cfg(self, x, t, cfg_scale, y=None, mask=None, exo_c=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, exo_c=exo_c)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def load_pretrained(self, pretrained_state_dict):
        model_state_dict = self.state_dict()
        #
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_state_dict)
        self.load_state_dict(model_state_dict)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, seq_len):
    """
    embed_dim: output dimension for each position
    seq_len: length of input sequence
    out: (L, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(seq_len, dtype=np.float64)  # (L,)
    out = np.einsum('l,d->ld', pos, omega)  # (L, D/2), outer product

    emb_sin = np.sin(out) # (L, D/2)
    emb_cos = np.cos(out) # (L, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (L, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_B_1(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)


def DiT_S_1(**kwargs):
    return DiT(depth=3, hidden_size=768, patch_size=1, num_heads=12, **kwargs)



DiT_models = {
    'DiT-B/1':  DiT_B_1,
    'DiT-S/1':  DiT_S_1,
}
