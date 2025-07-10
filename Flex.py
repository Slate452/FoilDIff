import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import Transformer 

# All rights reserved.
# This source code is licensed under the license found in thewher 
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# FLEX: arXiv:2505.17351v1 [cs.LG] 23 May 2025
# --------------------------------------------------------



class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        return self.dropout(self.pos_encoding[t].unsqueeze(1))


class EmbedTime(nn.Module):
    def __init__(self, out_c, dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, out_c))

    def forward(self, x, t):
        emb = self.mlp(t).view(t.size(0), -1, 1, 1)
        return x + emb


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        blocks = [ConvBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            blocks.append(ConvBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.blocks(x)
        return x, self.pool(x)


class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(num_channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        x_ln = self.ln(x)
        attn_output, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x = attn_output + x
        x = self.ff_self(x) + x
        return x.permute(0, 2, 1).view(B, C, H, W)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if use_attention:
            self.attn = TransformerEncoderSA(out_channels * 2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        if self.use_attention:
            x = self.attn(x)
        return self.conv(x)


class TaskEncoder(nn.Module):
    def __init__(self, in_channels, channel_list, block_counts):
        super().__init__()
        self.blocks = nn.ModuleList()
        for out_channels, num_blocks in zip(channel_list, block_counts):
            self.blocks.append(EncoderStage(in_channels, out_channels, num_blocks))
            in_channels = out_channels

    def forward(self, x):
        skips = []
        for block in self.blocks:
            skip, x = block(x)
            skips.append(skip)
        return x, skips


class FoilFlex(nn.Module):
    def __init__(
                 self, in_channels=3,
                 cond_channels=3, 
                 out_channels=3, 
                 channels=[64, 128, 256, 512], 
                 blocks=[2, 3, 3, 4], 
                 time_dim=256, 
                 noise_steps=1000, 
                 image_size=128, 
                 use_attention=True):
        
        super().__init__()
        self.depth = len(channels)
        self.pos_encoding = PositionalEncoding(time_dim, max_len=noise_steps)

        self.task_encoder = TaskEncoder(cond_channels, channels, blocks)
        self.common_encoder = TaskEncoder(in_channels, channels, blocks)

        self.time_embeds_down = nn.ModuleList([EmbedTime(ch, time_dim) for ch in channels])
        self.time_embeds_up = nn.ModuleList([EmbedTime(ch, time_dim) for ch in reversed(channels)])

        self.decoders = nn.ModuleList([
            DecoderBlock(channels[i + 1], channels[i], use_attention=use_attention)
            for i in reversed(range(self.depth - 1))
        ])

        self.dit_channels = channels[-1]
        self.dit = Transformer.Transformer_B_2(
            input_size=image_size // (2 ** self.depth),
            in_channels=self.dit_channels,
            learn_sigma=False
        )
        self.dit_proj_in = nn.Conv2d(self.dit_channels, self.dit.in_channels, 1)
        self.dit_proj_out = nn.Conv2d(self.dit.out_channels, self.dit_channels, 1)

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x, c, t: torch.LongTensor):
        t_embed = self.pos_encoding(t.squeeze(-1))  # (B, D)

        task_bottleneck, task_skips = self.task_encoder(c)

        common_skips = []
        z = x
        for i, block in enumerate(self.common_encoder.blocks):
            weak_input = z + task_skips[i] if i == 0 else z
            s_common, z = block(weak_input)
            z = self.time_embeds_down[i](z, t_embed)
            common_skips.append(s_common)

        z = self.dit_proj_in(z)
        z = self.dit(z, t=t, y=c)
        z = self.dit_proj_out(z)

        for i, decoder in enumerate(self.decoders):
            s_task = task_skips[-(i + 2)]  # account for bottleneck not used in skip
            s_common = common_skips[-(i + 2)]
            s_combined = torch.cat([s_task, s_common], dim=1)
            z = decoder(z, s_combined)
            z = self.time_embeds_up[i](z, t_embed)

        return self.final_conv(z)
