import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import Transformer


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
        return self.dropout(self.pos_encoding[t.squeeze(-1)])


class EmbedTime(nn.Module):
    def __init__(self, out_c, dim: int = 256):
        super().__init__()
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(dim, out_c))

    def forward(self, x, t):
        emb = self.emb_layer(t).view(t.size(0), -1, 1, 1)
        return x + emb


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2):
        super().__init__()
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ))
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(num_channels)
        self.ffn = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        x_ln = self.ln(x)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attn
        x = x + self.ffn(x)
        return x.permute(0, 2, 1).view(B, C, H, W)


class EncoderStage(nn.Module):
    def __init__(self, in_c, out_c, num_blocks):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, num_blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderStage(nn.Module):
    def __init__(self, in_c, out_c, num_blocks):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c, num_blocks)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetWithTransformer(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, out_channels=3, base_channels=64, blocks=[2, 3, 3, 4],
                 time_dim=256, noise_steps=1000, image_size=128):
        super().__init__()
        self.depth = len(blocks)
        self.pos_encoding = PositionalEncoding(time_dim, max_len=noise_steps)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.time_embeds_down = nn.ModuleList()
        self.time_embeds_up = nn.ModuleList()
        self.attns_down = nn.ModuleList()
        self.attns_up = nn.ModuleList()

        in_c = in_channels + cond_channels
        channels = []

        for i, num_blocks in enumerate(blocks):
            out_c = base_channels * (2 ** i)
            self.encoders.append(EncoderStage(in_c, out_c, num_blocks))
            self.time_embeds_down.append(EmbedTime(out_c, time_dim))
            if i < self.depth - 1:
                self.attns_down.append(TransformerEncoderSA(out_c))
            channels.append(out_c)
            in_c = out_c

        self.dit_channels = channels[-1]
        self.dit = Transformer.Transformer_S_2(
            input_size=image_size // (2 ** self.depth),
            in_channels=self.dit_channels,
            learn_sigma=False
        )
        self.dit_proj_in = nn.Conv2d(self.dit_channels, self.dit.in_channels, 1)
        self.dit_proj_out = nn.Conv2d(self.dit.out_channels, self.dit_channels, 1)

        for i in reversed(range(self.depth - 1)):
            self.decoders.append(DecoderStage(channels[i + 1], channels[i], blocks[i]))
            self.time_embeds_up.append(EmbedTime(channels[i], time_dim))
            if i != 0:
                self.attns_up.append(TransformerEncoderSA(channels[i]))

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x, t_raw: torch.LongTensor, c):
        t = self.pos_encoding(t_raw)
        x = torch.cat([c, x], dim=1)

        skips = []
        for i, encoder in enumerate(self.encoders):
            s, x = encoder(x)
            skips.append(s)
            x = self.time_embeds_down[i](x, t)
            if i < len(self.attns_down):
                x = self.attns_down[i](x)

        x = self.dit_proj_in(x)
        x = self.dit(x, t=t_raw, y=c)
        x = self.dit_proj_out(x)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            x = decoder(x, skip)
            x = self.time_embeds_up[i](x, t)
            if i < len(self.attns_up):
                x = self.attns_up[i](x)

        return self.final_conv(x)
