import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        return self.dropout(self.pos_encoding[t].squeeze(1))

class EmbedTime(nn.Module):
    def __init__(self, out_c, dim: int = 256):
        super().__init__()
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, out_c),
        )

    def forward(self, x, t):
        emb = self.emb_layer(t)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb

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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        pooled_out = self.pool(conv_out)
        return conv_out, pooled_out

class Bottleneck(nn.Module):
    def __init__(self, channels, num_convs=4):
        super().__init__()
        layers = []
        in_c = channels
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_c, in_c, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.convs(x)
        x = self.up(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetWithAttention(nn.Module):
    def __init__(self,noise_steps: int = 1000, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)

        """ Define UNet depth """
        self.in_channels = 6
        self.base_channels = 64
        self.depth = 2
        self.levels = 4

        """ Set Down blocks """
        self.encoders = nn.ModuleList()
        self.time_embeds_down = nn.ModuleList()
        self.attns_down = nn.ModuleList()

        in_c = self.in_channels
        for i in range(self.depth):
            out_c = self.base_channels * (2 ** i)
            self.encoders.append(EncoderBlock(in_c, out_c))
            self.time_embeds_down.append(EmbedTime(out_c))
            if i < self.depth - 1:  # No attention on the last downsample
                self.attns_down.append(TransformerEncoderSA(out_c))
            in_c = out_c

        """ Create Bottleneck """
        #self.bottleneck = Bottleneck(self.base_channels * 2**(self.levels-1), num_convs=6)
        self.bottleneck = ConvBlock(in_c, in_c * 2)
        bottleneck_out_c = in_c * 2

        """ setUp blocks """
        self.decoders = nn.ModuleList()
        self.time_embeds_up = nn.ModuleList()
        self.attns_up = nn.ModuleList()

        in_c = bottleneck_out_c
        for i in reversed(range(self.depth)):
            out_c = self.base_channels * (2 ** i)
            self.decoders.append(DecoderBlock(in_c, out_c))
            self.time_embeds_up.append(EmbedTime(out_c))
            if i != 0:  # No attention at last decoder block
                self.attns_up.append(TransformerEncoderSA(out_c))
            in_c = out_c

        """ Final output layer """
        self.final_conv = nn.Conv2d(self.base_channels, self.in_channels, kernel_size=1)

    def forward(self, x, t: torch.LongTensor):
        t = self.pos_encoding(t)
        skips = []
        """ Encoder """
        for i, encoder in enumerate(self.encoders):
            s, x = encoder(x)
            skips.append(s)
            x = self.time_embeds_down[i](x, t)
            if i < len(self.attns_down):
                x = self.attns_down[i](x)
        """ Bottleneck """
        x = self.bottleneck(x)

        """ Decoder """
        for i, decoder in enumerate(self.decoders):
            skip = skips.pop()
            x = decoder(x, skip)
            x = self.time_embeds_up[i](x, t)
            if i < len(self.attns_up):
                x = self.attns_up[i](x)
        """ Output """
        return self.final_conv(x)


    
