import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
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
            nn.Linear(in_features=dim, out_features=out_c),
        )

    def forward(self, x, t):
        emb = self.emb_layer(t)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb

class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4):
        super(TransformerEncoderSA, self).__init__()
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
        x = x.view(B, C, H * W).permute(0, 2, 1)  # Reshape to [B, H*W, C]
        x_ln = self.ln(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(B, C, H, W)  # Reshape back

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

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        return self.conv(x)

class UNetWithAttention(nn.Module):
    def __init__(self, noise_steps: int = 1000, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)

        """ Encoder """
        self.e1 = EncoderBlock(6, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        self.te1 = EmbedTime(64)
        self.te2 = EmbedTime(128)
        self.te3 = EmbedTime(256)
        self.te4 = EmbedTime(512)

        self.attn1 = TransformerEncoderSA(64)
        self.attn2 = TransformerEncoderSA(128)
        self.attn3 = TransformerEncoderSA(256)

        """ Bottleneck """
        self.bottleneck = ConvBlock(512, 1024)

        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        self.teU1 = EmbedTime(512)
        self.teU2 = EmbedTime(256)
        self.teU3 = EmbedTime(128)
        self.teU4 = EmbedTime(64)

        self.attnU1 = TransformerEncoderSA(256)
        self.attnU2 = TransformerEncoderSA(128)
        self.attnU3 = TransformerEncoderSA(64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 6, kernel_size=1)

    def forward(self, x, t: torch.LongTensor):
        t = self.pos_encoding(t)

        """ Encoder """
        s1, p1 = self.e1(x)
        p1 = self.te1(p1, t)
        p1 = self.attn1(p1)

        s2, p2 = self.e2(p1)
        p2 = self.te2(p2, t)
        p2 = self.attn2(p2)

        s3, p3 = self.e3(p2)
        p3 = self.te3(p3, t)
        p3 = self.attn3(p3)

        s4, p4 = self.e4(p3)
        p4 = self.te4(p4, t)

        """ Bottleneck """
        b = self.bottleneck(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d1 = self.teU1(d1, t)

        d2 = self.d2(d1, s3)
        d2 = self.teU2(d2, t)
        d2 = self.attnU1(d2)

        d3 = self.d3(d2, s2)
        d3 = self.teU3(d3, t)
        d3 = self.attnU2(d3)

        d4 = self.d4(d3, s1)
        d4 = self.teU4(d4, t)
        d4 = self.attnU3(d4)

        """ Output """
        return self.outputs(d4)
