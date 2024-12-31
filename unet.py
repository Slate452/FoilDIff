import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            dropout: float = 0.1,
            max_len: int = 1000,
            apply_dropout: bool = True,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding

class embed_time(nn.Module):
    def __init__(self, out_c,dim:int =256):
        super().__init__()
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=dim, out_features=out_c),
            )
    def forward(self, x, t, r:bool = False)->torch.tensor:
        emb = self.emb_layer(t)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        if r == True:
            print(x. shape,"\n", emb.shape, "\n" )#, (x+emb).shape )
        else:
            return x + emb    
  
class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, size: int, num_heads: int = 4):
        super(TransformerEncoderSA, self).__init__()
        self.num_channels = num_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([num_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_channels, self.size * self.size).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(-1, self.num_channels, self.size, self.size)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_blck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x , p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class build_unet(nn.Module):
    def __init__(self,
                noise_steps: int = 1000,
                time_dim: int = 256,):
        super().__init__()
        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)
        
        """ Encoder """
        self.e1 = encoder_blck(6, 64)
        self.e2 = encoder_blck(64, 128)
        self.e3 = encoder_blck(128, 256)
        self.e4 = encoder_blck(256, 512)
        self.attn1 =TransformerEncoderSA(64, 64)
        self.attn2 = TransformerEncoderSA(128, 32)
        self.attn3 = TransformerEncoderSA(256, 16)
        self.te1 = embed_time(64)
        self.te2 = embed_time(128)
        self.te3 = embed_time(256)
        self.te4 = embed_time(512)

        """ Bottleneck """
        self.b = conv_block(128, 256)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.attnU1 =TransformerEncoderSA(256, 32)
        self.attnU2 = TransformerEncoderSA(128, 64)
        self.attnU3 = TransformerEncoderSA(64, 128)
        self.teU1 = embed_time(512)
        self.teU2 = embed_time(256)
        self.teU3 = embed_time(128)
        self.teU4 = embed_time(64)


        """ Classifier """
        self.outputs = nn.Conv2d(64, 6, kernel_size=1, padding=0)

    def forward(self, inputs,t: torch.LongTensor):
        t = self.pos_encoding(t)
        """ Encoder """
        '''Frist Layer'''
       # print(inputs.shape)
        s1, p1 = self.e1(inputs)
        p1 = self.te1(p1,t)
        p1= self.attn1(p1)
     #   print(p1.shape,s1.shape)
        '''Second Layer'''
        s2, p2 = self.e2(p1) 
        p2 = self.te2(p2,t)
        p2= self.attn2(p2)
      #  print(p2.shape)
        """ Bottleneck """
        b = self.b(p2)
        """
        '''Third Layer'''
        s3, p3 = self.e3(p2)
        p3 = self.te3(p3,t)
        p3= self.attn3(p3)
        #print(p3.shape)
        '''Fourth Layer'''
        s4, p4 = self.e4(p3)
        p4 = self.te4(p4,t)
        #print(p4.shape)
        #s4 = self.b
        #print(b.shape, s4.shape)
        '''Fourth Layer'''
        d1 = self.d1(b, s4)
        d1 = self.teU1(d1,t)
        #print(s4.shape, d1.shape)
        '''Third Layer'''
        d2 = self.d2(d1, s3)
        d2 = self.teU2(d2,t)
        d2 = self.attnU1(d2)
        #print(d2.shape)
        #print( s3.shape, d2.shape)
        """

        """ Decoder """
        '''Second Layer'''
        d3 = self.d3(b, s2)
        d3 = self.teU3(d3,t)
        d3 = self.attnU2(d3)
        #print("layer 3",s2.shape, d3.shape)
        '''Frist Layer'''
        #print("layer 4", s1.shape)
        d4 = self.d4(d3, s1)
        #print("layer 4", s1.shape, d4.shape)
        d4 = self.teU4(d4,t)
        d4 = self.attnU3(d4)
#       print("layer 4", s1.shape, d4.shape)

        """Output """
        output = self.outputs(d4)
        #print(output.shape)

        return output

