import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class TimeEncodingBlock(nn.Module):
    def __init__(self, embed_dim, out_channels, activation_function="ReLU"):
        super(TimeEncodingBlock, self).__init__()

        if activation_function == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation_function == "GELU":
            activation = nn.GELU()
        elif activation_function == "SiLU":
            activation = nn.SiLU(inplace=True)
        
        self.embed_dim = embed_dim

        self.linear1 = nn.Linear(embed_dim, out_channels)
        self.activation = activation
        self.linear2 = nn.Linear(out_channels, out_channels)

        self.pos_embed = SinusoidalPositionEmbeddings(embed_dim)
    
    def forward(self, t):
        out = self.pos_embed(t)
        out = self.activation(self.linear1(out))
        out = self.linear2(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels_x, channels_g, out_channels):
        super(AttentionBlock, self).__init__()
        # assert channels_x == out_channels
        self.conv2d_x = nn.Sequential(
                nn.Conv2d(in_channels=channels_x, out_channels=out_channels, kernel_size=(1,1), stride=(1,1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.conv2d_g = nn.Sequential(
                nn.Conv2d(in_channels=channels_g, out_channels=out_channels, kernel_size=(1,1), stride=(1,1)),
                nn.BatchNorm2d(out_channels)
            )
        self.conv2d_psi = nn.Sequential(
                nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=(1,1)),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        self.conv2d_result = nn.Conv2d(out_channels, channels_x, (1,1))
        self.batch_norm = nn.BatchNorm2d(channels_x)

    
    def forward(self, gate, skip_connection):
        x1 = self.conv2d_x(skip_connection)
        g1 = self.conv2d_g(gate)
        psi = F.relu(x1 + g1)

        psi = self.conv2d_psi(psi)

        return skip_connection * psi
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out


class UpConv(nn.Module):
    """
    Not currently used
    """
    def __init__(self, in_channels, out_channels, activation_function="ReLU"):
        super(UpConv, self).__init__()
        if activation_function == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation_function == "GELU":
            activation = nn.GELU()
        elif activation_function == "SiLU":
            activation = nn.SiLU(inplace=True)
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            activation
        )
    def forward(self, x):
        return self.up(x)
    


"""
The Following Classes are to be used outside of this file.
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, activation_function="ReLU"):
        super(ConvBlock, self).__init__()
        
        self.conv2d_0 = self._conv_block(in_channels , out_channels, activation_function)
        self.conv2d_1 = self._conv_block(out_channels, out_channels, activation_function)
        self.conv2d_2 = self._conv_block(out_channels, out_channels, activation_function)
        self.conv2d_3 = self._conv_block(out_channels, out_channels, activation_function)

        self.time_mlp = TimeEncodingBlock(embed_dim  , out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.shortcut_activation = self._choose_activation(activation_function=activation_function)

    def _choose_activation(self, activation_function):
        if activation_function == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation_function == "GELU":
            activation = nn.GELU()
        elif activation_function == "SiLU":
            activation = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError("This activation function is not implemented")
        return activation

    def _conv_block(self, in_channels, out_channels, activation_function):

        activation = self._choose_activation(activation_function)

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )
    
    def forward(self, x, t):
        time_emb = self.time_mlp(t)[:, :, None, None].repeat(1,1, x.shape[-2], x.shape[-1])

        shortcut = self.shortcut(x)

        out = self.conv2d_0(x)
        # out = out + time_emb
        out = self.conv2d_1(out)
        # out = out + time_emb
        out = self.conv2d_2(out)
        # out = out + time_emb
        out = self.conv2d_3(out)
        out = out + time_emb

        out += shortcut
        out = self.shortcut_activation(out)
        # out = torch.relu(out)

        return out

# class UpBlock(nn.Module):
#     """
#     The Upsampling routine condensened into a single block. (I'll be damned if I'm going to write all of these everytime I want to increase complexity)
#     Uses AttentionBlock and ConvBlock
#     """
#     def __init__(self, in_channels, out_channels, time_embed_dim, activation_function="ReLU"):
#         super(UpBlock, self).__init__()
#         self.out_channels = out_channels
#         if in_channels == out_channels:
#             self.mid_channels = out_channels
#         else:
#             self.mid_channels = out_channels//2
#         self.Up = UpConv(in_channels=in_channels, out_channels=out_channels)
#         self.Att = AttentionBlock(channels_x=out_channels, channels_g=out_channels, out_channels=self.mid_channels)
#         self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, embed_dim=time_embed_dim, activation_function=activation_function)

#     def forward(self, gate, skip_connection, t):
#         assert skip_connection.shape[-3] == self.out_channels
#         d = self.Up(gate)
#         s = self.Att(gate=d, skip_connection=skip_connection)
#         d = torch.cat((s, d), dim=1) # TODO: Consider other strategies of combining these two
#         d = self.conv(d, t)
#         return d

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, activation_function="ReLU"):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, time_embed_dim, activation_function)
        self.attention = SelfAttention(out_channels)
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv_block(x, t)
        x = self.attention(x)
        return x

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, activation_function="ReLU"):
        super(BottleNeckBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_embed_dim, activation_function)
        self.attention = SelfAttention(out_channels)

    def forward(self, x, t):
        out = self.conv(x, t)
        out = self.attention(out)
        return out
        


if __name__ == "__main__":

    BATCH_SIZE = 4

    print(f"{'='*20}Upsampling{'='*20}")

    time = torch.randint(0, 1000, (BATCH_SIZE, ), device="cpu")
    gated = torch.randn((BATCH_SIZE, 128,128,128))
    skip_connection = torch.randn((BATCH_SIZE, 64, 256, 256))

    test_UB = UpBlock(128, 64)

    out = test_UB(gated, skip_connection, time)
    print(out.shape)