import torch
import torch.nn as nn
try:
    from .block_layers import BottleNeckBlock, UpBlock, ConvBlock
except:
    from block_layers import BottleNeckBlock, UpBlock, ConvBlock
    
class UNet1(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super(UNet, self).__init__()
        self.MaxPool  = nn.MaxPool2d(kernel_size = 2)

        encoder_activation = "GELU"

        ## Encoders
        self.encoder1 = ConvBlock(in_channels, 64, time_embed_dim, encoder_activation)
        self.encoder2 = ConvBlock(64, 128, time_embed_dim, encoder_activation)
        self.encoder3 = ConvBlock(128, 256, time_embed_dim, encoder_activation)
        self.encoder4 = ConvBlock(256, 512, time_embed_dim, encoder_activation)
        self.encoder5 = ConvBlock(512, 1024, time_embed_dim, encoder_activation)

        ## Decoders
        decoder_activation = "GELU"

        self.decoder5 = UpBlock(1024, 512, time_embed_dim, decoder_activation)
        self.decoder4 = UpBlock(512, 256, time_embed_dim, decoder_activation)
        self.decoder3 = UpBlock(256, 128, time_embed_dim, decoder_activation)
        self.decoder2 = UpBlock(128, 64, time_embed_dim, decoder_activation)
        

        ## Final
        self.final = nn.Conv2d(64, in_channels, kernel_size=1)

        self.tanh = nn.Tanh()

    def forward(self, x, t):

        ## Encoders
        e1 = self.encoder1(x, t)

        e2 = self.MaxPool(e1)
        e2 = self.encoder2(e2, t)

        e3 = self.MaxPool(e2)
        e3 = self.encoder3(e3, t)

        e4 = self.MaxPool(e3)
        e4 = self.encoder4(e4, t)

        e5 = self.MaxPool(e4)
        e5 = self.encoder5(e5, t)

        ## Decoders (gate, skip_connection)
        out = self.decoder5(e5, e4, t)
        out = self.decoder4(out, e3, t)
        out = self.decoder3(out, e2, t)
        out = self.decoder2(out, e1, t)

        out = self.final(out)
        out = self.tanh(out)

        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super(UNet, self).__init__()
        self.MaxPool  = nn.MaxPool2d(kernel_size = 2)

        encoder_activation = "GELU"

        ## Encoders
        self.encoder1 = ConvBlock(in_channels, 64, time_embed_dim, encoder_activation)
        self.encoder2 = ConvBlock(64, 128, time_embed_dim, encoder_activation)

        bottleneck_activation = "SiLU"

        self.bottleneck1 = BottleNeckBlock(128, 256, time_embed_dim, bottleneck_activation)
        self.bottleneck2 = BottleNeckBlock(256, 256, time_embed_dim, bottleneck_activation)

        decoder_activation = "SiLU"
        self.decoder2 = UpBlock(256, 128, time_embed_dim, decoder_activation)
        self.decoder1 = UpBlock(128, 64,  time_embed_dim, decoder_activation)
        
        ## Final
        self.final = nn.Conv2d(64, in_channels, kernel_size=1)
        
        self.tanh = nn.Tanh()

    def forward(self, x, t):
        x1 = self.encoder1(x, t)
        x2 = self.encoder2(self.MaxPool(x1), t)

        b = self.bottleneck1(self.MaxPool(x2), t)
        b = self.bottleneck2(b, t)

        out = self.decoder2(b, x2, t)
        out = self.decoder1(out, x1, t)

        
        out = self.final(out)
        out = self.tanh(out)

        return out



if __name__ == "__main__":
    BATCH_SIZE = 8
    model = UNet(in_channels=3, time_embed_dim=32).cuda()
    print(model)
    random_tensor = torch.randn((BATCH_SIZE, 3, 64, 64)).cuda()
    time = torch.randint(0, 300, (BATCH_SIZE, )).cuda()
    out = model(random_tensor, time)

    print(out.shape, out.shape == random_tensor.shape)