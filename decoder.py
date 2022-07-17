import torch
from torch import nn

class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

class MLPDecoder(nn.Module):
    def __init__(self, out_channels, widths, scale_factors):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                DecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features


class Head(nn.Module):
    def __init__(self, channels: int, num_classes, num_features = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

class Decoder_AND_Head(nn.Module):
    def __init__(self, decoder_channels, widths, scale_factors, num_classes):

        super().__init__()
        self.decoder = MLPDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = Head(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, features):
        features = self.decoder(features)
        segmentation = self.head(features)
        return segmentation