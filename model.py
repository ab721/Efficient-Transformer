import torch
from torch import nn
from encoders import SwinTransformerV2, EfficientNetV2
from decoder import Decoder_AND_Head


class Efficient_Transformer(nn.Module):
    def __init__(self, swin_weight_path, decoder_channels, decoder_scale_factors, decoder_widths, 
                 num_classes, swin_drop_rate, swin_attn_drop_rate, swin_drop_path_rate
                 ):
        super().__init__()
        
        self.swin_features = SwinTransformerV2(drop_rate=swin_drop_rate, attn_drop_rate=swin_attn_drop_rate, 
                                               drop_path_rate=swin_drop_path_rate)  #load the weights later
        self.swin_features.load_state_dict(torch.load(swin_weight_path)['model'])
        self.efficient_features = EfficientNetV2(pretrained = True)
        self.decoder = Decoder_AND_Head(decoder_channels, decoder_widths, decoder_scale_factors, num_classes)
        if num_classes == 1: self.activation = nn.Sigmoid()
        else: self.activation = nn.Softmax()
        self.upsample = nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = True)

    def forward(self, x):

        batch_size = x.size()[0]

        ##################                      ENCODER                  #####################################
            
        #1x1024x1,   128x96x96, 256x48x48, 512x24x24, 1024x12x12 (reshaped to these shapes below) (Note that s6 is not the output, rather the layer after normalization and pooling)
        #(If this was the UNet, think of s6 as the standalone node between the encoder and decoder)
        s6, (s2, s3, s4, s5) = self.swin_features(x)
        s2 = torch.reshape(torch.permute(s2, (0, 2, 1)), (batch_size, 128, 96, 96))
        s3 = torch.reshape(torch.permute(s3, (0, 2, 1)), (batch_size, 256, 48, 48))
        s4 = torch.reshape(torch.permute(s4, (0, 2, 1)), (batch_size, 512, 24, 24))
        s5 = torch.reshape(torch.permute(s5, (0, 2, 1)), (batch_size, 1024, 12, 12))
        s6 = torch.reshape(torch.permute(s6, (0, 2, 1)), (batch_size, 1024, 1, 1))

        #32x192x192, 56x96x96, 80x48x48, 192x24x24,328x12x12                                                                                                                                
        e1, e2, e3, e4, e5 = self.efficient_features(x) 

        #184x96x96, 336x48x48, 704x24x24, 1352x12x12, 1352x1x1
        x2 = torch.cat((s2, e2), dim = 1)   
        x3 = torch.cat((s3, e3), dim = 1) 
        x4 = torch.cat((s4, e4), dim = 1)
        x5 = torch.cat((s5, e5), dim = 1)
        #x6 = s6              #The pooled output from the final layer of the encoder could come in handy for other decoders, but it is not useful in mine
        print(x5.size(), x4.size(), x3.size(), x2.size())
        y = self.decoder([x5, x4, x3, x2]) #Note: Sigmoid/softmax has not been applied

        y = self.upsample(y)

        y = self.activation(y)

        return y


if __name__ == '__main__':
    x = torch.randn((1, 3, 384, 384))
    mdl = Efficient_Transformer(swin_weight_path = '.\swin_weights.pth', decoder_channels = 256, 
                                  decoder_scale_factors = [8, 4, 2, 1], swin_drop_rate = 0, swin_attn_drop_rate = 0, 
                                  swin_drop_path_rate = 0.2, decoder_widths = [152,304,632,1376], num_classes = 1
                                 )                                           #[184, 336, 704, 1352]
    y = mdl(x)
    print(y.size())
    print(sum(p.numel() for p in mdl.parameters() if p.requires_grad))
