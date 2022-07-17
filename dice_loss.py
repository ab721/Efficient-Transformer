#Courtesy https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch . Moreover, all the elements used in LovaszLoss was copied from https://github.com#/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py, the only difference being that I have written around 10 lines of LovaszLoss class that calls those elements

from torch import nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-15):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice