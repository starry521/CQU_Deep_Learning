import torch
import torch.nn as nn

from .basic_unit import _ConvIN3D, _ConvINReLU3D


"""two layer convolution, instance normalization"""
class UnetTwoLayerBlock(nn.Module):

    def __init__(self, in_channel, inter_channel, out_channel, stride=1, is_dynamic_empty_cache=False):
        
        super(UnetTwoLayerBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache

        self.unet_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1),
            _ConvINReLU3D(inter_channel, out_channel, 3, stride=stride, padding=1))
        

    def forward(self, x):

        output = self.unet_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        return output