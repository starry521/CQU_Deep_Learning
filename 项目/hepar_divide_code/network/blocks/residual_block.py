import torch
import torch.nn as nn

from .basic_unit import _ConvIN3D, _ConvINReLU3D


"""two layer convolution, instance normalization, drop out and ReLU"""
class ResTwoLayerConvBlock(nn.Module):

    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
        
        super(ResTwoLayerConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache

        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        
        self.shortcut_unit = _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output


"""four layer convolution, instance normalization, drop out and ReLU"""
class ResFourLayerConvBlock(nn.Module):
    
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):

        super(ResFourLayerConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        
        self.residual_unit_1 = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1))
        
        self.residual_unit_2 = nn.Sequential(
            _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        
        self.shortcut_unit_1 = _ConvIN3D(in_channel, inter_channel, 1, stride=stride, padding=0)
        self.shortcut_unit_2 = nn.Sequential()

        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()
        output_1 = self.relu_1(output_1)

        output_2 = self.residual_unit_2(output_1)
        output_2 += self.shortcut_unit_2(output_1)
        if self.is_dynamic_empty_cache:
            del output_1
            torch.cuda.empty_cache()

        output_2 = self.relu_2(output_2)

        return output_2