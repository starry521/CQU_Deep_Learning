
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Input layer, including re-sample, clip and normalization image."""
class InputLayer(nn.Module):

    def __init__(self, input_size, clip_window):
        super(InputLayer, self).__init__()
        self.input_size = input_size
        self.clip_window = clip_window

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)  # 以插值的方式转换张量大小
        x = torch.clamp(x, min=self.clip_window[0], max=self.clip_window[1]) # 将张量中的值限制在指定范围内
        mean = torch.mean(x) # 张量均值
        std = torch.std(x)  # 张量标准差
        x = (x - mean) / (1e-5 + std)  # 归一化操作
        return x


"""Output layer, re-sample image to original size."""
class OutputLayer(nn.Module):

    def __init__(self):
        super(OutputLayer, self).__init__()

    def forward(self, x, output_size):
        x = F.interpolate(x, size=output_size, mode='trilinear', align_corners=True)
        return x