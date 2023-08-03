import torch.nn as nn


"""conv+norm"""
class _ConvIN3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        
        super(_ConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)


    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        return x



"""conv+norm+drop+relu"""
class _ConvINReLU3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, p=0.2):
        
        super(_ConvINReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.drop = nn.Dropout3d(p=p, inplace=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.relu(x)

        return x


class _ConvINLeakyReLU3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, p=0.2):

        super(_ConvINLeakyReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)#channel set
        self.norm = nn.InstanceNorm3d(out_channels)
        self.drop = nn.Dropout3d(p=p, inplace=True)
        self.leakyrelu = nn.LeakyReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.leakyrelu(x)

        return x