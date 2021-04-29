import torch
from torch import Tensor
import torch.nn as nn

class BasicConv(nn.Module) :
    def __init__(self, in_channels, out_channels, _kernel_size = 1, _stride = 1, _padding = 0) :
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = _kernel_size, stride=_stride, padding=_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x) :

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class BasicDWConv(nn.Module) :
    def __init__(self, in_channels, out_channels, _kernel_size = 1, _stride = 1, _padding = 0) :
        super(BasicDWConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = _kernel_size, stride=_stride, groups=in_channels, padding=_padding)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x) :

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class MobileNet(nn.Module) :
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()

        self.feature = nn.Sequential(
            BasicConv(3, 32, _kernel_size = 3, _stride = 2, _padding = 1),
            BasicDWConv(32, 64, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(64, 128, _kernel_size = 3, _stride = 2, _padding = 1),
            BasicDWConv(128, 128, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(128, 256, _kernel_size = 3, _stride = 2, _padding = 1),
            BasicDWConv(256, 256, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(256, 512, _kernel_size = 3, _stride = 2, _padding = 1),
            BasicDWConv(512, 512, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(512, 512, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(512, 512, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(512, 512, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(512, 512, _kernel_size = 3, _stride = 1, _padding = 1),
            BasicDWConv(512, 1024, _kernel_size = 3, _stride = 2, _padding = 1),
            BasicDWConv(1024, 1024, _kernel_size = 3, _stride = 1, _padding = 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x) :

        out = self.feature(x)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out