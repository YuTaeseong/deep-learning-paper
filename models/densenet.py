import torch
from torch import Tensor
import torch.nn as nn

class BasicConv(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0) :
        super(BasicConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x) :
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)

        return out

class Transition(nn.Module) :
    def __init__(self, in_channels) :
        super(Transition, self).__init__()

        self.conv = BasicConv(in_channels, in_channels, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride= 2)
    
    def forward(self, x) :

        out = self.conv(x)
        out = self.avgpool(out)

        return out

class Bottleneck(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        super(Bottleneck, self).__init__()

        self.conv1 = BasicConv(in_channels, 4*out_channels, kernel_size = 1)
        self.conv2 = BasicConv(4*out_channels, out_channels, kernel_size=3, stride=1, padding= 1)

    def forward(self, x) :

        out = self.conv1(x)
        out = self.conv2(out)

        return torch.cat((out, x), dim=1)

class DenseBlock(nn.Module) :
    def __init__(self, in_channels, num_layer, growth_rate) :
        super(DenseBlock, self).__init__()

        self.denseblock = self._make_layer(num_layer, in_channels, growth_rate)

    def _make_layer(self, num_layer, init_in_channel, growth_rate) :

        in_channels = init_in_channel
        layers = []
        for i in range(num_layer) :
            layers.append(Bottleneck(in_channels, growth_rate))
            in_channels = in_channels + growth_rate

        return nn.Sequential(*layers)


    def forward(self, x) :

        out = self.denseblock(x)
        
        return out

class DenseNet(nn.Module) :
    def __init__(self, growth_rate, num_classes=10) :
        super(DenseNet, self).__init__()

        num_channel = 3
        self.conv = BasicConv(num_channel, 16, kernel_size = 7, stride = 2)
        #num_channel = 16

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        self.denseblock1 = DenseBlock(num_channel, 6, growth_rate)
        num_channel = num_channel + 6 * growth_rate
        self.transition1 = Transition(num_channel)

        self.denseblock2 = DenseBlock(num_channel, 12, growth_rate)
        num_channel = num_channel + 12 * growth_rate
        self.transition2 = Transition(num_channel)
        
        self.denseblock3 = DenseBlock(num_channel, 24, growth_rate)
        num_channel = num_channel + 24 * growth_rate
        self.transition3 = Transition(num_channel)

        self.denseblock4 = DenseBlock(num_channel, 16, growth_rate)
        num_channel = num_channel + 16 * growth_rate

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channel, num_classes)

        self.feature = nn.Sequential(
            #self.conv,
            #self.maxpool,
            self.denseblock1,
            self.transition1,
            self.denseblock2,
            self.transition2,
            self.denseblock3,
            self.transition3,
            self.denseblock4
        )

    def forward(self, x) :

        out = self.feature(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out