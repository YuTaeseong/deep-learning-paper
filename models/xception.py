import torch
from torch import Tensor
import torch.nn as nn

class SeparableConv(nn.Module) :
    def __init__(self, in_channels, out_channels, _kernel_size, _padding=0, _stride=1) :
        super(SeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size = _kernel_size, padding =_padding, stride = _stride, groups = in_channels) 
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1)

    def forward(self, x) :

        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class ResBlock(nn.Module) :
    def __init__(self, pos):
        super(ResBlock, self).__init__()

        self.pos = pos
        self.blocks = nn.ModuleList()
        self.downsampler = nn.ModuleList()

        if(pos == 0) :
            self.blocks.append(nn.Sequential(
                SeparableConv(64, 128, _kernel_size = 3, _padding= 1),
                nn.BatchNorm2d(128),
                self._make_entry_layer(1, 128, 128),
            ))
            self.downsampler.append(
                nn.Conv2d(64, 128, kernel_size = 1, stride=2)
            )

            self.blocks.append(self._make_entry_layer(2, 128, 256))
            self.downsampler.append(
                nn.Conv2d(128, 256, kernel_size = 1, stride=2)
            )

            self.blocks.append(self._make_entry_layer(2, 256, 728))
            self.downsampler.append(
                nn.Conv2d(256, 728, kernel_size = 1, stride=2)
            )

        elif(pos == 1) :
            for i in range(8) : 
                self.blocks.append(self._make_middle_layer(3, 728, 728))

        elif(pos == 2) :
            self.blocks.append(nn.Sequential(
                nn.ReLU(),
                SeparableConv(728, 728, _kernel_size = 3, _padding= 1),
                nn.BatchNorm2d(728),
                nn.ReLU(),
                SeparableConv(728, 1024, _kernel_size = 3, _padding= 1),
                nn.BatchNorm2d(1024),
                nn.MaxPool2d(kernel_size = 3, stride= 2, padding=1)
            ))
            self.downsampler.append(
                nn.Conv2d(728, 1024, kernel_size = 1, stride=2)
            )

    def _make_entry_layer(self, num_layer, in_channels, out_channels) :
        layers = []

        for i in range(num_layer) :
            layers.append(nn.ReLU())
            layers.append(SeparableConv(in_channels, out_channels, _kernel_size = 3, _padding= 1))
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size = 3, stride= 2, padding= 1))

        return nn.Sequential(*layers)

    def _make_middle_layer(self, num_layer, in_channels, out_channels) :
        layers = []

        for i in range(num_layer) :
            layers.append(nn.ReLU())
            layers.append(SeparableConv(in_channels, out_channels, _kernel_size = 3, _padding= 1))
            layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x) :

        out = x
        for i in range(len(self.blocks)) :
            identity = out

            if(i < len(self.downsampler)) :
                identity = self.downsampler[i](identity)
                
            out = self.blocks[i](out)
            out += identity

        return out

class Xception(nn.Module) :
    def __init__(self, num_classes):
        super(Xception, self).__init__()

        self.entry1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.entry2 = ResBlock(0)
    
        self.middle = ResBlock(1)

        self.exit1 = ResBlock(2)
        self.exit2 = nn.Sequential(
            SeparableConv(1024, 1536, _kernel_size = 3, _padding= 1),
            nn.ReLU(),
            SeparableConv(1536, 2048, _kernel_size = 3, _padding= 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x) :

        out = self.entry1(x)
        out = self.entry2(out)
        out = self.middle(out)
        out = self.exit1(out)
        out = self.exit2(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out