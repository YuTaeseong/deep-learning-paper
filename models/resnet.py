import torch
from torch import Tensor
import torch.nn as nn
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

class ResidualBlock(nn.Module) :
    def __init__(self, in_channels : int, out_channels : int, stride : int) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 :
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x : Tensor) -> Tensor :
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module) :
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.maxpool = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1)
        )
        
        self.conv3 = nn.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1)
        )

        self.conv4 = nn.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1)
        )

        self.conv5 = nn.Sequential(
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 1)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.fc = nn.Linear(512,num_classes)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.maxpool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out