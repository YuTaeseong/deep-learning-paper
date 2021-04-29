import torch
from torch import Tensor
import torch.nn as nn
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

class ConvWithBN(nn.Module) :
    def __init__(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0) :
        super(ConvWithBN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x) :

        out = conv(x)
        out = bn(out)
        out = relu(out)

        return out

class StemConcat_A(nn.Module) :
    def __init__(self) :
        super(StemConcat_A, self).__init__()

        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = nn.ConvWithBN(64, 96, kernel_size = 3, stride=2, bias=False)

    def forward(self, x) :

        out = []
        out.append(self.maxpool(x))
        out.append(self.conv(x))

        return torch.cat(out, 1)

class StemConcat_B(nn.Module) :
    def __init__(self) :
        super(StemConcat_B, self).__init__()

        self.branch_a = nn.Sequential(
            nn.ConvWithBN(160, 64, kernel_size = 1, stride=1, bias=False),
            nn.ConvWithBN(64, 96, kernel_size = 3, stride=1, bias=False)
        )

        self.branch_b = nn.Sequential(
            nn.ConvWithBN(160, 64, kernel_size = 1, stride=1, bias=False),
            nn.ConvWithBN(64, 64, kernel_size = (7, 1), stride=1, bias=False, padding=(3, 0)),
            nn.ConvWithBN(64, 64, kernel_size = (1, 7), stride=1, bias=False, padding=(0, 3)),
            nn.ConvWithBN(64, 96, kernel_size = 3, stride=1, bias=False)
        )

    def forward(self, x) :

        out = []
        out.append(self.maxpool(x))
        out.append(self.maxpool(x))

        return torch.cat(out, 1)

class StemConcat_C(nn.Module) :
    def __init__(self) :
        super(StemConcat_C, self).__init__()

        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = nn.ConvWithBN(192, 192, kernel_size = 3, stride=2, bias=False)

    def forward(self, x) :

        out = []
        out.append(self.maxpool(x))
        out.append(self.conv(x))

        return torch.cat(out, 1)


class Stem(nn.Module) :
    def __init__(self, in_channels : int, out_channels : int, stride : int) -> None:
        super(Stem, self).__init__()

        self.stem_a = StemConcat_A()
        self.stem_b = StemConcat_B()
        self.stem_c = StemConcat_C()
    
    def forward(self, x : Tensor) -> Tensor :

        out = self.stem_a(x)
        out = self.stem_b(out)
        out = self.stem_c(out)

        return out

class Inception_A(nn.Module) :
    def __init__(self) :
        super(Inception_A, self).__init__()
        
        self.branch_a = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            ConvWithBN(384, 96, kernel_size = 1, stride= 1)
        )

        self.branch_b = ConvWithBN(384, 96, kernel_size = 1, stride = 1)

        self.branch_c = nn.Sequential(
            ConvWithBN(384, 64, kernel_size = 1, stride = 1),
            ConvWithBN(64, 96, kernel_size = 3, stride = 1, padding = 1)
        )

        self.branch_d = nn.Sequential(
            ConvWithBN(384, 64, kernel_size = 1, stride = 1),
            ConvWithBN(64, 96, kernel_size = 3, stride = 1, padding = 1),
            ConvWithBN(96, 96, kernel_size = 3, stride = 1, padding = 1)
        )

    def forward(self, x) :

        out = []
        out.append(self.branch_a(x))
        out.append(self.branch_b(x))
        out.append(self.branch_c(x))
        out.append(self.branch_d(x))

        return torch.cat(out, 1)

class Reduction_A(nn.Module) :
    def __init__(self) :
        super(Reduction_A, self).__init__()

        self.branch_a = nn.MaxPool2d(3, stride=2)

        self.branch_b = ConvWithBN(384, 384, kernel_size = 3, stride = 2)

        self.branch_c = nn.Sequential(
            ConvWithBN(384, 192, kernel_size = 1, stride = 1),
            ConvWithBN(192, 224, kernel_size = 3, stride = 1, padding = 1),
            ConvWithBN(224, 256, kernel_size = 3, stride = 2)
        )

    def forward(self, x) :

        out = []
        out.append(self.branch_a(x))
        out.append(self.branch_b(x))
        out.append(self.branch_c(x))

        return torch.cat(out, 1)

class Inception_B(nn.Module) :
    def __init__(self) :
        super(Inception_B, self).__init__()
        
        self.branch_a = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            ConvWithBN(1024, 128, kernel_size = 1, stride= 1)
        )

        self.branch_b = ConvWithBN(1024, 384, kernel_size = 1, stride = 1)

        self.branch_c = nn.Sequential(
            ConvWithBN(1024, 192, kernel_size = 1, stride = 1),
            ConvWithBN(192, 224, kernel_size = (1,7), stride = 1, padding = (0, 3)),
            ConvWithBN(224, 256, kernel_size = (7,1), stride = 1, padding = (3, 0))
        )

        self.branch_d = nn.Sequential(
            ConvWithBN(1024, 192, kernel_size = 1, stride = 1),
            ConvWithBN(192, 192, kernel_size = (1,7), stride = 1, padding = (0, 3)),
            ConvWithBN(192, 224, kernel_size = (7,1), stride = 1, padding = (3, 0)),
            ConvWithBN(224, 224, kernel_size = (1,7), stride = 1, padding = (0, 3)),
            ConvWithBN(224, 256, kernel_size = (7,1), stride = 1, padding = (3, 0))
        )

    def forward(self, x) :

        out = []
        out.append(self.branch_a(x))
        out.append(self.branch_b(x))
        out.append(self.branch_c(x))
        out.append(self.branch_d(x))

        return torch.cat(out, 1)

class Reduction_B(nn.Module) :
    def __init__(self) :
        super(Reduction_B, self).__init__()

        self.branch_a = nn.MaxPool2d(3, stride=2)

        self.branch_b = nn.Sequential(
            ConvWithBN(1024, 192, kernel_size = 1, stride = 1),
            ConvWithBN(192, 192, kernel_size = 3, stride = 2),
            ConvWithBN(224, 256, kernel_size = 3, stride = 2)
        )

        self.branch_c = nn.Sequential(
            ConvWithBN(1024, 256, kernel_size = 1, stride = 1),
            ConvWithBN(256, 256, kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            ConvWithBN(256, 320, kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            ConvWithBN(320, 320, kernel_size = 3, stride = 2)
        )

    def forward(self, x) :

        out = []
        out.append(self.branch_a(x))
        out.append(self.branch_b(x))
        out.append(self.branch_c(x))

        return torch.cat(out, 1)

class Inception_C(nn.Module) :
    def __init__(self) :
        super(Inception_A, self).__init__()
        
        self.branch_a = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            ConvWithBN(1536, 256, kernel_size = 1, stride= 1)
        )

        self.branch_b = ConvWithBN(1536, 256, kernel_size = 1, stride = 1)

        self.branch_c1 = ConvWithBN(1536, 384, kernel_size = 1, stride = 1)
        self.branch_c2 = ConvWithBN(384, 256, kernel_size = (1, 3), stride = 1, padding = (0, 1))
        self.branch_c3 = ConvWithBN(384, 256, kernel_size = (3, 1), stride = 1, padding = (1, 0))

        self.branch_d1 = nn.Sequential(
            ConvWithBN(1536, 384, kernel_size = 1, stride = 1),
            ConvWithBN(384, 448, kernel_size = (1,3), stride = 1, padding = (0, 1)),
            ConvWithBN(448, 512, kernel_size = (3,1), stride = 1, padding = (1, 0))
        )
        self.branch_d2 = ConvWithBN(512, 256, kernel_size = (3, 1), stride = 1, padding = (1, 0))
        self.branch_d3 = ConvWithBN(512, 256, kernel_size = (1, 3), stride = 1, padding = (1, 0))

    def forward(self, x) :

        out = []
        out.append(self.branch_a(x))
        out.append(self.branch_b(x))

        out_branch_c = self.branch_c1(x)
        out_branch_c = torch.cat([self.branch_c2(out_branch_c), self.branch_c3(out_branch_c)])
        out.append(out_branch_c)

        out_branch_d = self.branch_d1(x)
        out_branch_d = torch.cat([self.branch_d2(out_branch_d), self.branch_d3(out_branch_d)])
        out.append(out_branch_d)

        return torch.cat(out, 1)

class Inception4(nn.Module) :
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.feature = nn.Sequential(
            ConvWithBN(3, 32, kernel_size=3, stride = 2),
            ConvWithBN(32, 32, kernel_size=3, stride = 1),
            ConvWithBN(32, 64, kernel_size=3, stride = 1),
            # Stem,
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),
            Inception_B,
            Inception_B,
            Inception_B,
            Inception_B,
            Inception_B,
            Inception_B,
            Inception_B,
            Reduction_B,
            Inception_C,
            Inception_C,
            Inception_C,
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(1536,num_classes)

    def forward(self, x):
        out = self.feature(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out

def inception4(data='cifar10'):
    if data == 'cifar10':
        return Inception4(num_classes=10)
    elif data == 'cifar100':
        return Inception4(num_classes=100)