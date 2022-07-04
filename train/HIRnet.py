import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from math import sqrt


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyrelu(out)

        return out

class hard_encoder(nn.Module):

    def __init__(self,in_channels,filter_nums):
        super(hard_encoder, self).__init__()
        self.in_channels = in_channels
        self.output_channels = filter_nums
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,out_channels=self.output_channels,stride=1, kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(filter_nums)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.up = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self,x1):

        out = self.up(x1)
        out = self.bn(out)
        out = self.leakyrelu(out)
        return out


class HIRnet(nn.Module):
    def __init__(self,filters: int,
                 layers: List[int], input_channel, output_channel,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(HIRnet, self).__init__()
        self.in_channels = input_channel
        self.out_channels = output_channel
        self.filters_nums = filters #default =16

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.hard_encoder = hard_encoder(in_channels=self.in_channels, filter_nums=self.filters_nums)  # H, W, 16

        self.spectral_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=self.filters_nums,out_channels=64, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=1, stride=1),
                                    nn.LeakyReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, inplanes=self.filters_nums, planes=64, blocks=layers[0],stride=1)  # H,W,64
        self.layer2 = self._make_layer(BasicBlock, inplanes=64, planes=128, blocks=layers[1],stride=2)  # H/2. W/2,128
        self.layer3 = self._make_layer(BasicBlock, inplanes=128, planes=256, blocks=layers[2], stride=2)  # H/4, W/4, 256
        self.layer4 = self._make_layer(BasicBlock, inplanes=256, planes=512, blocks=layers[3], stride=2)  # H/8, W/8, 512

        self.up1 = Up(in_channels=512,out_channels=1024,kernel_size=2,stride=2)  # H/4, W/4, 512
        self.up2 = Up(in_channels=1024,out_channels=2048,kernel_size=2,stride=2)  # H/2, W/2, 512
        self.up3 = Up(in_channels=2048, out_channels=self.out_channels,kernel_size=2,stride=2)  # H, W, 301

        self.conv1 = nn.Conv2d(in_channels=602, out_channels=self.out_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',a=1e-2, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes: int, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample,previous_dilation, norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):       # (H,W,301)
        out = self.hard_encoder(x)    # (torch.Size([1, 16, 256, 256]))
        spectral_res = self.spectral_conv(out)  # torch.Size([1, 301, 256, 256])

        out = self.layer1(out)  # (1,64,256,256)
        out = self.layer2(out)  # (1,128,128,128)
        out = self.layer3(out)  # (1,256,64,64)
        out = self.layer4(out)  # (1,512,32,32)

        out = self.up1(out)   # (1,512,64,64)
        out = self.up2(out)  # (1,512,128,128)
        out = self.up3(out)  # (1,301,256,256)

        out = torch.cat((spectral_res,out),dim=1)  # (H, W, 602)
        out = self.conv1(out)     # (H, W, 301)
        out = self.sigmoid(out)
        return out


class HIRnet_new(nn.Module):
    def __init__(self,filters: int,
                 layers: List[int], input_channel, output_channel,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(HIRnet_new, self).__init__()
        self.in_channels = input_channel
        self.out_channels = output_channel
        self.filters_nums = filters #default =16

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.hard_encoder = hard_encoder(in_channels=self.in_channels, filter_nums=self.filters_nums)  # H, W, 16
        self.spectral_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=self.filters_nums,out_channels=64, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=1, stride=1),
                                    nn.LeakyReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, inplanes=self.filters_nums, planes=32, blocks=layers[0],stride=1)  # H,W,16
        self.layer2 = self._make_layer(BasicBlock, inplanes=32, planes=128, blocks=layers[1],stride=2)  # H/2. W/2,64
        self.layer3 = self._make_layer(BasicBlock, inplanes=128, planes=256, blocks=layers[2], stride=2)  # H/4, W/4, 128
        # self.layer4 = self._make_layer(BasicBlock, inplanes=128, planes=256, blocks=layers[2],stride=2)  # H/4, W/4, 128

        self.up1 = Up(in_channels=256,out_channels=256,kernel_size=2,stride=2)  # H/2, W/2, 256
        self.layer2_con1_1 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1, stride=1)

        self.up2 = Up(in_channels=256, out_channels=512,kernel_size=2,stride=2)  # H, W, 301
        self.layer1_con1_1 = nn.Conv2d(in_channels=544, out_channels=self.out_channels, kernel_size=1, stride=1)

        self.conv1 = nn.Conv2d(in_channels=602, out_channels=self.out_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',a=1e-2, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes: int, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample,previous_dilation, norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):       # (H,W,301)
        dim_16 = self.hard_encoder(x)    # (torch.Size([1, 16, 256, 256]))
        spectral_res = self.spectral_conv(dim_16)  # torch.Size([1, 301, 256, 256])

        lay1_out = self.layer1(dim_16)
        lay2_out = self.layer2(lay1_out)
        lay3_out = self.layer3(lay2_out)

        up2_out = self.up1(lay3_out)
        up_lay2_out = self.layer2_con1_1(torch.cat((lay2_out,up2_out),dim=1))
        up1_out = self.up2(up_lay2_out)
        up_lay1_out = self.layer1_con1_1(torch.cat((lay1_out,up1_out),dim=1))

        out = torch.cat((spectral_res,up_lay1_out),dim=1)  # (H, W, 602)
        out = self.conv1(out)  # (H, W, 301)
        out = self.sigmoid(out)
        return out

class HIRnet_Predict(nn.Module):
    def __init__(self,
                 layers: List[int], input_channel, output_channel,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(HIRnet_Predict, self).__init__()
        self.in_channels = input_channel
        self.out_channels = output_channel

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.spectral_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=self.in_channels,out_channels=64, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(in_channels=256, out_channels=301, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(301),
                                    nn.LeakyReLU(inplace=True),
                                    # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                                    # nn.BatchNorm2d(512),
                                    # nn.LeakyReLU(inplace=True),
                                    # nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=1, stride=1),
                                    # nn.LeakyReLU(inplace=True)
            )

        self.layer1 = self._make_layer(BasicBlock, inplanes=16, planes=32, blocks=layers[0],stride=1)  # H,W,16
        self.layer2 = self._make_layer(BasicBlock, inplanes=32, planes=128, blocks=layers[1],stride=2)  # H/2. W/2,64
        self.layer3 = self._make_layer(BasicBlock, inplanes=128, planes=256, blocks=layers[2], stride=2)  # H/4, W/4, 128
        # self.layer4 = self._make_layer(BasicBlock, inplanes=128, planes=256, blocks=layers[2],stride=2)  # H/4, W/4, 128

        self.up1 = Up(in_channels=256,out_channels=256,kernel_size=2,stride=2)  # H/2, W/2, 256
        self.layer2_con1_1 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1, stride=1)

        self.up2 = Up(in_channels=256, out_channels=512,kernel_size=2,stride=2)  # H, W, 301
        self.layer1_con1_1 = nn.Conv2d(in_channels=544, out_channels=self.out_channels, kernel_size=1, stride=1)

        self.conv1 = nn.Conv2d(in_channels=602, out_channels=self.out_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',a=1e-2, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes: int, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample,previous_dilation, norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):       # (H,W,301)
        spectral_res = self.spectral_conv(x)

        lay1_out = self.layer1(x)
        lay2_out = self.layer2(lay1_out)
        lay3_out = self.layer3(lay2_out)

        up2_out = self.up1(lay3_out)
        up_lay2_out = self.layer2_con1_1(torch.cat((lay2_out,up2_out),dim=1))
        up1_out = self.up2(up_lay2_out)
        up_lay1_out = self.layer1_con1_1(torch.cat((lay1_out,up1_out),dim=1))

        out = torch.cat((spectral_res,up_lay1_out),dim=1)  # (H, W, 602)
        out = self.conv1(out)  # (H, W, 301)
        out = self.sigmoid(out)
        return out



