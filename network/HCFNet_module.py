import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from network.non_local.embedded_gaussian import NONLocalBlock2D_EGaussian
from torch.nn.parameter import Parameter
import math
from numpy import *
from typing import Union
import einops


class SiLU(torch.nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class HAFM(nn.Module):
    def __init__(self, input_channels):
        super(HAFM, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        transformed_feature = self.conv2(self.activ(self.conv1(x)))
        attention_weights = F.softmax(transformed_feature, dim=1)
        fused_feature = torch.sum(attention_weights * x, dim=1, keepdim=True)
        return fused_feature


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(F.relu(x1))
        return 0.5 * (new_features + x2), x2


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features

class _DenseLayer_DS(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer_DS, self).__init__()
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))
        self.add_module('conv3', LCDConv(out_features, out_features, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer_DS, self).forward(x1)
        return 0.5 * (new_features + x2), x2


class DenseBlock_DS(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(DenseBlock_DS, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer_DS(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16
        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_bs=True):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features=None, stride=1, use_act=True):
        super(DoubleConvBlock, self).__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x

class DoubleConvBlock_DS(nn.Module):
    def __init__(self, in_features, mid_features, out_features=None, stride=1, use_act=True):
        super(DoubleConvBlock_DS, self).__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.LCDConv = LCDConv(mid_features, out_features, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.LCDConv(x)
        return x


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=32):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class SDCM(nn.Module):
    
    """
    Now no longer used in the network, replaced by SCP, used only for ablation analysis.
    """
    def __init__(self, in_channels, out_channels):
        super(SDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class DepthwiseDiagonalConv(nn.Module):
    """
    Diagonal Convolution. We use depth-separable convolutions to improve efficiency
    """
    def __init__(self, channels, kernel_size, dilation=1, direction='main'):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv2d(channels, channels, kernel_size, 
                              padding=padding, dilation=dilation, 
                              groups=channels, bias=False)
        
        # Creating a Diagonal Mask
        mask = torch.zeros_like(self.conv.weight)
        if direction == 'main':
            for i in range(kernel_size):
                mask[:, 0, i, i] = 1
        else: # anti
            for i in range(kernel_size):
                mask[:, 0, i, kernel_size - 1 - i] = 1
        self.register_buffer('mask', mask)

    def forward(self, x):
        masked_weight = self.conv.weight * self.mask
        return F.conv2d(x, masked_weight, bias=None, 
                        padding=self.conv.padding, 
                        dilation=self.conv.dilation,
                        stride=self.conv.stride,
                        groups=x.size(1))

class SCP(nn.Module):
    """
    Structured Context Processor for side-output boundary maps.
    Replaces the SDCM module.
    """
    def __init__(self, in_channels, out_channels=21):
        super(SCP, self).__init__()
        
        bottleneck_channels = max(in_channels // 4, out_channels)
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )

        self.kernel_configs = { 5: 2, 9: 2, 13: 2 }
        
        self.depthwise_convs = nn.ModuleList()
        num_depthwise_paths = 0

        for k, d in self.kernel_configs.items():
            # 1. Horizontal Convolution
            self.depthwise_convs.append(
                nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=(1, k), 
                          padding=(0, (k // 2) * d), dilation=d, groups=bottleneck_channels, bias=False)
            )
            # 2. vertical convolution
            self.depthwise_convs.append(
                nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=(k, 1), 
                          padding=((k // 2) * d, 0), dilation=d, groups=bottleneck_channels, bias=False)
            )
            # 3. Main diagonal convolution
            self.depthwise_convs.append(
                DepthwiseDiagonalConv(bottleneck_channels, k, dilation=d, direction='main')
            )
            # 4. Anti diagonal convolution
            self.depthwise_convs.append(
                DepthwiseDiagonalConv(bottleneck_channels, k, dilation=d, direction='anti')
            )
            num_depthwise_paths += 4

        self.pointwise_fuse = nn.Sequential(
            nn.Conv2d(bottleneck_channels * num_depthwise_paths, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x_proc = self.preprocess(x)
        path_outputs = [path(x_proc) for path in self.depthwise_convs]
        fused_output = torch.cat(path_outputs, dim=1)
        return self.pointwise_fuse(fused_output)


class _LearnableMaskConv(nn.Module):
    """
    Helper module for ADP. 
    create "holed" convolutions in a learnable way.
    """
    def __init__(self, in_channels, out_channels, kernel_size=11):
        super(_LearnableMaskConv, self).__init__()
        padding = kernel_size // 2
        

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        
        self.mask_generator = nn.Parameter(torch.zeros_like(self.conv.weight))
        
        self.temperature = 1.0 

    def forward(self, x):

        learnable_mask = torch.sigmoid(self.temperature * self.mask_generator)
        

        masked_weight = self.conv.weight * learnable_mask
        

        return F.conv2d(x, masked_weight, bias=None, padding=self.conv.padding)

class ADP(nn.Module):
    """
    Adaptive Detail Processor for concated boundary map.
    Used for final refinement before HAFM fusion.
    """
    def __init__(self, in_channels, out_channels=6, kernel_size=7, num_kernels=4):
        super(ADP, self).__init__()
        
        self.learnable_convs = nn.ModuleList()
        for _ in range(num_kernels):
            # Each convolution will learn a different convolution kernel
            self.learnable_convs.append(
                _LearnableMaskConv(in_channels, out_channels, kernel_size)
            )

        # residual connection
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        


    def forward(self, x):
        # Accumulate the output of all learnable mask convolutions
        out = sum(conv(x) for conv in self.learnable_convs)
        
        out += self.residual(x)
        
        return out



class LCDConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        if self.morph == 0:
            self.dsc_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(kernel_size, 1),
                padding=0,
            )
        elif self.morph == 1:
            self.dsc_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=(1, kernel_size),
                padding=0,
            )

    def forward(self, input: torch.Tensor):
        offset = self.offset_conv(input)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )
        
        output = self.dsc_conv(deformed_feature)
        output = self.gn(output)
        output = self.relu(output)

        return output

def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: Union[str, torch.device] = "cuda",
):

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        
        y_offset_new_ = torch.zeros_like(y_offset_)
        
        if center + 1 < kernel_size:
            y_offset_new_[center + 1:] = torch.cumsum(y_offset_[center + 1:], dim=0)

        if center > 0:
            y_offset_rev = torch.flip(y_offset_[:center], dims=[0])
            y_offset_rev_cumsum = torch.cumsum(y_offset_rev, dim=0)
            y_offset_new_[:center] = torch.flip(y_offset_rev_cumsum, dims=[0])

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        
        x_offset_new_ = torch.zeros_like(x_offset_)
        if center + 1 < kernel_size:
            x_offset_new_[center + 1:] = torch.cumsum(x_offset_[center + 1:], dim=0)
        if center > 0:
            x_offset_rev = torch.flip(x_offset_[:center], dims=[0])
            x_offset_rev_cumsum = torch.cumsum(x_offset_rev, dim=0)
            x_offset_new_[:center] = torch.flip(x_offset_rev_cumsum, dims=[0])
        
        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled
