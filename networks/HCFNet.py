import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from networks.non_local.embedded_gaussian import NONLocalBlock2D_EGaussian
from torch.nn.parameter import Parameter
from networks.boundarymodule import *

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class NetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class NetModule_DS(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = LCDConv(out, out, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class HCFNet(nn.Module):  # add non-local block
    module = NetModule
    def __init__(
            self,
            input_channels: int = 3,
            filters_base: int = 32,
            down_filter_factors=(1, 2, 4, 8, 16),
            up_filter_factors=(1, 2, 4, 8, 16),
            bottom_s=4,
            num_classes=1,
    ):
        super().__init__()

        # region module
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )

        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.sge = SpatialGroupEnhance(32)

        
        self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        #self.conv_final2 = nn.Conv2d(up_filter_sizes[0], 1, 1)
        
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(4, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock_DS(2, 128, 256) # [128,256,100,100]
        self.nonlocal4 = NONLocalBlock2D_EGaussian(256)
        self.dblock_4 = DenseBlock(3, 256, 512)
        self.dblock_5 = DenseBlock(3, 512, 512)
        self.dblock_6 = DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)
        # Stacked dilation convolution-based module for multi-level maps
        self.SDCM1 = SDCM(64, 21)
        self.SDCM2 = SDCM(128, 21)
        self.SDCM3 = SDCM(256, 21)
        self.SDCM4 = SDCM(512, 21)
        self.SDCM5 = SDCM(512, 21)
        self.SDCM6 = SDCM(256, 21)


        self.up_block_1 = UpConvBlock(21, 1)
        self.up_block_2 = UpConvBlock(21, 1)
        self.up_block_3 = UpConvBlock(21, 2)
        self.up_block_4 = UpConvBlock(21, 3)
        self.up_block_5 = UpConvBlock(21, 4)
        self.up_block_6 = UpConvBlock(21, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # Traditional edge fusion 
        self.block_cat = HAFM(6)

    def forward(self, x):
        # Encoder
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers, self.up))
        ):

            x_out2 = upsample(x_out)
            x_out= (torch.cat([x_out2, x_skip], 1))
            x_out = up(x_out)
        
        x_out = self.sge(x_out)
        reg = self.conv_final1(x_out)
        #dis = self.conv_final2(x_out)
        
        reg = F.sigmoid(reg)
        #dis = F.sigmoid(dis)
        
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        #assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(fusecat)
        block_1_side = self.side_1(block_1)
        #block_1_att = self.CoordAtt1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_NL = self.nonlocal3(block_2)
        block_2_down = self.maxpool(block_2_NL)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)
        block_3_down = self.maxpool(block_3_NL)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(
            block_4_down) #block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        SDCM1 = self.SDCM1(block_1)
        SDCM2 = self.SDCM2(block_2)
        SDCM3 = self.SDCM3(block_3)
        SDCM4 = self.SDCM4(block_4)
        SDCM5 = self.SDCM5(block_5)
        SDCM6 = self.SDCM6(block_6)
        
        
        # upsampling blocks
        out_1 = self.up_block_1(SDCM1)
        out_2 = self.up_block_2(SDCM2)
        out_3 = self.up_block_3(SDCM3)
        out_4 = self.up_block_4(SDCM4)
        out_5 = self.up_block_5(SDCM5)
        out_6 = self.up_block_6(SDCM6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        #results.append(block_cat)
        

        return [reg,bou_results]    