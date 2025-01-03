"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from networks.common_module import DecoderBlock, nonlinearity
from networks.non_local.embedded_gaussian import NONLocalBlock2D_EGaussian
from torch.nn.parameter import Parameter
from networks.boundarymodule import *
from networks.boundarymodule_for_ablation import *
#AKConv

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
        self.l1 = Conv3BN(in_, out)#LCDConv(in_, out, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True)
        self.l2 = LCDConv(out, out, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class HCFNet_unet_DexiNed(nn.Module):  # add non-local block
    def __init__(self, num_classes=1, num_channels=3):
        super(HCFNet_unet_DexiNed, self).__init__()

        # region module
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(4, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
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
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        reg = F.sigmoid(out)
        
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        #assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(fusecat)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        

        return [reg,results]
        
        
class HCFNet_BSiNet_DexiNed(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.sge = SpatialGroupEnhance(32)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        
        self.block_1 = DoubleConvBlock(4, 32, 64, stride=2,)
        self.CoordAtt1 = CoordAtt(64,64)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
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
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        #self.block_cat = HAFM(6)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        x_out = self.sge(x9)
        reg = self.conv_final1(x_out)
        dis = self.conv_final2(x_out)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        #assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(fusecat)
        block_1_side = self.side_1(block_1)
        block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1_att)#block_1_att原本是block_1
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        results.append(block_cat)
        

        return [reg,bou_results,dis]

class HCFNet_BSiNet_DexiNed_DS(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        #self.sge = SpatialGroupEnhance(32)
        
        self.aspp = ASPP(512, 512)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(4, 32, 64, stride=2,)
        #self.CoordAtt1 = CoordAtt(64,64)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
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
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = HAFM(6)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)
        x5 = self.aspp(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        #x_out = self.sge(x9)
        reg = self.conv_final1(x9)
        dis = self.conv_final2(x9)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        #assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(fusecat)
        block_1_side = self.side_1(block_1)
        #block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1)#block_1_att原本是block_1
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        results.append(block_cat)
        

        return [reg,bou_results,dis]

class HCFNet_BSiNet_DexiNed_DS_attention(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        #self.sge = SpatialGroupEnhance(32)
        
        self.aspp = ASPP(512, 512)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(4, 32, 64, stride=2,)
        self.SEAtt1 = CBAM(64)
        #self.CoordAtt1 = CoordAtt(64,64)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.SEAtt2 = CBAM(128)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
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
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = HAFM(6)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)
        x5 = self.aspp(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        #x_out = self.sge(x9)
        reg = self.conv_final1(x9)
        dis = self.conv_final2(x9)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        #assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(fusecat)
        block_1 = self.SEAtt1(block_1)
        block_1_side = self.side_1(block_1)
        #block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1)#block_1_att原本是block_1
        block_2 = self.SEAtt2(block_2)
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        results.append(block_cat)
        

        return [reg,bou_results,dis]

        
class HCFNet_BSiNet_DexiNed_DS_123(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        #self.sge = SpatialGroupEnhance(32)
        
        self.aspp = ASPP(512, 512)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(4, 32, 64, stride=2,)
        self.nonlocal1 = NONLocalBlock2D_EGaussian(64)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal2 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
        self.nonlocal3 = NONLocalBlock2D_EGaussian(256)
        self.dblock_4 = DenseBlock(3, 256, 512)
        self.nonlocal4 = NONLocalBlock2D_EGaussian(512)
        self.dblock_5 = DenseBlock(3, 512, 512)
        self.dblock_6 = DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = HAFM(6)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)
        x5 = self.aspp(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        #x_out = self.sge(x9)
        reg = self.conv_final1(x9)
        dis = self.conv_final2(x9)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        #assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(fusecat)
        block_1_side = self.side_1(block_1)
        #block_1 = self.nonlocal1(block_1)
        #block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1)#block_1_att原本是block_1
        #block_2_NL = self.nonlocal2(block_2)#加上的
        block_2_down = self.maxpool(block_2)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal3(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_NL = self.nonlocal4(block_4)
        block_4_down = self.maxpool(block_4_NL)
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

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        results.append(block_cat)
        

        return [reg,bou_results,dis]

        
class HCFNet_BSiNet_rcf(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.sge = SpatialGroupEnhance(32)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        self.aspp = ASPP(512, 512)
        
        # boundary module
        
        self.bou_conv1 = NetModule(4, 32)
        self.bou_conv2 = NetModule(32, 64)
        self.bou_conv3 = NetModule(64, 128)
        self.bou_conv4 = NetModule(128, 256)
        self.bou_conv5 = NetModule(256, 512)

        self.bou_conv6 = NetModule(768, 256)
        self.bou_conv7 = NetModule(384, 128)
        self.bou_conv8 = NetModule(192, 64)
        self.bou_conv9 = NetModule(96, 32)

        self.bou_pool1 = nn.MaxPool2d(2, 2)
        self.bou_pool2 = nn.MaxPool2d(4, 4)
        self.bou_upsample1 = nn.Upsample(scale_factor=2)
        self.bou_upsample2 = nn.Upsample(scale_factor=4)
        
        self.bou_CDCM5 = CDCM(512, 21)
        self.bou_CDCM6 = CDCM(256, 21)
        self.bou_CDCM7 = CDCM(128, 21)
        self.bou_CDCM8 = CDCM(64, 21)
        self.bou_CDCM9 = CDCM(32, 21)
        
        self.boufea5 = nn.Conv2d(21, 1, 1)
        self.boufea6 = nn.Conv2d(21, 1, 1)
        self.boufea7 = nn.Conv2d(21, 1, 1)
        self.boufea8 = nn.Conv2d(21, 1, 1)
        self.boufea9 = nn.Conv2d(21, 1, 1)
        
        self.attentionfusion = HAFM(5)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)
        x5 = self.aspp(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        #x_out = self.sge(x9)
        reg = self.conv_final1(x9)
        dis = self.conv_final2(x9)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        x1 = self.bou_conv1(fusecat)

        x2 = self.bou_conv2(x1)
        x2 = self.bou_pool1(x2)

        x3 = self.bou_conv3(x2)
        x3 = self.bou_pool1(x3)

        x4 = self.bou_conv4(x3)
        x4 = self.bou_pool1(x4)

        x5 = self.bou_conv5(x4)
        x5 = self.bou_pool2(x5)
        bou_CDCM5 = self.bou_CDCM5(x5)

        x_6 = self.bou_upsample2(x5)
        x6 = self.bou_conv6(torch.cat([x_6, x4], 1))
        bou_CDCM6 = self.bou_CDCM6(x6)
        x6 = self.bou_upsample1(x6)
        

        x7 = self.bou_conv7(torch.cat([x6, x3], 1))
        bou_CDCM7 = self.bou_CDCM7(x7)
        x7 = self.bou_upsample1(x7)
        

        x8 = self.bou_conv8(torch.cat([x7, x2], 1))
        bou_CDCM8 = self.bou_CDCM8(x8)
        x8 = self.bou_upsample1(x8)
       

        x9 = self.bou_conv9(torch.cat([x8, x1], 1))
        bou_CDCM9 = self.bou_CDCM9(x9)
        
        boufea5 = self.boufea5(bou_CDCM5)
        boufea6 = self.boufea6(bou_CDCM6)
        boufea7 = self.boufea7(bou_CDCM7)
        boufea8 = self.boufea8(bou_CDCM8)
        boufea9 = self.boufea9(bou_CDCM9)   
       
        bou1 = F.interpolate(boufea5, size=(H, W), mode='bilinear', align_corners=True)
        bou2 = F.interpolate(boufea6, size=(H, W), mode='bilinear', align_corners=True)
        bou3 = F.interpolate(boufea7, size=(H, W), mode='bilinear', align_corners=True)
        bou4 = F.interpolate(boufea8, size=(H, W), mode='bilinear', align_corners=True)
        bou5 = F.interpolate(boufea9, size=(H, W), mode='bilinear', align_corners=True)
        fusecat = torch.cat((bou1, bou2, bou3, bou4, bou5), dim=1)
        bou_final = self.attentionfusion(fusecat)
        bou_results = [bou1, bou2, bou3, bou4, bou5, bou_final]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        return [reg,bou_results,dis]
        

       

class HCFNet_BSiNet_rcf_DS(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)
        
        self.att = CoordAtt(512,512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.sge = SpatialGroupEnhance(32)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        
        self.bou_conv1 = NetModule_DS(4, 32)
        self.bou_conv2 = NetModule_DS(32, 64)
        self.bou_conv3 = NetModule_DS(64, 128)
        self.bou_conv4 = NetModule_DS(128, 256)
        self.bou_conv5 = NetModule_DS(256, 512)

        self.bou_conv6 = NetModule(768, 256)
        self.bou_conv7 = NetModule(384, 128)
        self.bou_conv8 = NetModule(192, 64)
        self.bou_conv9 = NetModule(96, 32)

        self.bou_pool1 = nn.MaxPool2d(2, 2)
        self.bou_pool2 = nn.MaxPool2d(4, 4)
        self.bou_upsample1 = nn.Upsample(scale_factor=2)
        self.bou_upsample2 = nn.Upsample(scale_factor=4)
        
        self.bou_CDCM5 = CDCM(512, 21)
        self.bou_CDCM6 = CDCM(256, 21)
        self.bou_CDCM7 = CDCM(128, 21)
        self.bou_CDCM8 = CDCM(64, 21)
        self.bou_CDCM9 = CDCM(32, 21)
        
        self.boufea5 = nn.Conv2d(21, 1, 1)
        self.boufea6 = nn.Conv2d(21, 1, 1)
        self.boufea7 = nn.Conv2d(21, 1, 1)
        self.boufea8 = nn.Conv2d(21, 1, 1)
        self.boufea9 = nn.Conv2d(21, 1, 1)
        
        self.attentionfusion = HAFM(5)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        x_out = self.sge(x9)
        reg = self.conv_final1(x_out)
        dis = self.conv_final2(x_out)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        x1 = self.bou_conv1(fusecat)

        x2 = self.bou_conv2(x1)
        x2 = self.bou_pool1(x2)

        x3 = self.bou_conv3(x2)
        x3 = self.bou_pool1(x3)

        x4 = self.bou_conv4(x3)
        x4 = self.bou_pool1(x4)

        x5 = self.bou_conv5(x4)
        x5 = self.bou_pool2(x5)
        X5_att = self.att(x5)
        bou_CDCM5 = self.bou_CDCM5(x5)

        x_6 = self.bou_upsample2(X5_att)#x5
        x6 = self.bou_conv6(torch.cat([x_6, x4], 1))
        bou_CDCM6 = self.bou_CDCM6(x6)
        x6 = self.bou_upsample1(x6)
        

        x7 = self.bou_conv7(torch.cat([x6, x3], 1))
        bou_CDCM7 = self.bou_CDCM7(x7)
        x7 = self.bou_upsample1(x7)
        

        x8 = self.bou_conv8(torch.cat([x7, x2], 1))
        bou_CDCM8 = self.bou_CDCM8(x8)
        x8 = self.bou_upsample1(x8)
       

        x9 = self.bou_conv9(torch.cat([x8, x1], 1))
        bou_CDCM9 = self.bou_CDCM9(x9)
        
        boufea5 = self.boufea5(bou_CDCM5)
        boufea6 = self.boufea6(bou_CDCM6)
        boufea7 = self.boufea7(bou_CDCM7)
        boufea8 = self.boufea8(bou_CDCM8)
        boufea9 = self.boufea9(bou_CDCM9)   
       
        bou1 = F.interpolate(boufea5, size=(H, W), mode='bilinear', align_corners=True)
        bou2 = F.interpolate(boufea6, size=(H, W), mode='bilinear', align_corners=True)
        bou3 = F.interpolate(boufea7, size=(H, W), mode='bilinear', align_corners=True)
        bou4 = F.interpolate(boufea8, size=(H, W), mode='bilinear', align_corners=True)
        bou5 = F.interpolate(boufea9, size=(H, W), mode='bilinear', align_corners=True)
        fusecat = torch.cat((bou1, bou2, bou3, bou4, bou5), dim=1)
        bou_final = self.attentionfusion(fusecat)
        bou_results = [bou1, bou2, bou3, bou4, bou5, bou_final]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        return [reg,bou_results,dis]


class DexiNed_DS(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(3, 32, 64, stride=2,)
        #self.CoordAtt1 = CoordAtt(64,64)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
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
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = HAFM(6)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        #block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1)#block_1_att原本是block_1
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        #results.append(block_cat)
        

        return bou_results
        
class DexiNed_DS_5(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(3, 32, 64, stride=2,)
        #self.CoordAtt1 = CoordAtt(64,64)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
        self.nonlocal4 = NONLocalBlock2D_EGaussian(256)
        self.dblock_4 = DenseBlock(3, 256, 512)
        self.dblock_5 = DenseBlock(3, 512, 512)
        #self.dblock_6 = DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        #self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        #self.up_block_6 = UpConvBlock(256, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = HAFM(5)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        #block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1)#block_1_att原本是block_1
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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
        #block_5_add = block_5 + block_4_side

        # Block 6
        #block_6_pre_dense = self.pre_dense_6(block_5)
        #block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        #out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        #results.append(block_cat)
        

        return bou_results

class DexiNed_DS_7(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(3, 32, 64, stride=2,)
        #self.CoordAtt1 = CoordAtt(64,64)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
        self.nonlocal4 = NONLocalBlock2D_EGaussian(256)
        self.dblock_4 = DenseBlock(3, 256, 512)
        self.dblock_5 = DenseBlock(3, 512, 512)
        self.dblock_6 = DenseBlock(3, 512, 256)
        self.dblock_7 = DenseBlock(3, 256, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(
        self.side_6 = SingleConvBlock(256, 256, 1) 

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)
        self.pre_dense_7 = SingleConvBlock(256, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.up_block_7 = UpConvBlock(256, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = HAFM(7)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        #block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1)#block_1_att原本是block_1
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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
        block_5_side = self.side_5(block_5_add)

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])
        block_6_add = block_6 + block_5_side
        
        
        block_7_pre_dense = self.pre_dense_7(block_6)
        block_7, _ = self.dblock_7([block_6_add, block_7_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        out_7 = self.up_block_7(block_7)
        results = [out_1, out_2, out_3, out_4, out_5, out_6, out_7]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        #results.append(block_cat)
        

        return bou_results

class BSiNet_(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.sge = SpatialGroupEnhance(32)
        
        #self.aspp = ASPP(512, 512)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)
        #x5 = self.aspp(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        x_out = self.sge(x9)
        reg = self.conv_final1(x_out)
        dis = self.conv_final2(x_out)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        
        return [reg,dis]

class BSiNet_4(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        #self.conv5 = NetModule(256, 512)

        #self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        #self.sge = SpatialGroupEnhance(32)
        
        self.aspp = ASPP(256, 256)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        #x5 = self.conv5(x4)
        #x5 = self.pool2(x5)
        x5 = self.aspp(x4)

        x6 = self.upsample1(x5)
        #x6 = self.conv6(torch.cat([x_6, x4], 1))
        #x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        #x_out = self.sge(x9)
        reg = self.conv_final1(x9)
        dis = self.conv_final2(x9)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        
        return [reg,dis]

class BSiNet_6(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)
        self.conv5_1 = NetModule(512, 1024)

        self.conv5_2 = NetModule(1536, 512)
        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        #self.sge = SpatialGroupEnhance(32)
        
        self.aspp = ASPP(1024, 1024)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool1(x5)
        X5_1 = self.conv5_1(x5)
        x5_1 = self.pool2(X5_1)
        x5_1 = self.aspp(x5_1)

        x5_1_U = self.upsample2(x5_1)
        
        X5_2 = self.conv5_2(torch.cat([x5_1_U, x5], 1))
        x5_2 = self.upsample1(X5_2)
        
        x6 = self.conv6(torch.cat([x5_2, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        #x_out = self.sge(x9)
        reg = self.conv_final1(x9)
        dis = self.conv_final2(x9)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        
        return [reg,dis]

class BSiNet_7(nn.Module):  # add non-local block
    def __init__(
            self,
            input_channels: int = 3,
    ):
        super().__init__()

        # region module
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)
        self.conv5_1 = NetModule(512, 1024)
        self.conv5_2 = NetModule(1024, 2048)

        self.conv5_3 = NetModule(3072, 1024)
        self.conv5_4 = NetModule(1536, 512)
        
        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        #self.sge = SpatialGroupEnhance(32)
        
        self.aspp = ASPP(2048, 2048)
        
        self.conv_final1 = nn.Conv2d(32, 1, 1)
        self.conv_final2 = nn.Conv2d(32, 1, 1)
        
        # boundary module
        

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool1(x5)
        X5_1 = self.conv5_1(x5)
        x5_1 = self.pool1(X5_1)
        
        X5_2 = self.conv5_2(x5_1)
        x5_2 = self.pool2(X5_2)
        x5_2 = self.aspp(x5_2)

        x5_2_U = self.upsample2(x5_2)
        
        X5_3 = self.conv5_3(torch.cat([x5_2_U, x5_1], 1))
        x5_3 = self.upsample1(X5_3)
        
        X5_4 = self.conv5_4(torch.cat([x5_3, x5], 1))
        X5_4 = self.upsample1(X5_4)
        
        x6 = self.conv6(torch.cat([X5_4, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        #x_out = self.sge(x9)
        reg = self.conv_final1(x9)
        dis = self.conv_final2(x9)
        reg = F.sigmoid(reg)
        dis = F.sigmoid(dis)
        #boundary module
        
        return [reg,dis]


class HCFNet_UNET_DexiNed_DS(nn.Module):  # add non-local block
    def __init__(
            self,
            in_ch: int = 3,
            out_ch: int = 1,
    ):
        super().__init__()

        # region module
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)
        
        # boundary module
        
        self.block_1 = DoubleConvBlock_DS(4, 32, 64, stride=2,)
        #self.CoordAtt1 = CoordAtt(64,64)
        self.block_2 = DoubleConvBlock_DS(64, 128, use_act=False)
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.dblock_3 = DenseBlock(2, 128, 256) # [128,256,100,100]
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
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        #self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = HAFM(6)
        # self.block_cat = CoFusion(6,6)# cats fusion method

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        #print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        reg = F.sigmoid(c10)
        dis = F.sigmoid(c10)
        #boundary module
        fusecat = torch.cat((reg, x), dim=1)
        #assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(fusecat)
        block_1_side = self.side_1(block_1)
        #block_1_att = self.CoordAtt1(block_1)#后面加的

        # Block 2
        block_2 = self.block_2(block_1)#block_1_att原本是block_1
        block_2_NL = self.nonlocal3(block_2)#加上的
        block_2_down = self.maxpool(block_2_NL)#block_2_NL原来是block_2
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_NL = self.nonlocal4(block_3)#加上的
        block_3_down = self.maxpool(block_3_NL) # [128,256,50,50]#block_3_NL原来是block_3
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

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        
        bou_results = [out_1, out_2, out_3, out_4, out_5, block_cat]
        bou_results = [torch.sigmoid(r) for r in bou_results]

        # return results
        results.append(block_cat)
        

        return [reg,bou_results,dis]
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)    