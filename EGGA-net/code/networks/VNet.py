import torch
from torch import nn
import pdb
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, kernel_size=3, padding=1, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class EdgeAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv = self.conv(x)
        sigmoid = self.sigmoid(conv)
        return x * sigmoid


class EdgeTrackingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeTrackingModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Conv3d(out_channels, out_channels, kernel_size=1)

        # 添加1x1卷积调整通道数
        self.adjust_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.adjust_channels(x)  # 使用1x1卷积调整通道数
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual  # 添加残差连接
        x = torch.sigmoid(self.conv4(x))
        return x


class GAM_Attention(nn.Module):
    def __init__(self, planes, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(planes, int(planes / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(planes / rate), planes)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(planes, int(planes / rate), kernel_size=7, padding=3),
            nn.BatchNorm3d(int(planes / rate)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(planes / rate), planes, kernel_size=7, padding=3),
            nn.BatchNorm3d(planes)
        )
        # self.ea = EdgeTrackingModule(planes, planes)

    def forward(self, x):
        b, c, d, h, w = x.shape

        # 应用边缘追踪模块
        # x_permute = self.ea(x)

        # 进行维度变换，适应通道注意力模块
        x_permute = x.permute(0, 2, 3, 4, 1).view(b, -1, c)

        # 通道注意力机制
        x_att_permute = self.channel_attention(x_permute).view(b, d, h, w, c)
        x_channel_att = x_att_permute.permute(0, 4, 1, 2, 3)

        # 与原始特征图相乘
        x = x * x_channel_att

        # 空间注意力机制
        x_spatial_att = self.spatial_attention(x).sigmoid()

        # 最终输出
        out = x * x_spatial_att

        return out


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


import torch.nn as nn


class VNet_encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='batch', has_dropout=False,
                 has_residual=False):
        super(VNet_encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        # Adding batch normalization layers
        self.norm1 = nn.BatchNorm3d(n_channels)
        self.norm2 = nn.BatchNorm3d(n_filters * 2)
        self.norm3 = nn.BatchNorm3d(n_filters * 4)
        self.norm4 = nn.BatchNorm3d(n_filters * 8)
        self.norm5 = nn.BatchNorm3d(n_filters * 16)

        self.block_one = nn.Sequential(
            self.norm1,
            convBlock(1, n_channels, n_filters, normalization=normalization)
        )
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = nn.Sequential(
            self.norm2,
            convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        )
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = nn.Sequential(
            self.norm3,
            convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        )
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = nn.Sequential(
            self.norm4,
            convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        )
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = nn.Sequential(
            self.norm5,
            convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        )

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res


class VNet_decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(VNet_decoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # self.__init_weight()
        # 添加
        self.edge_tracking = EdgeTrackingModule(n_filters, 16)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # 添加
        # x9 = self.edge_tracking(x9)

        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)

        return out


import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.avg_sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.max_sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.avg_sharedMLP(self.avg_pool(x))
        maxout = self.max_sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class cbam(nn.Module):
    def __init__(self, planes):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.ea = EdgeTrackingModule(planes, planes)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.ea(x) * x
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.norm1 = nn.BatchNorm3d(n_channels)
        self.norm2 = nn.BatchNorm3d(n_filters * 2)
        self.norm3 = nn.BatchNorm3d(n_filters * 4)
        self.norm4 = nn.BatchNorm3d(n_filters * 8)
        self.norm5 = nn.BatchNorm3d(n_filters * 16)

        self.block_one = nn.Sequential(
            self.norm1,
            convBlock(1, n_channels, n_filters, normalization=normalization)
        )
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = nn.Sequential(
            self.norm2,
            convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        )
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = nn.Sequential(
            self.norm3,
            convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        )
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = nn.Sequential(
            self.norm4,
            convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        )
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = nn.Sequential(
            self.norm5,
            convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        )

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # 添加
        # self.cbea1 = cbam(planes=16)
        # self.cbea2 = cbam(planes=32)
        # self.cbea3 = cbam(planes=64)
        # self.cbea4 = cbam(planes=128)
        # self.cbea5 = cbam(planes=256)

    def forward(self, input):
        x1 = self.block_one(input)
        # x1 = self.cbea1(x1)+x1

        x1_dw = self.block_one_dw(x1)
        # x1_dw = self.cbea2(x1_dw)+x1_dw

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        # x2_dw = self.cbea3(x2_dw)+x2_dw

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        # x3_dw = self.cbea4(x3_dw)+x3_dw

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        # x4_dw = self.cbea5(x4_dw)+x4_dw

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        upsampling = UpsamplingDeconvBlock
        self.block_zero_cbam = GAM_Attention(n_filters * 16)
        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_five_cbam = GAM_Attention(n_filters * 8)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 4, normalization=normalization)  # 修改卷积输出通道数
        self.block_six_cbam = GAM_Attention(n_filters * 4)
        self.block_six_up = upsampling(n_filters * 4, n_filters * 4, normalization=normalization)  # 修改上采样输出通道数

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 2, normalization=normalization)  # 修改卷积输出通道数
        self.block_seven_cbam = GAM_Attention(n_filters * 2)
        self.block_seven_up = upsampling(n_filters * 2, n_filters * 2, normalization=normalization)  # 修改上采样输出通道数

        self.block_eight = convBlock(3, n_filters * 2, n_filters, normalization=normalization)  # 修改卷积输出通道数
        self.block_eight_cbam = GAM_Attention(n_filters)
        self.block_eight_up = upsampling(n_filters, n_filters, normalization=normalization)  # 修改上采样输出通道数

        self.block_nine = convBlock(3, n_filters, n_filters, normalization=normalization)
        self.block_nine_cbam = GAM_Attention(n_filters)

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.edge_tracking = EdgeAttention(n_filters * 16, n_filters * 16)
        self.edge_tracking_x1 = EdgeTrackingModule(n_filters, n_filters)
        self.edge_tracking_x2 = EdgeAttention(n_filters * 2, n_filters * 2)

        self.x2_edg_conv = nn.Conv3d(n_filters * 4, n_filters * 2, 1, padding=0)
        self.x1_edg_conv = nn.Conv3d(n_filters * 2, n_filters * 1, 1, padding=0)

        self.cbam1 = GAM_Attention(planes=16)
        self.cbam2 = GAM_Attention(planes=32)
        self.cbam3 = GAM_Attention(planes=64)
        self.cbam4 = GAM_Attention(planes=128)
        self.cbam5 = GAM_Attention(planes=256)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        # x1 = self.cbam1(x1)+x1
        # x2 = self.cbam2(x2)+x2
        # x3 = self.cbam3(x3)+x3
        # x4 = self.cbam4(x4)+x4
        # x5 = self.cbam5(x5)+x5

        # x5 = self.edge_tracking(x5)
        x1_edg = self.edge_tracking_x1(x1)
        x2_edg = self.edge_tracking_x2(x2)

        x5 = self.block_zero_cbam(x5)
        x5_up = self.block_five_up(x5) + x4

        x5_up = self.block_five_cbam(x5_up) + x5_up
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6) + x3

        x6_up = self.block_six_cbam(x6_up) + x6_up
        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) + x2

        x7_up = torch.cat([x7_up, x2_edg], dim=1)
        x7_up = self.x2_edg_conv(x7_up)

        x7_up = self.block_seven_cbam(x7_up) + x7_up
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        x8_up = torch.cat([x8_up, x1_edg], dim=1)
        x8_up = self.x1_edg_conv(x8_up)

        x8_up = self.block_eight_cbam(x8_up) + x8_up
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9_cbam = self.dropout(x9)

        # x9_cbam = self.edge_tracking(x9_cbam)

        out_seg = self.out_conv(x9)

        return out_seg, x8_up


# class Decoder(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
#                  has_residual=False, up_type=0):
#         super(Decoder, self).__init__()
#         self.has_dropout = has_dropout
#
#         convBlock = ConvBlock if not has_residual else ResidualConvBlock
#
#         upsampling = UpsamplingDeconvBlock  ## using transposed convolution
#
#         self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)
#
#         self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)
#
#         self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)
#
#         self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)
#
#         self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
#         self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
#         self.dropout = nn.Dropout3d(p=0.5, inplace=False)
#
#
#         # 添加
#         self.edge_tracking = EdgeTrackingModule(n_filters, 1)
#
#     def forward(self, features):
#         x1 = features[0]
#         x2 = features[1]
#         x3 = features[2]
#         x4 = features[3]
#         x5 = features[4]
#
#         x5_up = self.block_five_up(x5)
#         x5_up = x5_up + x4
#
#         x6 = self.block_six(x5_up)
#         x6_up = self.block_six_up(x6)
#         x6_up = x6_up + x3
#
#         x7 = self.block_seven(x6_up)
#         x7_up = self.block_seven_up(x7)
#         x7_up = x7_up + x2
#
#         x8 = self.block_eight(x7_up)
#         x8_up = self.block_eight_up(x8)
#         x8_up = x8_up + x1
#         x9 = self.block_nine(x8_up)
#         # x9 = F.dropout3d(x9, p=0.5, training=True)
#         if self.has_dropout:
#             x9 = self.dropout(x9)
#
#         # 添加
#         x9 = self.edge_tracking(x9)
#
#         out_seg = self.out_conv(x9)
#
#         return out_seg, x8_up

class VNet_3D(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_3D, self).__init__()

        self.encoder = VNet_encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder_transpose = VNet_decoder(n_channels, n_classes, n_filters, normalization, has_dropout,
                                              has_residual, 0)
        self.decoder_linear = VNet_decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                                           1)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder_transpose(features)
        out_seg2 = self.decoder_linear(features)
        out_seg = (out_seg2 + out_seg1) / 2
        return out_seg, features


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)

        dim_in = 16
        feat_dim = 32
        self.pool = nn.MaxPool3d(3, stride=2)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(2):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(2):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, input):
        features = self.encoder(input)
        out_seg, x8_up = self.decoder(features)
        features = self.pool(features[4])
        return out_seg, features  # 4, 16, 112, 112, 80


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=False)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #   macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import pdb; pdb.set_trace()
