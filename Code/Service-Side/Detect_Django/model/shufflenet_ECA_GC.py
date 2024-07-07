

from functools import partial

import torch
import torch.nn as nn

import math


import torch.nn.functional as F
from timm.models.layers.create_act import create_act_layer, get_act_layer
from timm.models.layers import make_divisible
from timm.models.layers.mlp import ConvMlp
from timm.models.layers.norm import LayerNorm2d


class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1./4, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)

        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None

        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None

        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x



class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out
class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups):
        super().__init__()

        #"""Similar to [9], we set the number of bottleneck channels to 1/4
        #of the output channels for each ShuffleNet unit."""
        self.bottlneck = nn.Sequential(
            PointwiseConv2d(
                input_channels,
                int(output_channels / 4),
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        #"""Note that for Stage 2, we do not apply group convolution on the first pointwise
        #layer because the number of input channels is relatively small."""
        if stage == 2:
            self.bottlneck = nn.Sequential(
                PointwiseConv2d(
                    input_channels,
                    int(output_channels / 4),
                    groups=groups
                ),
                nn.ReLU(inplace=True)
            )

        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            groups=int(output_channels / 4),
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv2d(
            int(output_channels / 4),
            output_channels,
            groups=groups
        )

        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        #"""As for the case where ShuffleNet is applied with stride,
        #we simply make two modifications (see Fig 2 (c)):
        #(i) add a 3 × 3 average pooling on the shortcut path;
        #(ii) replace the element-wise addition with channel concatenation,
        #which makes it easy to enlarge channel dimension with little extra
        #computation cost.
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            self.expand = PointwiseConv2d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups
            )

            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output

class ShuffleNet(nn.Module):
    def __init__(self, num_blocks, num_classes=4, groups=3):
        super(ShuffleNet, self).__init__()

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv2d(3, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[0],
            out_channels[1],
            stride=2,
            stage=2,
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[1],
            out_channels[2],
            stride=2,
            stage=3,
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups
        )

        self.eca2 = ECA_block(out_channels[1])
        self.eca3 = ECA_block(out_channels[2])
        self.eca4 = ECA_block(out_channels[3])

        self.gc2 = GlobalContext(out_channels[1])
        self.gc3 = GlobalContext(out_channels[2])
        self.gc4 = GlobalContext(out_channels[3])

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.eca2(x)  # ECA block after stage 2
        x = self.gc2(x)
        x = self.stage3(x)
        x = self.eca3(x)  # ECA block after stage 3
        x = self.gc3(x)
        x = self.stage4(x)
        x = self.eca4(x)  # ECA block after stage 4
        x = self.gc4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups):
        strides = [stride] + [1] * (num_blocks - 1)

        stage_blocks = []

        for stride in strides:
            stage_blocks.append(
                block(
                    self.input_channels,
                    output_channels,
                    stride=stride,
                    stage=stage,
                    groups=groups
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage_blocks)

def shufflenet_eca_gc():
    return ShuffleNet([4, 8, 4])




