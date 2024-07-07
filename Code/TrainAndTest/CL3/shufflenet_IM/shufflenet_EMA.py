import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

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
        super().__init__()

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

        self.ema2 = EMA(out_channels[1])
        self.ema3 = EMA(out_channels[2])
        self.ema4 = EMA(out_channels[3])

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.ema2(x)  # ECA block after stage 2
        x = self.stage3(x)
        x = self.ema3(x)  # ECA block after stage 3
        x = self.stage4(x)
        x = self.ema4(x)  # ECA block after stage 4
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups):
        """make shufflenet stage

        Args:
            block: block type, shuffle unit
            out_channels: output depth channel number of this stage
            num_blocks: how many blocks per stage
            stride: the stride of the first block of this stage
            stage: stage index
            groups: group number of group convolution
        Return:
            return a shuffle net stage
        """
        strides = [stride] + [1] * (num_blocks - 1)

        stage = []

        for stride in strides:
            stage.append(
                block(
                    self.input_channels,
                    output_channels,
                    stride=stride,
                    stage=stage,
                    groups=groups
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)

def shufflenet_EMA():
    return ShuffleNet([4, 8, 4])