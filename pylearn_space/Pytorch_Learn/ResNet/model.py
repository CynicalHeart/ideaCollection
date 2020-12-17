# ResNet
import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    # downsample 对应虚线结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,  # stride=1对应实线,2对应虚线结构
            padding=1,
            bias=False)  # 不使用偏置,用批处理
        self.bn1 = nn.BatchNorm2d(out_channel)  # 输入输出通道
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,  # 输入上一层的输出
            out_channels=out_channel,
            kernel_size=3,
            stride=1,  # 第二层结构步距和padding都是1，保持维度
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)  # 输入输出通道
        self.downsample = downsample

    def forward(self, x):
        """
        前向传播
        """
        identity = x  # 捷径分支
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)  # 合并了再激活

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # 输入输出通道
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,  # 输入上一层的输出
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,  # 第二层卷积步距为2
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)  # 输入输出通道
        self.conv3 = nn.Conv2d(
            in_channels=out_channel,  # 输入上一层的输出
            out_channels=out_channel * self.expansion,  # 4被于输入
            kernel_size=1,
            stride=1,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  # 4被于初试输入
        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        """
        正向传播
        """
        identity = x
        if self.downsample is not None:
            # 如果有下采样函数证明是虚线结构,需要将identity维度下降
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    param
    ------
    block:选择BasicBlock还是Bottleneck
    block_num:一个列表存放每个block的数量
    num_classes:分类数目
    include_top:扩展功能接口,include_top=False,该模型可用于特征提取
    """
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        # 公用 7*7
        self.conv1 = nn.Conv2d(3,
                               self.in_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)  # =>112*112*64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=1)  # 56*56*64
        # ----------------------------------------------------------------
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,
                          channel * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(
            block(self.in_channel,
                  channel,
                  downsample=downsample,
                  stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top)
