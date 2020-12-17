# MobileNet-v2与 ResNet网络
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1  # 基础残差块,不4倍升维

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # shortcut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  # 合并分支后激活

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 深度残差块,4倍升维

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, dilation=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 4倍升维
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  # 4被于初始输入
        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # shortcut
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
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
    include_top:扩展功能接口,include_top=False该模型可用于特征提取
    """

    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # 公有conv 3 224*224 => 64 112*112
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(True)
        # 最大池 : 64 112*112 => 64 56*56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ---------------------------------------------------------------
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride)]
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
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


def _make_divisible(ch, divisor=8, min_ch=None):
    # 贴近8的整数倍
    if min_ch is None:
        min_ch = divisor
    # 将ch调整到离得最近的8的倍数
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 确保向下取整时不会超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    # groups=1-普通卷积；groups=in_channel-DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),  # 使用BN层bias就不起作用,所以设置为False
            nn.ReLU6(inplace=True)  # 就地：将得到的值计算得到的值覆盖之前的值
        )


class InvertedResidual(nn.Module):
    # 倒残差结构
    # expend_ratio 是扩展因子t,将c扩大多少倍
    def __init__(self, in_channel, out_channel, stride, expend_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expend_ratio  # 隐藏维度：tk-扩展维度t倍
        # 是否使用短接，满足条件stride=1且输入输出channel相同
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 对应第一个bottleneck的t=1时没有第一个1×1的卷积层
        if expend_ratio != 1:
            # 不是第一个bottleneck,加入1×1卷积层,输入维度k,输出维度tk（PW conv）
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        # extend和append相同,但是extend可以批量插入
        # 定义：extend() 函数用于在列表末尾一次性追加另一个序列中的多个值。
        layers.extend([
            # 3×3 DW conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # PW conv linear
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])
        # 装入Sequential容器
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # 是否是用shortcut
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    # MobileNet-v2提取特征,无预训练
    def __init__(self, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # conv1 layer
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # 每个bottleneck步距除了第一块需要按照表格，其他时间都为1
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        # last layer
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        features.append(nn.Conv2d(last_channel, 128, 1))
        # 封装特征提取层
        self.features = nn.Sequential(*features)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(nn.Dropout(0.2),
        #                                 nn.Linear(last_channel, num_classes))

        # weight initialization 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # 何凯明卷积初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏差为0
            elif isinstance(m, nn.BatchNorm2d):  # m属于bn,权重为1 偏差为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):  # m属于线性,权重符合均值为0,方差为0.01的正态
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # b,1280,1,1
        x = torch.flatten(x, 1)  # 展平 b,1280*1*1
        x = x.div(x.norm(p=2, dim=1, keepdim=True))
        # x = self.classifier(x)
        return x


class MobileNetPre(nn.Module):
    # 采用MobileNet预训练网络
    def __init__(self, num_classes=751, droprate=0.5):
        super(MobileNetPre, self).__init__()
        model_pre = models.mobilenet_v2(pretrained=True)  # 采用预训练
        self.conv_last = nn.Conv2d(model_pre.last_channel, 128, kernel_size=1)
        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_pre
        self.model.classifier = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(model_pre.last_channel, num_classes),
        )
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.model.features(x)
        f = self.conv_last(x)
        x = self.avgpool(x)
        f = self.avgpool(f)
        x = torch.flatten(x, 1)
        f = torch.flatten(f, 1)
        x = self.model.classifier(x)
        return x, f


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


if __name__ == "__main__":
    # net = resnet50(751, True)
    net = MobileNetV2()
    print(net)
    summary(net, input_size=(3, 128, 64))

    in_x = Variable(torch.FloatTensor(4, 3, 128, 64))
    output = net(in_x)
    print('output size:', output.size())

    # net = MobileNetPre(num_classes=751)
    # print(net)
    # summary(net, input_size=(3, 128, 64))
    # in_x = Variable(torch.FloatTensor(1, 3, 128, 64))
    # output = net(in_x)
    # out_x, out_f = output
    # print(out_x.size(), "\n", out_f.size())
    # print(out_f)
