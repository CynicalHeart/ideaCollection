# LeNet 网络结构
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # 初始化函数
    def __init__(self):
        # 涉及到多继承问题,调用基类的构造函数
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 输入c为3,输出c/多少个卷积核16,核大小5
        self.pool1 = nn.MaxPool2d(2, 2)  # 下采样 最大池化核大小2,步长为2
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # 线性
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 输出为10类

    def forward(self, x):  # x是数据[b,c,h,w]
        """
        前向传播
        """
        x = F.relu(self.conv1(x))  # (3,32,32)=>(16,28,28)
        x = self.pool1(x)  # (16,14,14)
        x = F.relu(self.conv2(x))  # (16,14,14)=>(32,10,10)
        x = self.pool2(x)  # (32,5,5)
        x = x.view(-1, 32 * 5 * 5)  # 打平成1维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # 测试
    input_1 = torch.rand(32, 3, 32, 32)  # 生成输入
    net = LeNet()
    print(net)
