# 调用模型训练的文件
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import *

transform = transforms.Compose([
    transforms.ToTensor(),  # 转成tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 标准化
])
# 50000张训练图片
train_set = torchvision.datasets.CIFAR10("G:\\datasets\\train_data",
                                         True,
                                         download=False,
                                         transform=transform)
train_loder = torch.utils.data.DataLoader(train_set,
                                          batch_size=36,
                                          shuffle=True,
                                          num_workers=0)
# 10000张验证图片
val_set = torchvision.datasets.CIFAR10(root='G:\\datasets\\train_data',
                                       train=False,
                                       download=False,
                                       transform=transform)
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=2000,
                                         shuffle=True,
                                         num_workers=0)
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()
# 类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

net = LeNet()
loss_function = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  #优化器

# 训练
for epoch in range(10):
    running_loss = 0.0
    for step, data in enumerate(train_loder, start=0):
        inputs, labels = data
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播+反向传播+优化
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(val_image)
                predict_y = torch.max(outputs, dim=1)[1]  # index
                acc = (predict_y == val_label).sum().item() / val_label.size(0)
                print('[%d,%5d]train loss: %.3f  test_acc:%.3f' %
                      (epoch + 1, step + 1, running_loss / 500, acc))
                running_loss = 0.0

print("Finished Train")

save_path = "G:\\datasets\\train_data\\LeNet.pth"
torch.save(net.state_dict, save_path)
