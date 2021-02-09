# pytorch模型转TorchScript测试
# resnet18实验
import torch
import torchvision
from torchsummary import summary
import os.path as osp

# 模型resnet18
model = torchvision.models.resnet18()

# print(type(model), '\n', model)

# 关键点路径root
path = r'D:/workspace/DeepL/checkpoints'

if osp.isdir(path):
    checkpoint = torch.load(osp.join(path, 'resnet18-5c106cde.pth'))  # 关键点
    model.load_state_dict(checkpoint)

model.eval()  # 包含batchnorm, 设置为eval模型
model.cuda()  # 送入GPU

# summary(model, input_size=(3, 224, 224))

test_input = torch.randn(16, 3, 224, 224).cuda()  # 设计输入
trace_model = torch.jit.trace(model, test_input)  # 跟踪模型 trace【传入模型，和输入参数】

# 显示中间脚本代码
print(trace_model.code)

# 展示结果
output = trace_model(test_input)  # 输入网络
print(output.size())

# 保存TorchScript
trace_model.save(osp.join(path, "resnet18_TorchScript.pt"))
