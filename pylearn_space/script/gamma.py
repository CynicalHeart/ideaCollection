import numpy as np
import random
import math
import matplotlib
from matplotlib import pyplot as plt

# 设置中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 正负号显示


def fountion(x):

    if 1 <= x < 40:
        return 32
    elif 40 <= x < 60:
        return 64
    elif 60 <= x < 70:
        return 128


plt.figure('γ')
plt.title('γ - warm up')
plt.xlabel('epoch')
plt.ylabel('γ value')
plt.xticks([1, 40, 60,70])
plt.yticks([0, 32, 64, 128])
x_epochs = np.arange(1, 80, 1)
y = [fountion(i) for i in x_epochs]
o = [64 for _ in x_epochs]
print(y)
plt.plot(x_epochs, y, 'b-', label='γ increment')
plt.plot(x_epochs, o, 'g--', label='γ original')
plt.legend()
# plt.savefig(r'D:\workspace\ideaCollection\pylearn_space\script\gamma.svg')
plt.show()
