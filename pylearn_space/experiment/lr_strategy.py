# 学习率绘制
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

max_epoch = 120  # 最大迭代数
init_lr = 0.00035  # 3.5 × 10^-4
lr = init_lr
warm_up_step = 10  # 前n个epoch启用warm-up策略

x_epoch = list(range(1, max_epoch + 1))  # 横坐标列表
lr_list_1 = []  # 记录学习率
lr_list_2 = []  # 记录学习率
lr_list_3 = []  # 记录学习率


def warm_up_rate_o(epoch):
    global lr
    if epoch <= warm_up_step:
        res = epoch / warm_up_step
        return init_lr * res
    elif warm_up_step < epoch <= 40:

        return init_lr
    else:
        lr = lr**1.01
        return lr


def warm_up_rate_t(epoch):
    global lr
    if epoch <= warm_up_step:
        res = epoch / warm_up_step
        return init_lr * res
    elif warm_up_step < epoch <= 40:
        return init_lr
    elif 40 < epoch <= 70:
        return init_lr * 0.1
    else:
        return init_lr * 0.01


for train_steps in range(1, max_epoch + 1):
    lr_list_1.append(warm_up_rate_o(train_steps))
    lr_list_2.append(warm_up_rate_t(train_steps))
    lr_list_3.append(init_lr)

plt.figure('lr')
plt.title('学习率策略')
plt.xlabel('学习轮次')
plt.ylabel('学习率')
plt.ylim()
plt.plot(x_epoch, lr_list_1, label='本文策略')
plt.plot(x_epoch, lr_list_2, '--', color='#f58220', label='传统策略', linewidth=1.2)
plt.plot(x_epoch, lr_list_3, '-.', color='#77ac98', label='无技巧', linewidth=1)
plt.legend()
plt.show()
