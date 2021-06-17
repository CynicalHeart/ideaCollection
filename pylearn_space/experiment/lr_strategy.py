# 学习率绘制
from matplotlib import pyplot as plt

max_epoch = 70  # 最大迭代数
init_lr = 0.00035  # 3.5 × 10^-4
lr = init_lr
warm_up_step = 10  # 前n个epoch启用warm-up策略

x_epoch = list(range(1, max_epoch+1))  # 横坐标列表
lr_list_1 = []  # 记录学习率
lr_list_2 = []  # 记录学习率
lr_list_3 = []  # 记录学习率


def warm_up_rate_o(epoch):
    global lr
    if epoch <= warm_up_step:
        res = epoch / warm_up_step
        return init_lr * res
    elif warm_up_step < epoch <= 30:

        return init_lr
    else:
        lr = lr ** 1.01
        return lr


def warm_up_rate_t(epoch):
    global lr
    if epoch <= warm_up_step:
        res = epoch / warm_up_step
        return init_lr * res
    elif warm_up_step < epoch <= 40:
        return init_lr
    elif 40 < epoch <= 60:
        return init_lr * 0.1
    else:
        return init_lr * 0.01


for train_steps in range(1, max_epoch+1):
    lr_list_1.append(warm_up_rate_o(train_steps))
    lr_list_2.append(warm_up_rate_t(train_steps))
    lr_list_3.append(init_lr)

plt.figure('lr')
plt.title('lr-strategy')
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.ylim()
plt.plot(x_epoch, lr_list_1, label='ours')
plt.plot(x_epoch, lr_list_2, '--', color='#f58220',
         label='tradition', linewidth=1.2)
plt.plot(x_epoch, lr_list_3, '-.', color='#77ac98',
         label='no trick', linewidth=1)
plt.legend()
plt.show()
