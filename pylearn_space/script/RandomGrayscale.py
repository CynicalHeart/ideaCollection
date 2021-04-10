# RGT和RGPR实验
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import math
import random
import matplotlib
from matplotlib import pyplot as plt

# 设置中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 正负号显示

# 随机灰度官方API已经存在: torchvision.transforms.RandomGrayscale. pi = 0.05
# 经典预处理讲解: <https://blog.csdn.net/weixin_38533896/article/details/86028509>


class RandomGrayscaleErasing(object):
    """ RGPR: Random Grayscale Patch Replace
    Args:
         probability:  （Pr）擦除块代替概率
         sl: Minimum proportion of erased area against input image. 面积比例最小阈值
         sh: Maximum proportion of erased area against input image. 面积比例最大阈值
         r1: Minimum aspect ratio of erased area. 最小纵横比
         r2: Maximum aspect ratio of erased area. 最大纵横比
    """

    def __init__(self, probability=0.5, sl=0.1, sh=0.4, r1=0.4, r2=2.5):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    # 实现__call__函数，这个类型就成为可调用的。换句话说，我们可以把这个类型的对象当作函数来使用，相当于重载了括号运算符。
    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        if random.uniform(0, 1) >= self.probability:
            return img  # 随机生成[0-1)间的浮点数，如果大于p，则不执行算法，返回原始图像

        height, width = img.size()[-2], img.size()[-1]  # 高（行）, 宽（列）
        area = height * width  # 输入图像面积

        # 只循环100次防止死循环
        for _ in range(100):
            # 面积比例范围内随机生成目标面积
            target_area = random.uniform(self.sl, self.sh) * area
            # 生成目标纵横比 --> h/w
            aspect_ratio = random.uniform(self.r1, self.r2)
            # round() 方法返回浮点数x的四舍五入值
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            x = random.randint(0, width)  # 左上角坐标x
            y = random.randint(0, height)  # 左上角坐标y

            if x + w <= width and y + h <= height:
                r, g, b = img.unbind(dim=-3)  # 移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片。
                # 加权平均法
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)  # 重新绑定通道至1

                img[:, x:x + w, y:y + h] = l_img[0, x: x + w, y: y + h]  # 灰度块

                return img  # 返回随机灰度擦除图像

        return img  # 没生成随机灰度擦除图像


plt.figure('随机灰度擦除实验')


def show_img(img, transforms_l=None):
    """
    展示图像
    Args:
        img: 输入的Tensor/PIL图像
        transforms_l: 预处理列表
    """
    plt.subplot(1, 6, 1)
    plt.axis('off')
    plt.title('target')
    plt.imshow(img)
    if transforms_l is not None:
        for i in range(1, 6):
            tmp_img = img
            for transform in transforms_l:
                tmp_img = transform(tmp_img)
            if isinstance(tmp_img, torch.Tensor):
                tmp_img = tmp_img.numpy()
                tmp_img = np.transpose(tmp_img, (1, 2, 0))
            ax = plt.subplot(1, 6, 1+i)
            ax.axis('off')
            ax.set_title(i, color='green')
            plt.imshow(tmp_img)
    plt.show()


# 参数设置
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


# 测试
if __name__ == "__main__":
    image_path = r'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0114_c2s3_071702_01.jpg'
    image = Image.open(image_path)
    transforms_list = [
        transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),  # 像数据转换为torch.FloatTensor(32位浮点数格式)
        # 归一化:channel=（channel-mean）/std
        # transforms.Normalize(norm_mean, norm_std),
        RandomGrayscaleErasing()
    ]

    show_img(image, transforms_list)
