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

res = []


class RandomGrayscaleErasing(object):
    """ RGPR: Random Grayscale Patch Replace
    Args:
         probability:  （Pr）擦除块代替概率
         sl: Minimum proportion of erased area against input image. 面积比例最小阈值
         sh: Maximum proportion of erased area against input image. 面积比例最大阈值
         r1: Minimum aspect ratio of erased area. 最小纵横比
         r2: Maximum aspect ratio of erased area. 最大纵横比
    """

    def __init__(self, probability=0.5, sl=0.1, sh=0.4, r1=0.3, r2=3.33):
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
        for _ in range(400):
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
                global res
                res.append([x, y, w, h])
                r, g, b = img.unbind(dim=-3)  # 移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片。
                # 加权平均法
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)  # 重新绑定通道至1

                img[0, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]  # 灰度块
                img[1, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]  # 灰度块
                img[2, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]  # 灰度块

                return img  # 返回随机灰度擦除图像

        return img  # 没生成随机灰度擦除图像


def show_img(img, transforms_l=None):
    """
    展示图像
    Args:
        img: 输入的Tensor/PIL图像
        transforms_l: 预处理列表
    """
    plt.figure('随机灰度擦除实验')
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
            plt.imshow(tmp_img)
    plt.show()


def show_img_2(img_list, re_rge):
    """
    RGE显示
    """
    print(type(img_list), type(re_rge))  # 显示类型
    to = transforms.ToTensor()  # 转Tensor
    lenght = len(img_list)
    plt.figure('RE')
    for i, path in enumerate(img_list):
        img = Image.open(path)  # 加载图像
        img = to(img)
        tmp_img = re_rge(img)  # 随机处理
        if isinstance(tmp_img, torch.Tensor):
            tmp_img = tmp_img.numpy()  # 转numpy
            tmp_img = np.transpose(tmp_img, (1, 2, 0))
        ax = plt.subplot(1, lenght, 1 + i)  # 子图
        ax.axis('off')  # 取消轴
        ax.add_patch(plt.Rectangle(xy=(res[i][0], res[i][1]), width=res[i][2],
                                   height=res[i][3], fill=False, linewidth=1, edgecolor="white"))  # xy左下角坐标
        ax.imshow(tmp_img)
    plt.suptitle('Random Grayscale Erasing')
    # 图间距
    plt.tight_layout()
    plt.show()  # 显示


def show_img_3(img, transforms_l=None):
    """
    展示图像
    Args:
        img: 输入的Tensor/PIL图像
        transforms_l: 预处理列表
    """

    plt.figure('预处理')
    plt.subplot(1, 6, 1)
    plt.axis('off')
    plt.title('target')
    plt.imshow(img)

    to = transforms.ToTensor()
    img = to(img)
    for i, t in enumerate(transforms_l):
        tmp_img = img
        tmp_img = t(tmp_img)
        if isinstance(tmp_img, torch.Tensor):
            tmp_img = tmp_img.numpy()
            tmp_img = np.transpose(tmp_img, (1, 2, 0))

        ax = plt.subplot(1, 6, 2+i)
        ax.axis('off')
        plt.imshow(tmp_img)
    plt.tight_layout()
    plt.show()


# 参数设置
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


# 测试
if __name__ == "__main__":
    image_path = r'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0114_c2s3_071702_01.jpg'
    image = Image.open(image_path)
    transforms_list = [
        transforms.Pad(10),
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(1),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
        # transforms.transforms.ToTensor(),  # 像数据转换为torch.FloatTensor(32位浮点数格式)
        # # 归一化:channel=（channel-mean）/std
        transforms.Normalize(norm_mean, norm_std)
    ]

    show_img_3(image, transforms_list)
    # image_list = [
    #     'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0042_c3s3_064169_01.jpg',
    #     'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0052_c2s1_005076_01.jpg',
    #     'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0093_c2s1_014226_01.jpg',
    #     'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0306_c6s3_090842_02.jpg',
    #     'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0554_c6s2_000593_01.jpg',
    #     'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0843_c3s2_107178_07.jpg',
    #     'D:/workspace/DeepL/dataset/market1501/bounding_box_train/0547_c2s1_157341_02.jpg'
    # ]
    # show_img_2(image_list, RandomGrayscaleErasing(1))
    # transforms_r = [
    #     transforms.RandomErasing(p=1),
    #     RandomGrayscaleErasing(1)
    # ]
