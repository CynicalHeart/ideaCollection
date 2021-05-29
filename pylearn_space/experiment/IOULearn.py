# IoU交并比学习 - Intersection over Union
# IoU 作为目标检测算法性能 mAP 计算的一个非常重要的函数


def IOU(set_a, set_b):  # 早上角横坐标0、纵坐标1,右下角横坐标2、纵坐标3
    # 计算中间矩阵的宽高
    w = min(set_a[2], set_b[2]) - max(set_a[0], set_b[0])
    h = min(set_a[3], set_b[3]) - max(set_a[1], set_b[1])
    # 计算交集与并集面积
    inter = 0 if w < 0 or h < 0 else w * h
    # 并集 = a + b - inter
    union = (set_a[3] - set_a[1]) * (set_a[2] - set_a[0]) + (
        set_b[3] - set_b[1]) * (set_b[2] - set_b[0]) - inter

    iou = inter / union

    return iou


if __name__ == "__main__":
    set_a = [0, 0, 6, 8]
    set_b = [3, 2, 9, 10]
    print(IOU(set_a=set_a, set_b=set_b))