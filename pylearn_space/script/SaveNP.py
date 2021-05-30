# 尝试各种保存数组的方案
import numpy as np


def plan_1():

    # 模拟数组1
    feat = np.random.rand(30, 512)
    print(feat.shape)
    # 模拟数组2
    center = np.random.randint(1, 640, size=(30, 2))
    print(center.shape)
    np.savez('feature', f=feat)
    np.savez('center', c=center)

    f = np.load('feature.npz')
    print(f['f'])
    c = np.load('center.npz')
    print(c['c'])

    # 再次模拟数组1
    feat = np.random.rand(30, 512)
    # 再次模拟数组2
    center = np.random.randint(1, 640, size=(30, 2))

    np.savez('feature', f=feat)
    np.savez('center', c=center)

    f = np.load('feature.npz')
    print(f['f'].shape)
    c = np.load('center.npz')
    print(c['c'].shape)


def plan_2():
    # 模拟数组1
    feat_1 = np.random.rand(30, 512)
    print(feat_1.shape)
    # 模拟数组2
    center_1 = np.random.randint(1, 640, size=(30, 2))
    print(center_1.shape)

    feat_1.tofile("feat.bin")
    center_1.tofile("center.bin")

    f_res = np.fromfile('feat.bin', dtype=np.float64)
    c_res = np.fromfile('center.bin', dtype=np.int32)

    print(f_res.shape, c_res.shape)


def plan_3():
    feat_1 = np.random.rand(30, 512)
    feat_2 = np.random.rand(30, 512)
    feat_3 = np.random.rand(30, 512)

    f = open('feature.npz', 'ab')
    np.save(f, feat_1)
    np.save(f, feat_2)
    np.save(f, feat_3)
    f.close()

    f = open('feature.npz', 'rb')
    res = np.load(f)
    print(res)
    print(res.shape)


def plan_4():
    feat_1 = np.random.rand(30, 512)
    feat_2 = np.random.rand(30, 512)
    feat_3 = np.random.rand(30, 512)

    with open('feature.bin', 'ab') as f:
        np.savetxt(f, feat_1, fmt="%.6f")
        np.savetxt(f, feat_2, fmt="%.6f")
        np.savetxt(f, feat_3, fmt="%.6f")

    r = np.loadtxt('feature.bin')
    print(r)
    print(r.shape)


def plan_5():
    feat = []
    for i in range(90):
        feat.append(np.random.rand(512))
    with open('feature.bin', 'ab') as f:
        np.savetxt(f, feat, fmt="%.6f")

    r = np.loadtxt('feature.bin')
    print(type(r))
    print(r)
    print(r.shape)


# plan_1()  # 无法追加

# plan_2()  # 读取出来的是一维 不存shape

# plan_3()  # 文件流会覆盖掉

# plan_4()  # 存txt转Bin可保存其格式和shape


plan_5()  # 证明list中存numpy可以直接转数组存入文件中
