# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
# 岭回归
def ridge(data):
    # 读入数据
    X, y = read_data()
    # 超参数
    alpha = 1
    # 正则化项
    A = alpha * np.ones(X.shape[1])# alpha * I
    # 最小二乘求权重 w = (X^T X + A)^-1 (X^T y)
    w = np.matmul((np.linalg.inv(np.matmul(X.T, X) + A)), np.matmul(X.T, y))
    return np.matmul(w, data)
# lasso回归
def lasso(data):
    # 读入数据
    X, y = read_data()
    # 超参数
    alpha = 0.01
    epoch = 10000
    lr = 0.001
    # 初始化w
    # w = np.zeros(X.shape[1])
    w = np.zeros(X.shape[1])
    # 梯度下降
    for i in range(epoch):
        # 梯度
        grad = np.matmul(X.T, np.matmul(X, w) - y) / X.shape[0] + alpha * np.sign(w)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1:
            grad /= grad_norm
        w -= lr * grad
    return np.matmul(w, data)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


# 以下代码用于本地测试lasso回归
# features = np.array([
#     [2.0133330e+03, 1.6400000e+01, 2.8932480e+02, 5.0000000e+00, 2.4982030e+01, 1.2154348e+02],
#     [2.0126670e+03, 2.3000000e+01, 1.3099450e+02, 6.0000000e+00, 2.4956630e+01, 1.2153765e+02],
#     [2.0131670e+03, 1.9000000e+00, 3.7213860e+02, 7.0000000e+00, 2.4972930e+01, 1.2154026e+02],
#     [2.0130000e+03, 5.2000000e+00, 2.4089930e+03, 0.0000000e+00, 2.4955050e+01, 1.2155964e+02],
#     [2.0134170e+03, 1.8500000e+01, 2.1757440e+03, 3.0000000e+00, 2.4963300e+01, 1.2151243e+02],
#     [2.0130000e+03, 1.3700000e+01, 4.0820150e+03, 0.0000000e+00, 2.4941550e+01, 1.2150381e+02],
#     [2.0126670e+03, 5.6000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02],
#     [2.0132500e+03, 1.8800000e+01, 3.9096960e+02, 7.0000000e+00, 2.4979230e+01, 1.2153986e+02],
#     [2.0130000e+03, 8.1000000e+00, 1.0481010e+02, 5.0000000e+00, 2.4966740e+01, 1.2154067e+02],
#     [2.0135000e+03, 6.5000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02]
#     ])
# labels = np.array([41.2, 37.2, 40.5, 22.3, 28.1, 15.4, 50. , 40.6, 52.5, 63.9])
#
# X, y = read_data()
#
# alpha = 0.01
# epoch = 10000
# lr = 0.001
# # 初始化w
# w = np.zeros((X.shape[1]))
# # 梯度下降
# for i in range(epoch):
#     # 梯度
#     grad = np.matmul(X.T, np.matmul(X, w) - y) / X.shape[0] + alpha * np.sign(w)
#     grad_norm = np.linalg.norm(grad)
#     if grad_norm > 1:
#         grad /= grad_norm
#     w -= lr * grad
# for i in range(10):
#     print(abs(np.matmul(w, features[i])-labels[i]))


