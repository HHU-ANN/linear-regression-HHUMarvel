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
    alpha = 0.1
    # 正则化项
    A = alpha * np.eye(X.shape[1])# alpha * I
    # 最小二乘求权重 w = (X^T X + A)^-1 (X^T y)
    w = np.matmul((np.linalg.inv(np.matmul(X.T, X) + A), np.matmul(X.T,y)))
    return w @ data
# lasso回归
def lasso(data):
    # 读入数据
    X, y = read_data()
    # 超参数
    alpha = 0.1
    epoch = 1000
    # 初始化w
    w = np.zeros(X.shape[1])
    # 梯度下降
    for i in range(epoch):
        # 梯度
        grad = np.matmul(X.T, np.matmul(X, w) - y) / X.shape[0] + alpha * np.sign(w)
        # 更新w
        w = w - grad
    return w @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

# read_data('../data/exp02/')