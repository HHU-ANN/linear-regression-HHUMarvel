# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
# 岭回归
def ridge(data):
    x, y = read_data()
    # w=(X^T X)^-1 (X^T y)
    weight = np.matmul((np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T,y)))
    return weight @ data
# lasso回归
def lasso(data):
    return ridge(data)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

read_data('../data/exp02/')