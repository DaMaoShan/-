#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random as rand

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X = []
Y = []

##################  以下注释掉的语句包含了其他情况，可以自行去掉注释，观察效果  ###########################
# for i in range(100000):   # square
#     X.append(rand.uniform(-10, 10))
#     Y.append(rand.uniform(-10, 10))

# for i in range(100000):   # circle
#     x = rand.uniform(-10, 10)
#     y = rand.uniform(-10, 10)
#     if x**2 + y**2 <= 100:
#         X.append(x)
#         Y.append(y)

# for i in range(100000):   # rectangle
#     x = rand.uniform(-10, 10)
#     y = rand.uniform(-100, 100)
#     X.append(x)
#     Y.append(y)

# NUM = 100
# for i in range(NUM):   # other
#     x = rand.uniform(-10, 10)
#     y = rand.uniform(-10, 10)
#     X.append(x)
#     Y.append(y)

# X1 = []
# Y1 = []
# for i in range(NUM):
#     Y1.append(Y[i])
#     X1.append(Y1[i]*(1-(Y[i])/10))
#     # X1.append(Y[i])

# D = np.array([X1, Y1])
# D = D.T
# pca = PCA()

# pca.fit(D)
# new_axis = pca.components_

# print(new_axis)

# print(pca.explained_variance_)  # 特征值
# print(pca.explained_variance_ratio_)  # 特征值标准化

# [0.9924... 0.0075...]

NUM = 1000
for i in range(NUM):   # 椭圆情形 
    x = rand.uniform(-10, 10)
    y = rand.uniform(-10, 10)
    d1 = np.sqrt((x-6.5)**2 + (y-6.5)**2)
    d2 = np.sqrt((x+6.5)**2 + (y+6.5)**2)
    if d1 + d2 <= 20:
        X.append(x)
        Y.append(y) 

m1 = np.mean(X)
m2 = np.mean(Y)
M = np.array([m1, m2]).reshape(2, 1)       
D = np.array([X, Y])
D = D - M
D = D.T

pca = PCA()
pca.fit(D)
new_axis = pca.components_

print('新座标轴向量,即特征向量：\n', new_axis[0, :], '\n', new_axis[1, :])
print('对应的特征值:\n', pca.explained_variance_)  
print('协方差矩阵：\n', np.dot(D.T, D), '\n')
D1 = np.dot(new_axis, D.T)  # 新坐标系中的坐标(数据)
print('协方差矩阵：\n', np.dot(D1, D1.T), '\n')

axes = plt.gca()
axes.set_xlim([-5, 5])
axes.set_ylim([-5, 5])

plt.scatter(X1, Y1, c='gray')

plt.plot([0, 12*new_axis[0, 0]], [0, 12*new_axis[0, 1]], color ='r')  # 新坐标轴1
plt.plot([0, 12*new_axis[1, 0]], [0, 12*new_axis[1, 1]], color = 'b')  # 新坐标轴2
plt.axis('equal')   # 坐标刻度一致

plt.show()

