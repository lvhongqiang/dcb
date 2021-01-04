import neurolab as nl
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

min_value = 1
max_value = 32

df = pd.read_excel('data/dcb_data.xlsx')
# print(df)

# 取前六列
df_num = df.iloc[:, 0:-1]
# print(df_num)
print(df_num[:1500])
print(df_num[1500:])
# 转为二维数组
train_nums = np.hstack((df_num[0:1500].to_numpy(), df_num[1:1501].to_numpy(), df_num[2:1502].to_numpy()))
train_out = df_num[3:1503].to_numpy()
test_nums = np.hstack((df_num[1500:1779].to_numpy(), df_num[1501:1780].to_numpy(), df_num[1502:1781].to_numpy()))
test_out = df_num[1503:1782].to_numpy()

print(train_nums[0])
# print(test_nums)


# 定义一个深度神经网络，带有两个隐藏层，每个隐藏层由10个神经元组成，输出层由一个神经元组成
multilayer_net = nl.net.newff([[min_value, max_value]], [30, 30, 10, 10, 1])

# 设置训练算法为梯度下降法
multilayer_net.trainf = nl.train.train_gd

# 训练网络
error = multilayer_net.train(data, labels, epochs=800, show=100, goal=0.01)

# 用训练数据运行该网络，预测结果
predicted_output = multilayer_net.sim(data)


# 生成训练数据
