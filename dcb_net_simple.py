import neurolab as nl
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

min_value = -15
max_value = 15

df = pd.read_excel('data/dcb_data.xlsx')
# print(df)

# 取前六列
df_num = df.iloc[:, 0:1]
# print(df_num)
print(df_num[:1500])
print(df_num[1500:])
# 转为二维数组
# train_nums = np.hstack((df_num[0:1500].to_numpy(), df_num[1:1501].to_numpy(), df_num[2:1502].to_numpy()))
# train_out = df_num[3:1503].to_numpy()/100
# train_nums = df_num[:1500].to_numpy()
# train_out = df_num[1:1501].to_numpy()/32

train_nums = np.linspace(min_value, max_value, 100).reshape(100, 1)
train_out = (2 * np.square(train_nums)+7).reshape(100, 1)
train_out = train_out/3000

print(train_nums)
print(train_out)

# test_nums = np.hstack((df_num[1500:1779].to_numpy(), df_num[1501:1780].to_numpy(), df_num[1502:1781].to_numpy()))
# test_out = df_num[1503:1782].to_numpy()
# test_nums = df_num[1501:1781].to_numpy()
# test_out = df_num[1502:1781].to_numpy()/32
test_nums = np.linspace(min_value, max_value, 300).reshape(300, 1)


# print(test_nums)


# 定义一个深度神经网络，带有两个隐藏层，每个隐藏层由10个神经元组成，输出层由一个神经元组成
input_layer = np.ndarray((1, 2), dtype=np.int16)
input_layer[:] = [min_value, max_value]

multilayer_net = nl.net.newff(input_layer, [30, 30, 1])

# 设置训练算法为梯度下降法
multilayer_net.trainf = nl.train.train_gd

# 训练网络
error = multilayer_net.train(train_nums, train_out, epochs=100, show=20, goal=0.01)

# 用训练数据运行该网络，预测结果
predicted_output = multilayer_net.sim(test_nums)

# 画出训练误差结果
plt.figure()
plt.plot(error)
plt.xlabel('Number of epoches')
plt.ylabel('Error')
plt.title('Training error progress')
plt.show()

print(predicted_output)
# print(predicted_output*32)
# print(test_out)
