import neurolab as nl
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('data/dcb_data.xlsx')
# print(df)

# 按行读取
# for index, row in df.iterrows():
#     print(row['n1'])

# 取前六列
df_num = df.iloc[:, 0:-1]
# print(df_num)
print(df_num[:1500])
print(df_num[1500:])
# 转为二维数组
train_nums = df_num[0:1500].to_numpy()
test_nums = df_num[1500:1782].to_numpy()
print(train_nums)
print(test_nums)




# 生成训练数据
