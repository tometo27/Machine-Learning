from sklearn.linear_model import LinearRegression   #正规方程的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估
from sklearn.preprocessing import StandardScaler    #特征处理
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.linear_model import SGDRegressor   #梯度下降的回归模型    import pandas as pd
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge,RidgeCV

"""
1. 准备数据
2. 数据处理
3. 特征工程
4. 模型训练
5. 模型预测
6. 模型评估
"""

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#数据预处理

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=412)

# 创建标准化对象
transfer = StandardScaler()
#队训练集进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#创建线性回归正规方程模型对象
estimator  = LinearRegression(fit_intercept=True) #fit_intercept 是否需要截距
estimator.fit(x_train, y_train)
#打印模型计算出的权重和偏置
print(estimator.coef_)
print(estimator.intercept_)

#模型预测
y_pre = estimator.predict(x_test)
# print(y_pre)

#模型评估
print(mean_squared_error(y_test, y_pre))#均方误差
print(root_mean_squared_error(y_test, y_pre))#均方根误差
print(mean_absolute_error(y_test, y_pre))#绝对值误差