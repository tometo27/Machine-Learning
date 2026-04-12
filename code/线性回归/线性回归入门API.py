"""
	线性回归
		利用线性回归方程对一个或多个变量和因变量之间的关系进行建模的一种分析方式
		一元线性回归: y = kx + b
		多元线性回归: y = wT + b

		损失函数 : 衡量每个样本预测值与真实值效果的函数, 也叫代价函数, 成本函数, 目标函数
            误差 = 预测值 - 真实值
"""
from sklearn.linear_model import LinearRegression
# 线性回归API入门

#1.准备数据
x_train = [[160],[166],[172],[174],[180]]
y_train = [56.3,60.6,65.1,68.5,75]
x_test = [[176]]
#2.数据预处理(这里不需要)
#3.特征工程(这里不需要)
#4.模型训练
estimator = LinearRegression()
estimator.fit(x_train,y_train)
#查看斜率,截距
print(f'权重{estimator.coef_}')
print(f'偏置{estimator.intercept_}')
#5.模型预测
y_pre = estimator.predict(x_test)
print(f'预测值{y_pre}'
      f'')