#导包
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
#1.数据预处理
#1.1 加载数据
df_ttn_train = pd.read_csv('./data/titanic_train.csv')
df_ttn_test = pd.read_csv('./data/titanic_test.csv')

#2. 特征工程
#2.1 特征提取
x = df_ttn_train[['Pclass','Sex','Age']]
y = df_ttn_train['Survived']
#2.3 Age 列有缺失, 用平均值填充Age
# 会报警告, 没办法直接修改原数据, 需要先对原数据进行copy操作
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())
#2.4 对sex列使用热处理操作
x = pd.get_dummies(x,columns = ['Sex'])
x.drop('Sex_male',inplace = True,axis = 1)
#2.5 划分训练集和测试局
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#3. 模型训练
#3.1 创建模型对象
estimator = DecisionTreeClassifier(max_depth=10)
#3.2 模型训练
estimator.fit(x_train,y_train)
#4. 模型预测
y_pred = estimator.predict(x_test)
print(f'预测值为{y_pred}')

#5. 模型评估
print(f'模型分类评估报告{classification_report(y_test,y_pred)}')

#绘制树形图
plt.figure(figsize=(90,60))#设置图片大小
plot_tree(estimator,filled = True,max_depth=10)
plt.savefig('./data/my_titanic.png')
plt.show()
