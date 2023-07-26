
#导入模块
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#先从sklearn中获取数据集，然后进行数据集的分割
#1.获取数据集
iris = load_iris()

#2.数据基本处理
#x_train,x_test,y_train,y_test为训练集特征值、测试集特征值、训练集目标值、测试机目标值
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2)

#进行数据标准化
#特征值的标准化

#3.特征工程：标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train) #fit相当于在计算均值标准差
x_test = transfer.transform(x_test)

#模型进行训练预测

#4.机器学习（模型训练）
estimator = KNeighborsClassifier(n_neighbors=5)  #实例化一个估计器
estimator.fit(x_train,y_train)
#5.模型评估
#方法1：对比真实值和预期值
y_predict = estimator.predict(x_test)
print("预测结果为：\n",y_predict)
print("对比真实值和预测值：\n",y_predict == y_test)
#方法2：直接计算准确率
score = estimator.score(x_test,y_test)
print("准确率为：\n",score)