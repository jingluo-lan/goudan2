# 交叉验证：将拿到的训练数据，分为训练和验证集。
# 网格搜索：需要手动指定的参数称为超参数。但是手动过程繁琐，所以需要对模型预设几种超参数组合。每组超参数都采用交叉验证来进行评估。
# 最后选出最优参数组合建立模型


#导入模块
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#1.获取数据集
iris = load_iris()

#2.数据基本处理
#x_train,x_test,y_train,y_test为训练集特征值、测试集特征值、训练集目标值、测试机目标值
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2)

#3.特征工程：标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train) #fit相当于在计算均值标准差
x_test = transfer.transform(x_test)

#4.KNN预估器流程
#4.1实例化预估器类
estimator = KNeighborsClassifier()  #实例化一个估计器
#4.2模型选择与调优--网格搜索和交叉验证
#准备要调的超参数
param_dict = {"n_neighbors":[1,3,5]}
estimator = GridSearchCV(estimator,param_grid=param_dict,cv=3)  #cv=3，三折交叉验证
#4.3fit数据进行训练
estimator.fit(x_train,y_train)
#5.评估模型效果
#方法1：对比真实值和预期值
y_predict = estimator.predict(x_test)
print("对比真实值和预测值：\n",y_predict == y_test)
#方法2：直接计算准确率
score = estimator.score(x_test,y_test)
print("准确率为：\n",score)

#然后进行评估查看最终选择的结果和交叉验证的结果
print("在交叉验证中验证的最好结果：\n",estimator.best_score_)
print("最好的参数模型：\n",estimator.best_estimator_)
print("每次交叉验证后的准确率：\n",estimator.cv_results_)
