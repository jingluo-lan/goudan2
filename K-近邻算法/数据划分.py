
#设置显示中文
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
#设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


from sklearn.datasets import load_iris,fetch_20newsgroups


#获取小数据集
iris = load_iris()
# print(iris)

#获取大数据集
# news = fetch_20newsgroups()
# print(news)

#数据集属性描述
# print("数据集特征值是：\n",iris.data)
# print("数据集目标值是：\n",iris["target"])
# print("数据集的特征值名字是：\n",iris.feature_names)
# print("数据集的目标值名字是：\n",iris.target_names)
# print("数据集特征值是：\n",iris.DESCR)

#数据可视化
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#1.把数据转化为Dateframe的格式
iris_d = pd.DataFrame(data=iris.data,columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
iris_d['target'] = iris.target #添加一列目标值
print(iris_d)

#2.函数实现
def iris_plot(data,col1,col2):
    sns.lmplot(data=data,x=col1,y=col2,hue="target",fit_reg=False)
    plt.title("鸢尾花数据显示")
    plt.show()

iris_plot(iris_d,'Sepal_Width','Petal_Length')

#数据集划分
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
print("训练集的特征值是：\n",x_train)
print("训练集的目标值是：\n",y_train)
print("测试集的特征值是：\n",x_test)
print("测试集的目标值是：\n",y_test)

print("训练集的目标值的形状是：\n",y_train.shape)
print("测试集的目标值的形状是：\n",y_test.shape)