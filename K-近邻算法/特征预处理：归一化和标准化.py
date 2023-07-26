import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def minmax_demo():
    """
    归一化演示
    :return: None
    """
    data = pd.read_csv("C:/Users/Lenovo/Desktop/dating.txt")
    print(data)
    #1.实例化
    transfer = MinMaxScaler(feature_range=(3,5))
    #2.进行转换
    ret_date = transfer.fit_transform(data[["milage","Liters","Consumtime"]])
    print("归一化后的数据为：\n",ret_date)

minmax_demo()

def standard_demo():
    """
    归一化演示
    :return: None
    """
    data = pd.read_csv("C:/Users/Lenovo/Desktop/dating.txt")
    print(data)
    #1.实例化
    transfer = StandardScaler()
    #2.进行转换
    ret_date = transfer.fit_transform(data[["milage","Liters","Consumtime"]])
    print("标准化后的数据为：\n",ret_date)
    print("每一列的方差为：\n", transfer.var_)
    print("每一列的平均值为：\n", transfer.mean_)

standard_demo()