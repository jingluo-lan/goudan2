#california房价预测(波士顿房价数据集已从sklearn中移除）

#导入需要的API
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.datasets import fetch_california_housing
import pandas as pd

# 获取加利福尼亚房价数据集
data = fetch_california_housing()

# 获取特征矩阵
X = data.data

# 获取目标向量
y = data.target

# 将特征矩阵和目标向量转换为DataFrame
df = pd.DataFrame(data=X, columns=data.feature_names)
df['target'] = y

# 输出前5行数据
print(df.head())

def linear_model():
    # 1.获取数据
    california = fetch_california_housing()
    print(california)
    # 2.数据基本处理
    # 2.1分割数据
    # 3.特征工程-标准化
    # 4.机器学习-线性回归
    # 5.模型评估




#1.获取数据
#2.数据基本处理
#2.1分割数据
#3.特征工程-标准化
#4.机器学习-线性回归
#5.模型评估

linear_model()