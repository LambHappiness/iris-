#环境
conda env list

#导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

print(iris_df.head())

print("数据集的维度:",iris_df.shape)

iris_df.info()
print(iris_df.describe())print(iris_df.head())

#直方图
iris_df.hist(figsize=(12,8))
plt.tight_layout()
plt.show()

#箱线图
plt.figure(figsize=(12,6))
sns.boxplot(data=iris_df)
plt.tight_layout()
plt.show()

plt.savefig('boxplot_features.png')

#热力分布图
correlation_matrix = iris_df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',square=True)
plt.show()

#散点图
sns.pairplot(iris_df,hue="species", height=3)
