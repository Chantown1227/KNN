# -*- coding = utf-8 -*-
# @time:2020/9/7 22:28
# Author:TC
# @File:马绞痛.py
# @Software:PyCharm

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer #KNN数据空值填充
from sklearn.metrics.pairwise import nan_euclidean_distances #计算带有空值的欧式距离
from sklearn.model_selection import cross_val_score #交叉验证

# KFold默认是不shuffle的，则切割的时候数据永远都是按照原始数据的顺序切割的
# StratifiedKFold分层抽样
# sklearn不放回抽样，每次采样的测试集样本都不同
from sklearn.model_selection import RepeatedStratifiedKFold #重复K折n次，当需要运行KFoldn次，每次重复产生不同的分割时，可以使用
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #train_test_split默认会对原始数据进行shuffle
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv('horse-colic.csv',header=None,na_values='?') #把问号改为NaN
print(df.head())
print(df.shape)

data=df.values #数据的第23列表示是否病变
x_index=[i for i in range(data.shape[1]) if i != 23]
x,y=data[:,x_index],data[:,23]
print(x.shape,y.shape)

cols_null=[]
for i in range(x.shape[1]):
    cols_null.append(df[i].isnull().sum()) #每一列数据缺失个数
print(cols_null)

# imputer=KNNImputer()
# x1=imputer.fit_transform(x)
# print(sum(np.isnan(x1)))

clf=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')

pipe=Pipeline(steps=[('imputer',KNNImputer(n_neighbors=5)),('model',clf)]) #clf不需要括号

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

pipe.fit(x_train,y_train)
score=pipe.score(x_test,y_test)
print(score)
