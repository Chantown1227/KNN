# -*- coding = utf-8 -*-
# @time:2020/9/7 22:27
# Author:TC
# @File:鸢尾花.py
# @Software:PyCharm

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
# print(iris)
# print(type(iris))
x=iris['data']
y=iris['target']
print(x.shape,y.shape)
# print(x)
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

clf=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
clf.fit(x_train,y_train)

y_predict=clf.predict(x_test)
print(y_predict)
y_predict.shape

acc=sum(y_predict==y_test)/y_test.shape[0]
print(acc)