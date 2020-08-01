import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import random

style.use('fivethirtyeight')

df1 = pd.read_csv('datasets/breast-cancer-wisconsin.data' )
df = pd.DataFrame(df1)

# df.drop(index=df.index[69], inplace=True)
df.replace('?', -99999, inplace=True)   #to establish thm as outliers

X = np.array(df.drop('class',1))
y = np.array(df['class'])

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
sd = np.array([[4,1,1,1,2,1,1,2,2,1]])
# or we can do sd.reshape(1,-1)


pred = clf.predict(sd)
print(pred)
