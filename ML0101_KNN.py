# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:34:28 2020

@author: Mercedeh Mohaghegh
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing

df=pd.read_csv(r'C:\Users\tutu_\OneDrive\Desktop\Data_Science\IBM_ML\teleCust1000t.csv')
print(df.head(),  
df.columns.tolist(),
df['custcat'].value_counts())

#df.hist(column='income', bins=50)

#define feature set
df.columns
#convert pandas dataframe to numpy array to be able to use scikit-learn

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values

print(X[0:5])

#define labels

y = df['custcat'].values
print(y[0:5])

#Normalize/scale data by getting mean and std using fit then scaling to unit variance and removing mean using standardscaler and transform
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#split data to train and test samples
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=4)
print('Train set:',X_train.shape,y_train.shape)
print('Test set:',X_test.shape,y_test.shape)

#KNN-K Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#training the model (looping through k values from 1 to 10 to pick the most accuract model)

k=10
mean_accuracy=np.zeros((k-1))
std_accuracy=np.zeros((k-1))
ConfusionMx=[];

for n in range(1,k):
	neigh=KNeighborsClassifier(n_neighbors= n).fit(X_train, y_train)

#predicting
	yhat=neigh.predict(X_test)

#accuracy
	mean_accuracy[n-1]=metrics.accuracy_score(y_test,yhat)
	std_accuracy[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
print(mean_accuracy)
	

print ("Train set Accuracy:",metrics.accuracy_score(y_train,neigh.predict(X_train)))

print("Test set accuracy:",metrics.accuracy_score(y_test,yhat))

#plot model accuracy for various ks

plt.plot(range(1,k),mean_accuracy,'g')
plt.fill_between(range(1,k),mean_accuracy-1*std_accuracy,mean_accuracy+1*std_accuracy,alpha=0.10)
plt.tight_layout()
print("the best accuracy was with:",mean_accuracy.max(),"with k=",mean_accuracy.argmax()+1)













