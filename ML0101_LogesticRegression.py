# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:56:56 2020

@author: Mercedeh_Mgh
"""
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

churn_df=pd.read_csv(r"https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv")

print(churn_df.head())
print(churn_df.columns)

# we need to change target data type to integer so it can be used with skitlearn algorithm

churn_df=churn_df[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]
churn_df['churn']=churn_df['churn'].astype('int')
print(churn_df.head())
print(churn_df['churn'].unique())
print(churn_df.shape)

# defining X and y for our dataset
X=np.asarray(churn_df[['tenure','age','address','income','ed','employ','equip']])
print(X[0:5])
y=np.asarray(churn_df['churn'])
print(y[0:5])

# normalize the dataset
X=preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

# define the train and test datasets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
print('Train set:',X_train.shape,y_train.shape)
print('Test shape:',X_test.shape,y_test.shape)

# fit a logestic regression model on our train set. C in LogesticRegression is for defining regularization to handle overfitting of models. Smaller C indicates stronger regularization. Solver is to determine numerical optimizer type.
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train)
print(log_reg)

# predication using the test set
yhat=log_reg.predict(X_test)
print(yhat)

# calculate predicted probability for each class of 1 and 0 of yhat (first column is probability of 1 and second column is for 0)
yhat_prob=log_reg.predict_proba(X_test)
print(yhat_prob)

# evaluate accuracy of the logestic regression model by using Jaccard index: the size of the intersection divided by the size of the union of two labels

# from sklearn.metrics import jaccard_similarity_score
# print(jaccard_similarity_score(y_test, yhat))
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat))
# another way to evaluate accuracy is by using confusion matrix (cm)

from sklearn.metrics import confusion_matrix, classification_report
import itertools

# define a fuction to print and plot a confusion matrix
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix', cmap=plt.cm.Blues):
	if normalize:
		cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
		print('Normlizated Confusion Matrix')
	else:
		print('Confusion Matrix, without normalization')
	print(cm)
	
	plt.imshow(cm, interpolation='nearest',cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks,classes,rotation=45)
	plt.yticks(tick_marks,classes)
	fmt=' .2f' if normalize else 'd'
	thresh=cm.max()/2.
	for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i,format(cm[i,j],fmt), 
		   horizontalalignment='center',
		   color='white' if cm[i,j] > thresh else 'black')
	plt.tight_layout()
	plt.ylabel('true label')
	plt.xlabel('predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))
		
# compute confusion matrix
cnf_matrix=confusion_matrix(y_test, yhat,labels=[1,0])
np.set_printoptions(precision=2)

# plotting non-normalized confusion matrix
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize=False,  title='Confusion matrix')

print(classification_report(y_test, yhat))


# Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)

# Recall is true positive rate. It is defined as: Recall = TP / (TP + FN)


# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.

# We can also measure performance of logestic regression classifier by calculating the log loss (probability of customer churn being yes/1):

from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))

