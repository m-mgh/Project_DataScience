# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 01:12:36 2020

@author: Mercedeh_Mgh
"""

import pandas as pd
import numpy as np
import pylab as pl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.optimize as optimize
import matplotlib.pyplot as plt


cell_df=pd.read_csv(r'C:\Users\Mercedeh_Mgh\OneDrive\Desktop\Data_Science\IBM_ML\cell_samples.csv')
print(cell_df.shape)
print(cell_df.head())
print(cell_df.columns)
print(cell_df['Clump'].unique())
print(cell_df['Class'].unique())


# visualization of a section of data
vis=cell_df[cell_df['Class']==4][0:50].plot(kind='scatter',x='Clump', y='UnifSize', color='DarkBlue',label='mlignant');
cell_df[cell_df['Class']==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize', label='benign',color='yellow',ax=vis)

print(cell_df.dtypes)
print(cell_df['BareNuc'].unique())
# BareNuc has some non-numerical values; we drop them
cell_df=cell_df[pd.to_numeric(cell_df['BareNuc'],errors='coerce').notnull()]
print(cell_df['BareNuc'].unique())
cell_df['BareNuc']=cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

feature_df=cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X=np.asarray(feature_df)
print(X[0:5])

cell_df['Class']=cell_df['Class'].astype('int')
y=np.asarray(cell_df['Class'])
print(y[0:10])

# define train/test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
print('train set:',X_train.shape,y_train.shape)
print('test set:',X_test.shape,y_test.shape)

#There are various Kernel functions to try to transform linear data into multidimentional space. Here we try RBF (radial basis function) only, but usually we should try various functions and pick the one that works the best

from sklearn import svm
clf=svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

y_hat=clf.predict(X_test)
print(y_hat[0:5])

#evaluation
from sklearn.metrics import classification_report,confusion_matrix
import itertools

#This function prints and plots the confusion matrix.Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
	if normalize:
		cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
		print('normalized confusion matrix')
	else:
		print('confusion matrix')
	print(cm)
	plt.imshow(cm,interpolation='nearest',cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks,classes,rotation=45)
	plt.yticks(tick_marks,classes)
	
	fmt='.2f' if normalize else 'd'
	thresh=cm.max()/2
	for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j, i, format(cm[i,j],fmt), 
		   horizontalalignment='center',
		   color='white' if  cm[i,j] >thresh else 'black')
	plt.tight_layout()
	plt.ylabel('true label')
	plt.xlabel('predicted label')
	
#compute the confusion matrix
cnf_matrix=confusion_matrix(y_test, y_hat,labels=[2,4])
np.set_printoptions(precision=2)
print(classification_report(y_test, y_hat))
#plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['benign(2)','malignent(4)'],normalize=False,title='confusion matrix')	

#evaluate using f1 score (balance between precision and recall or 2*((precision*recall)/(precision+recall)))
from sklearn.metrics import f1_score
print(f1_score(y_test, y_hat,average='weighted'))

#evalute using jaccard index
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test,y_hat,pos_label=2))

#we try linear_kernel function this time for kernelling

clf2=svm.SVC(kernel='linear')
clf2.fit(X_train,y_train)

y_hat2=clf2.predict(X_test)
print(y_hat2[0:5])

cnf2_matrix=confusion_matrix(y_test, y_hat2,labels=[2,4])
print(cnf2_matrix)

print(f1_score(y_test, y_hat2, average='weighted'))

print(jaccard_score(y_test,y_hat2,pos_label=2))	

   
   
	
	







