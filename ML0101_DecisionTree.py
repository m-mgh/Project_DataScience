# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:34:09 2020

@author: Mercedeh_Mgh
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

df=pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv", delimiter=",")
print(df[0:5])
print(df.shape)
print(df.columns)
#preprocess data
X=df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])
y=df[['Drug']]

#sklearn does not handle categorical features in decision trees and we need to convert these features to numerical first

print(df['Sex'].unique())

le_sex=preprocessing.LabelEncoder()
le_sex.fit(["F","M"])
X[:,1]=le_sex.transform(X[:,1])
   

print(df['BP'].unique())
le_BP=preprocessing.LabelEncoder()
le_BP.fit(['HIGH','LOW','NORMAL'])
X[:,2]=le_BP.transform(X[:,2])


print(df['Cholesterol'].unique())
le_Cholesterol=preprocessing.LabelEncoder()
le_Cholesterol.fit(['HIGH','NORMAL'])
X[:,3]=le_Cholesterol.transform(X[:,3])
print(X[0:5])
print(y[0:5])

X_trainset, X_testset,y_trainset,y_testset=train_test_split(X,y,test_size=0.3,random_state=3)

print(X_trainset.shape)
print(y_trainset.shape)
print(X_testset.shape)
print(y_testset.shape)

#Modeling a decision tree
DrugTree=DecisionTreeClassifier(criterion="entropy",max_depth=4)
print(DrugTree)

DrugTree.fit(X_trainset,y_trainset)

#prediction based on the model
predTree=DrugTree.predict(X_testset)
print(predTree)
print(y_testset)
  
#Evaluation
print("DecisionTree's accuracy: ",metrics.accuracy_score(y_testset,predTree))

#visualizatio
# need to install the pydotplus and graphviz libraries if you have not installed these before. To install Graphiz you also need to install the package directly from the website on the PC then add the location to the path. Call it in the code if dont want to permanently add to path
import os     
os.environ["PATH"] += os.pathsep + ''
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'
#Text-IO used in string IO is a library for creating Java console applications. It can be used in applications that need to read interactive input from the user.
dot_data=StringIO()
filename="drugtree.png"
featurenames=df.columns[0:5]
targetnames=df["Drug"].unique().tolist()
out=tree.export_graphviz(DrugTree,feature_names=featurenames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')






