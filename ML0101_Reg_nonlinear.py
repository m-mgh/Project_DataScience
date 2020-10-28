# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 00:04:25 2020

@author: Mercedeh_Mgh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv( r"C:\Users\Mercedeh_Mgh\OneDrive\Desktop\Data_Science\IBM_ML\china_gdp.csv")

df.head()
plt.figure(figsize=(8,5))
x_data,y_data=(df["Year"].values,df["Value"].values)
plt.plot(x_data,y_data,"ro")
plt.xlabel("year")
plt.ylabel("GDP")
plt.show()

def sigmoid(x, Beta1, Beta2):
	y=1/(1+np.exp(-Beta1*(x-Beta2)))
	return y

#beta_1=0.1
#beta_2=1990
#
#y_predict=sigmoid(x_data,beta_1,beta_2)
#plt.plot(x_data,y_predict*15000000000000)
#plt.plot(x_data,y_data,"ro")
#
##normalize data
#x_norm=x_data/max(x_data)
#y_norm=y_data/max(y_data)

#from scipy.optimize import curve_fit
#popt, pcov=curve_fit(sigmoid,x_norm,y_norm)
#print("beta_1=%f, beta_2=%f" %(popt[0],popt[1]))

#x=np.linspace(1960,2015,55)
#x=x/max(x)
#plt.figure(figsize=(8,5))
#y=sigmoid(x,*popt)
#plt.plot(x_norm,y_norm,"ro",label="data")
#plt.plot(x,y,linewidth=3.0,label="fit")
#plt.legend(loc="best")
#plt.ylabel("GDP")
#plt.xlabel("year")
#plt.show()

#split data for training and testing
msk=np.random.rand(len(df))<0.8
train_x=x_norm[msk]
train_y=y_norm[msk]
test_x=x_norm[~msk]
test_y=y_norm[~msk]

#build the model using train set
from scipy.optimize import curve_fit
popt, pcov=curve_fit(sigmoid,train_x, train_y)
#test the predictedive model
y_hat=sigmoid(test_x,*popt)
plt.plot(x_norm,y_norm,"ro",label="data")
plt.plot(test_x,y_hat,linewidth=3,label="fit")
#evaluate model's accuracy
print("Residual sum of squares: %.2f" % np.mean((y_hat-test_y)**2))
print('Mean absolute error:%.2f' %np.mean(y_hat-test_y))
from sklearn.metrics import r2_score
print("R2-score:%.2f" % r2_score(test_y,y_hat))










