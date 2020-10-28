# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:39:38 2020

@author: Mercedeh_Mgh
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df=pd.read_csv(r"C:\Users\tutu_\OneDrive\Desktop\Data_Science\IBM_ML\FuelConsumption.csv")

cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='purple')
#plt.xlabel('ENGINESIZE')
#plt.ylabel('CO2EMISSIONS')

msk=np.random.rand(len(df)) <0.8
train=cdf[msk]
test=cdf[~msk]

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='red')
plt.xlabel('Enginesize')
plt.ylabel('Emissions')

from sklearn import linear_model
regr=linear_model.LinearRegression()
#x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
#y=np.asanyarray(train[['CO2EMISSIONS']])
#
#regr.fit(x,y)
#print('coefficients: ',regr.coef_)
#print('intercept: ',regr.intercept_)
#
#y_hat=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
#x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
#y=np.asanyarray(test[['CO2EMISSIONS']])
#print('residual sum of squares: %.2f' % np.mean((y_hat-y)**2 ))
#print('variance score: %.2f' % regr.score(x,y))

x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print('coefficients: ', regr.coef_)
print('inercept: ',regr.intercept_)

y_hat=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y=np.asanyarray(test[['CO2EMISSIONS']])

print('residual sum of squares: %.2f' % np.mean((y_hat-y)**2))
print('variance score: %.2f' % regr.score(x,y))
