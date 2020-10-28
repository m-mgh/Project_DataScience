# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 00:06:22 2020

@author: Mercedeh_Mgh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl

df=pd.read_csv(r'C:\Users\Mercedeh_Mgh\OneDrive\Desktop\Data_Science\IBM_ML\FuelConsumption.csv')

cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

print(cdf.head())
pl.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='green')

pl.ylabel('Emission')
pl.xlabel('Engine size')

msk=np.random.rand(len(cdf)) <0.8
train=cdf[msk]
test=cdf[~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])

poly=PolynomialFeatures(degree=2)
train_x_poly=poly.fit_transform(train_x)
print(train_x_poly)

regr=linear_model.LinearRegression()
regr.fit(train_x_poly,train_y)
print('coefficient: ',regr.coef_)
print('intercept: ',regr.intercept_)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')

pl.xlabel('engine size')
pl.ylabel('co2emissions')
XX=np.arange(0.0,10.0,0.1)
yy=regr.intercept_[0]+regr.coef_[0][1]*XX+regr.coef_[0][2]*np.power(XX,2)

plt.plot(XX,yy,'-r')
plt.ylabel('co2emission')
plt.xlabel('engine size')

from sklearn.metrics import r2_score
test_x_poly=poly.fit_transform(test_x)
test_y_=regr.predict(test_x_poly)

print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
print('residual sum of squars (SME): %.2f' % np.mean(test_y_ - test_y)**2)
print ('r2_core: %.2f hello' % r2_score(test_y_, test_y))

poly = PolynomialFeatures(degree=3)
train_x_poly=poly.fit_transform(train_x)

clf=linear_model.LinearRegression()
train_ybar=clf.fit(train_x_poly,train_y)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.xlabel("enginesize")
plt.ylabel("co2emission")
XX=np.arange(0.0,10.0,0.1)
yy=clf.intercept_[0]+clf.coef_[0][1]*XX+clf.coef_[0][2]*np.power(XX,2)+clf.coef_[0][3]*np.power(XX,3)
plt.plot(XX,yy, '-r')

from sklearn.metrics import r2_score
test_x_poly=poly.fit_transform(test_x)
test_ybar=clf.predict(test_x_poly)

print("mean absolute error: %.2f" %np.mean(np.absolute(test_ybar-test_y)))
print("residual sum of squares: %.2f" %np.mean((test_y_ - test_y)**2))
print("r2 score: %.2f" %r2_score(test_y_,test_y))
