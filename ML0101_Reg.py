# -*- coding: utf-8 -*-
"""
Created on  Tues Jan 21 12:14:36 2020

@author: Mercedeh_Mgh
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv(r'C:\Users\Mercedeh_Mgh\OneDrive\Desktop\Data_Science\IBM_ML\FuelConsumption.csv')

cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#df.describe()
#print(cdf.head(9))
#cdf.hist()
#plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,color='blue')
#plt.xlabel("FUELCONSUMPTION_COMB")
#plt.ylabel("CO2EMISSIONS")
#plt.show()
#plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='purple')
#plt.xlabel("ENGINESIZE")
#plt.ylabel("CO2EMISSIONS")
msk=np.random.rand(len(df)) <0.8
train= cdf[msk]
test=cdf[~msk]

#plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='pink')
#plt.xlabel("ENGINESIZE")
#plt.ylabel("CO2EMISSIONS")
#

from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])

regr.fit(train_x,train_y)
print('coefficient: ',regr.coef_)
print('Intercept: ',regr.intercept_)

#plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color='green')
#plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0], '-r')
#plt.xlabel("Engine size")
#plt.ylabel("Emission")

from sklearn.metrics import r2_score
test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])
test_y_hat=regr.predict(test_x)

print ("Mean absolute error: %.2f"% np.mean(np.absolute(test_y_hat-test_y)))
print("Residual sum of square: %.2f"%np.mean((test_y_hat-test_y)**2))
print("r2_score: %.2f"%r2_score(test_y_hat,test_y))






