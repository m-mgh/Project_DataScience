# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:38:06 2020

@author: tutu_
"""
#DBSCAN stands for Density Based Spatial Clusteringof Applications with Noise
#The wonderful attribute of DBSCAN algorithm is that it can find out any arbitrary shape cluster without getting affected by noise.

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#define a function to generate the data points with three type of inputs (centroidLocation, numSamples and clusterDeviation)
def createDataPoints (centroidLocation,numSamples,clusterDeviation):
	#create random data points and store them in feature matrix X and response vector y
	X,y=make_blobs(n_samples=numSamples,centers=centroidLocation,cluster_std=clusterDeviation)
	#standardize features by removing mean and scaling to unit variance
	X=StandardScaler().fit_transform(X)
	return X,y

#create variables X and y using the function
X,y=createDataPoints([[4,3],[2,-1],[-1,4]],1500,0.5)

#DBSCAN works based on two parameter: Epsilon and Minimum Samples. 
#Epsilon: determines a specified radius that if includes enough number of points we call it a dense area
#Minimum Samples:minimum number of points we want in a neighborhood to define a cluster

epsilon=0.3
minimumSamples=7
db=DBSCAN(eps=epsilon,min_samples=minimumSamples).fit(X)

labels=db.labels_
print(labels)

#next detecting outliers
#create an array of booleans using the labels from db
core_samples_mask=np.zeros_like(db.labels_,dtype=bool)
core_samples_mask[db.core_sample_indices_]=True
print(core_samples_mask)

#get the number of clusters in labels ignoring noise if present
n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
print(n_clusters_)

#get unique labels
unique_labels=set(labels)
print(unique_labels)

#data visualization
#create colors for clusters
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
#plot the points with colors
for k,col in zip(unique_labels,colors):
	if k==-1:
		#use black for noise
		col='k'
	
	class_member_mask=(labels==k)
	#plot the datapoints that are clustered
	xy=X[class_member_mask & core_samples_mask]
	plt.scatter(xy[:,0],xy[:,1],s=50, c=[col],marker=u'o',alpha=0.5)
	#plot the outliers
	xy=X[class_member_mask & ~core_samples_mask]
	plt.scatter(xy[:,0],xy[:,1],s=50,c=[col],marker=u'o',alpha=0.5)

#compare the clusters created via DBSCAN and the ones via KMeans
from sklearn.cluster import KMeans 
k=3
k_means=KMeans(n_clusters=k,init="k-means++",n_init=12)
k_means.fit(X)
fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(1,1,1)
for k,col in zip(range(k),colors):
    my_members=(k_means.labels_==k)
    plt.scatter(X[my_members,0],X[my_members,1],c=col,marker=u'o',alpha=0.5)

plt.show

#try DBSCAN on real data

db=pd.read_csv(r'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv')
print(db.head(5))
print(db.columns)
print(db['Tm'].unique())

#remove null values from 'Tm'
db=db[pd.notnull(db["Tm"])]
db=db.reset_index(drop='True')
print(db.head(5))

#Visualization of stations on map using basemap package of matplotlib (initially the interactive plot did not work, needed to change ipython consol graphic backend to automatic)
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize']=(14,10)

llon=-140
ulon=-50
llat=40
ulat=65

db=db[(db['Long']>llon) & (db['Long']<ulon) & (db['Lat']> llat) & (db['Lat']<ulat)]
my_map=Basemap(projection='merc',resolution='c',area_thresh=1000.0,llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=ulon,urcrnrlat=ulat)
my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundiry()
my_map.fillcontinents(color='white',alpha=0.3)
my_map.shadedrelief()

#to collect data based on stations
xs,ys=my_map(np.asarray(db.Long),np.asarray(db.Lat))
db['xm']=xs.tolist()
db['ym']=ys.tolist()

#visualization1
# for index,row in db.iterrows():
# 	my_map.plot(row.xm,row.ym,markerfacecolor=([1,0,0]), marker='o',markersize=5,alpha=0.75)
# plt.show()

#clustering of stations using an array of lon and lat (sklearn can also use a distance matrix)
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet=db[['xm','ym']]
Clus_dataSet=np.nan_to_num(Clus_dataSet)
Clus_dataSet=StandardScaler().fit_transform(Clus_dataSet)

#compute DBSCAN
dbp=DBSCAN(eps=0.15,min_samples=10)
dbp.fit(Clus_dataSet)
core_samples_mask=np.zeros_like(dbp.labels_,dtype=bool)
print(core_samples_mask)
core_samples_mask[dbp.core_sample_indices_]=True
labels=dbp.labels_
#add labels as a feature column to the original dataset
db['Clus_Db']=labels

#number of clusters considering outliers are not part of real clusters
realClusterNum=len(set(labels))-(1 if -1 in labels else 0)
clusterNum=len(set(labels))

#A sample of clusters
print(db[['Stn_Name','Tx','Tm','Clus_Db']].head(5))
#print all available labels including outliers which is -1
print(set(labels))

#visualization of clusters based on location
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize']=(14,10) 

my_map=Basemap(projection='merc',resolution='l',area_thresh=1000.0,llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=ulon,urcrnrlat=ulat)
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white',alpha=0.3)
my_map.shadedrelief()
#to create a color map
colors=plt.get_cmap('jet')(np.linspace(0.0,1.0,clusterNum))

#visualization1
for clust_number in set(labels):
	c=(([0.4,0.4,0.4]) if clust_number==-1 else colors[np.int(clust_number)])
	clust_set=db[db.Clus_Db==clust_number]
	my_map.scatter(clust_set.xm,clust_set.ym,color=c,s=20,alpha=0.85)
	if clust_number !=-1:
		cenx=np.mean(clust_set.xm)
		ceny=np.mean(clust_set.ym)
		plt.text(cenx,ceny,str(clust_number),fontsize=25,color='red')
		print("Cluster "+str(clust_number)+" ,Avg Temp: "+ str(np.mean(clust_set.Tm)))
	
#clustering based on location, mean,max, and min temperature (importing packages similar to above)
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)	
Clus_dataSet=db[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet=np.nan_to_num(Clus_dataSet)
Clus_dataSet=StandardScaler().fit_transform(Clus_dataSet)
#compute DBSCAN
dbp=DBSCAN(eps=0.3,min_samples=10).fit(Clus_dataSet)
core_samples_mask=np.zeros_like(dbp.labels_,dtype=bool)
core_samples_mask[dbp.core_sample_indices_]=True
labels=dbp.labels_
db["Clus_Db"]=labels
realClusterNum=len(set(labels))-(1 if -1 in labels else 0)
clusterNum=len(set(labels))
print(db[['Stn_Name','Tx','Tm','Clus_Db']].head(5))

#visualization of clusters based on location and temperature (importing similar packages from basemap and matplotlib as the previous vis)
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize']=(14,10)
my_map=Basemap(projection='merc',resolution='l',area_thresh=1000.0,llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=ulon,urcrnrlat=ulat)
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white',alpha=0.3)
my_map.shadedrelief()

#to create color map
colors=plt.get_cmap('jet')(np.linspace(0.0,1.0,clusterNum))
#visualization2
for clust_number in set(labels):
 	c=(([0.4,0.4,0.4]) if clust_number==-1 else colors[np.int(clust_number)])
 	clust_set=db[db.Clus_Db==clust_number]
 	my_map.scatter(clust_set.xm,clust_set.ym,color=c,marker='o',s=20,alpha=0.85)
 	if clust_number !=-1:
		 cenx=np.mean(clust_set.xm)
		 ceny=np.mean(clust_set.ym)
		 plt.text(cenx,ceny,str(clust_number),fontsize=25,color='red',)
		 print("Cluster "+str(clust_number)+', Avg Temp: '+str(np.mean(clust_set.Tm)))
		



