# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:59:04 2020

@author: Mercedeh_Mgh
"""

import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import numpy as np
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

#generate random data
# We will be generating a set of data using the make_blobs class.
# Input these parameters into make_blobs:
# n_samples: The total number of points equally divided among clusters. Choose a number from 10-1500
# centers: The number of centers to generate, or the fixed center locations. Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers (ex. centers=[[1,1], [2,5]])
# cluster_std: The standard deviation of the clusters. The larger the number, the further apart the clusters
# Choose a number between 0.5-1.5
X1,y1=make_blobs(n_samples=50,centers=[[4,4],[-2,-2],[1,1],[10,4]],cluster_std=0.9)

plt.scatter(X1[:,0],X1[:,1],marker='x')

#agglomerative clustering with randomly generated data
agglom=AgglomerativeClustering(n_clusters=4,linkage='complete')
# agglom=AgglomerativeClustering(n_clusters=4,linkage='average')
agglom.fit(X1,y1)

#plotting the clustering result
#create a fig of 4x6 inches
plt.figure(figsize=(6,4))
# These two lines of code are used to scale the data points down, Or else the data points will be scattered very far apart.
x_min,x_max=np.min(X1,axis=0),np.max(X1,axis=0)
#get the average distance for X1
X1=(X1-x_min)/(x_max-x_min)
#this loop displays all the data points
for i in range(X1.shape[0]):
	#Replace the data points with their respective cluster value (ex. 0) and is color coded with a colormap (plt.cm.spectral)
	plt.text(X1[i,0],X1[i,1],str(y1[i]),color=plt.cm.nipy_spectral(agglom.labels_[i]/10.),fontdict={'weight':'bold','size':9})
	
#remove the x and y ticks and xand y axis
plt.xticks([])
plt.yticks([])
plt.axis('off')
#display the plot of the original data before clustering
plt.scatter(X1[:,0],X1[:,1],marker='.')
#display the plot
plt.show()

#create dendrogram using scipy. first get the distance matrix
dis_matrix=distance_matrix(X1,X1)
print(dis_matrix)

#select the type of linkage in the hierarchy
Z=hierarchy.linkage(dis_matrix,'complete')
#draw a dendrogram to show the hierarchical clustering
dendro=hierarchy.dendrogram(Z)

#now try clustering on a real dataset
df=pd.read_csv(r'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv')

print(df.shape)
print(df.head(5))
print(df.columns)
print(df['manufact'].unique)

#data cleaning. drop categorical features and NAs
print("Shape of dataset before cleaning: ",df.size)
df[['sales', 'resale', 'type', 'price', 'engine_s',          'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt','fuel_cap','mpg','lnsales']]=df[['sales', 'resale', 'type', 'price', 'engine_s',          'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt','fuel_cap','mpg','lnsales']].apply(pd.to_numeric,errors='coerce')
df=df.dropna()
df=df.reset_index(drop=True)
print("Shape of dataset after cleaning: ",df.size)
print(df.head(5))

#select a feature set to include in the analysis
featureset=df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
#Normalize feature set by using min_max_scaler which scales each feature to a given range (default is between 0-1).
from sklearn.preprocessing import MinMaxScaler
#change feature set to a numpy array so it can be used in sklearn algorithm
x=featureset.values
print(featureset.head(5)) 
print(x[:,:2])
min_max_scaler=MinMaxScaler()
feature_mtx=min_max_scaler.fit_transform(x)
print(feature_mtx[0:5])

#clustering using scipy.First calculate distance matrix
import scipy
leng=feature_mtx.shape[0]
D=scipy.zeros([leng,leng])
for i in range (leng):
	for j in range (leng):
		D[i,j]=scipy.spatial.distance.euclidean(feature_mtx[i],feature_mtx[j])

#in agglomerative clustering, at each iteration, the algorithm must update the distance matrix to reflect the distance of the newly formed cluster with the remaining clusters in the forest. The following methods are supported in Scipy for calculating the distance between the newly formed cluster and certain points in clusters: - single - complete - average - weighted - centroid
import pylab
import scipy.cluster.hierarchy
Z=hierarchy.linkage(D, 'complete')
#Essentially, Hierarchical clustering does not require a pre-specified number of clusters. However, in some applications we want a partition of disjoint clusters just as in flat clustering. So you can use a cutting line
from scipy.cluster.hierarchy import fcluster
max_d=3
clusters=fcluster(Z, max_d, criterion='distance')
print(clusters)

#Also, you can set number of clusters in your data directly
from scipy.cluster.hierarchy import fcluster
k=5
clusters=fcluster(Z,k,criterion='maxclust')
print(clusters)

#now plot the dedrogram of the scipy cluster
fig=pylab.figure(figsize=(18,50))
def llf(id):
	return '[%s %s %s]' % (df['manufact'][id],df['model'][id],int(float(df['type'][id])))
dendro=hierarchy.dendrogram(Z,leaf_label_func=llf,leaf_rotation=0,leaf_font_size=12, orientation='right')

#clustering using scikitlearn package
dist_matrix=distance_matrix(feature_mtx,feature_mtx)
print(dist_matrix)
#The AgglomerativeClustering performs a hierarchical clustering using a bottom up approach. The linkage criteria determines the metric used for the merge strategy:

#Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
# Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
# Average linkage minimizes the average of the distances between all observations of pairs of clusters.
		
agglom=AgglomerativeClustering(n_clusters=6,linkage='complete')
agglom.fit(feature_mtx)
agglom.labels_

#add a new field in our dataframe to show the cluster of each row:
df['cluster_']=agglom.labels_
print(df.head())
print(df["cluster_"].unique())		

#plot the results
import matplotlib.cm as cm
n_clusters=max(agglom.labels_)+1
colors=cm.rainbow(np.linspace(0,1,n_clusters))
cluster_labels=list(range(0,n_clusters))
plt.figure(figsize=(16,14))
for color, label in zip(colors,cluster_labels):
	subset=df[df.cluster_== label]
	for i in subset.index:
		plt.text(subset.horsepow[i],subset.mpg[i],str(subset['model'][i]),rotation=25)
	plt.scatter(subset.horsepow,subset.mpg,s=subset.price*10,c=color,label='cluster'+str(label),alpha=0.5)
plt.legend()
plt.title('Cluster')	
plt.xlabel('horsepow')
plt.ylabel('mpg')

#there are 2 types of vehicles in our dataset, "truck" (value of 1 in the type column) and "car" (value of 2 in the type column). So, we use them to distinguish the classes, and summarize the cluster. First we count the number of cases in each group:
df.groupby(['cluster_','type'])['cluster_'].count()
#now look at characteristics of each cluster
agg_cars=df.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
print(agg_cars)


# It is obvious that we have 3 main clusters with the majority of vehicles in those.

# Cars:

# Cluster 1: with almost high mpg, and low in horsepower.
# Cluster 2: with good mpg and horsepower, but higher price than average.
# Cluster 3: with low mpg, high horsepower, highest price.
# Trucks:

# Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
# Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
# Cluster 3: with good mpg and horsepower, low price.
# Please notice that we did not use type , and price of cars in the clustering process, but Hierarchical clustering could forge the clusters and discriminate them with quite high accuracy.

plt.figure(figsize=(16,10))
for color, label in zip(colors,cluster_labels):
	subset=agg_cars.loc[(label,),]
	for i in subset.index:
		plt.text(subset.loc[i][0]+5,subset.loc[i][2],'type='+str(int(i))+', price='+str(int(subset.loc[i][3]))+'k')
	plt.scatter(subset.horsepow,subset.mpg,s=subset.price*20,c=color,label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepower')
plt.ylabel('mpg')

