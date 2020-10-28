# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:12:13 2020

@author: Mercedeh_Mgh
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#creating our own random data set
#first we need to creat a random seed
np.random.seed(0)

#generate random clusters of points. n_samples defines the number of points.centers is the number of centers to be generated or location of fixed centers.cluster_std is the standard deviation of the clusters.
X, y=make_blobs(n_samples=5000, centers=[[4,4],[-2,-1],[2,-3],[1,1]],cluster_std=0.9)

plt.scatter(X[:,0], X[:,1], marker='.')

#parameters of kmeans class of sklearn.cluster are init which is the method of selecting initial centroids, n_clusters which is the number of centroids/clusters, and n-init which is the number of times the algorithm will run with different centroid points

k_means=KMeans(n_clusters=4,init="k-means++",n_init=12)

k_means.fit(X)

#get labels for each point
k_means_labels=k_means.labels_
print(k_means_labels)

#get coordinates of centroids
k_means_cluster_centers=k_means.cluster_centers_
print(k_means_cluster_centers)

# plotting
fig=plt.figure(figsize=(6,4))
# Colors uses a color map, which will produce an array of colors based on the number of labels there are. We use set(k_means_labels) to get the unique labels.
colors=plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))
ax=fig.add_subplot(1,1,1)
# For loop that plots the data points and centroids.k will range from 0-3, which will match the possible clusters that each data point is in.
for k, col in zip(range(len([[4,4],[-2,-1],[2,-3],[1,-1]])),colors): 
	# Create a list of all data points, where the data poitns that are in the cluster (ex. cluster 0) are labeled as true, else they are    labeled as false.
	my_members=(k_means_labels==k)
	# Define the centroid, or cluster center.
	cluster_center=k_means_cluster_centers[k]
	# Plots the datapoints with color col.
	ax.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor=col, marker='.')
	 # Plots the centroids with specified color, but with a darker outline
	ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
# Title of the plot
ax.set_title('KMeans')
# Remove x-axis ticks
ax.set_xticks(())
# Remove y-axis ticks
ax.set_yticks(())
# Show the plot
plt.show()

#running KMeans on real data set next
cust_df=pd.read_csv(r'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv')

print(cust_df.head())
print(cust_df.columns)
print(cust_df['Edu'].unique())
print(cust_df['Address'].unique())
#because KMeans is not applicable to categorical variables- because euclidean distance is not meaningful for discrete variables- we need to drop the feature 'Address' which is categorical.
df=cust_df.drop('Address',axis=1)
print(df.head(5))

#next we normalize the data over the std. "Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally.We use StandardScaler() to normalize our dataset."
from sklearn.preprocessing import StandardScaler
X=df.values[:,1:]
X=np.nan_to_num(X)
Clus_dataSet=StandardScaler().fit_transform(X)
print(Clus_dataSet)

#apply Kmeans clustering algorithm to the preprocessed dataset
ClusterNum=3
k_means=KMeans(n_clusters=ClusterNum,init='k-means++',n_init=12)
k_means.fit(X)
labels=k_means.labels_
print (labels)

# assign labels to each row
df['clus_km']=labels
print(df.head(5))
#check centroids by averaging the features in each cluster
print(df.groupby('clus_km').mean())

#check distribution of customers based on age and income through plotting
area=np.pi*(X[:,1])**2
plt.scatter(X[:,0],X[:,3], s=area,c=labels.astype(np.float), alpha=0.5)
plt.xlabel('age',fontsize=18)
plt.ylabel('income',fontsize=16)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(1,figsize=(8,6))
plt.clf()
ax=Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
plt.cla()
ax.set_xlabel('education')
ax.set_ylabel('age')
ax.set_zlabel('income')
ax.scatter(X[:,1],X[:,0],X[:,3],c=labels.astype(np.float))









