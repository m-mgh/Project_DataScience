# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import requests
import zipfile
from io import BytesIO
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# #get the zip folder from the online source and extract its contents in a specified location
# z=requests.get("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip")

# with zipfile.ZipFile(BytesIO(z.content)) as zf:
# 	zf.extractall('C:\\Users\\Mercedeh_Mgh\\OneDrive\\Desktop\\Data_Science\\IBM_ML\\zf')
# 	

movies_df=pd.read_csv(r'C:\Users\Mercedeh_Mgh\OneDrive\Desktop\Data_Science\IBM_ML\zf\ml-latest\movies.csv')
ratings_df=pd.read_csv(r'C:\Users\Mercedeh_Mgh\OneDrive\Desktop\Data_Science\IBM_ML\zf\ml-latest\ratings.csv')

print (movies_df.head(5))
print (ratings_df.head(5))

#clean up dataframes using pandas extract function
movies_df['year']=movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year']=movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title']=movies_df.title.str.replace('(\(\d\d\d\d\))','')
#apply the strip function to get rid of any whitespace etc
movies_df['title']=movies_df['title'].apply(lambda x: x.strip())
#also drop the geners column
movies_df=movies_df.drop('genres',1)
print(movies_df.head(5))

#from ratings_df also drop timestamp
ratings_df=ratings_df.drop('timestamp',1)
print(ratings_df.head(5))

#start creating collaborative_filtering or user-user filtering by finding similarity of users to the input user via pearson correlation function
# The process for creating a User Based recommendation system is as follows:
# - Select a user with the movies the user has watched
# - Based on his rating to movies, find the top X neighbours 
# - Get the watched movie record of the user for each neighbour.
# - Calculate a similarity score using some formula
# - Recommend the items with the highest score

#create an input user to recommend movies to
userInput=[{'title':'Breakfast Club, The','rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}]
inputMovies=pd.DataFrame(userInput)
# extract movie ids from movies_df and add them to user input
#first filtering out the movies by titles in inputMovies from movies_df and merging the info with the user input. Then drop info we don't need
inputId=movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies=pd.merge(inputId,inputMovies)
inputMovies=inputMovies.drop('year',1)
print(inputMovies)

#get a subset of users that have watched the same movies in out userinput
userSubset=ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userSubset)
#group data in userSubset by userIds
userSubsetGroup=userSubset.groupby(['userId'])
print(userSubsetGroup.get_group(1130))
#sort data groups in userSubset so that users with most movies in common with the input get priority
userSubsetGroup=sorted(userSubsetGroup, key=lambda x:len(x[1]), reverse=True)
print(userSubsetGroup[0:3])
#use Pearson Correlation Coefficient to get the linear association/similarity of users to user input. PCC is a good measure of correlation here because it is invarient to scaling/multiplying or adding constants to variables, which makes it possible to calculate similarity for users with different absolute rating values but who are in fact similar on a different scale.
#first limit the subset to a smaller group for the sake of saving calculation time
userSubsetGroup=userSubsetGroup[0:100]
#calculate pcc between input user and subset group and store it in a dictionary, where the key is the userid and the value is the coefficient
pearsonCorrelationDict={}
for name,group in userSubsetGroup:
	group=group.sort_values(by='movieId')
	inputMovies=inputMovies.sort_values(by='movieId')
	#get the number of ratings for the formula
	nRatings=len(group)
	#get the review scores for the shared movies
	temp_df=inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
	#store them in a temporary variable buffer as a list to facilitate future calculations
	tempRatingList=temp_df['rating'].tolist()
	#also put the user group in a list format
	tempGroupList=group['rating'].tolist()
	#calculate pearson correlation coefficient between user input and other subset group
	Sxx=sum([i**2 for i in tempRatingList])-pow(sum(tempRatingList),2)/float(nRatings)
	Syy=sum([i**2 for i in tempGroupList])-pow(sum(tempGroupList),2)/float(nRatings)
	Sxy=sum(i*j for i,j in zip(tempRatingList,tempGroupList))-sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
	#if denominator is zero then correlation is 0
	if Sxx!=0 and Syy!=0:
		pearsonCorrelationDict[name]=Sxy/sqrt(Sxx*Syy)
	else:
		pearsonCorrelationDict[name]=0
		
print(pearsonCorrelationDict.items())
pearsonDF=pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns=['similarityIndex']
pearsonDF['userId']=pearsonDF.index
pearsonDF.index=range(len(pearsonDF))
print(pearsonDF.head())

#get top 50 users who are most similar to the input user
topUsers=pearsonDF.sort_values(by='similarityIndex',ascending=False)[0:50]
print(topUsers.head())
	
#start recommending movies to the input user
##get the name of the movies and correspond them with the weighted rating of each top user
topUsersRating=topUsers.merge(ratings_df,left_on='userId',right_on='userId',how='inner')
print(topUsersRating.head())	

#multiply the movie ratings by their weights (ie. the similarity index), then get the average weighted ratings by summing up the weighted ratings and deviding by the sum of the weights
topUsersRating['weightedRating']=topUsersRating['similarityIndex']*topUsersRating['rating']
print(topUsersRating.head())

tempTopUsersRating=topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns=['sum_similarityIndex','sum_weightedRating']
print(tempTopUsersRating.head())
recommendation_df=pd.DataFrame()
recommendation_df['weighted average recommendation score']=tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId']=tempTopUsersRating.index
print(recommendation_df.head())
recommendation_df=recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))
recommended_movies=movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
print(recommended_movies.head())






