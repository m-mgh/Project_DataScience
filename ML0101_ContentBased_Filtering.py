# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:16:42 2020

@author: tutu_
"""
import zipfile
import requests
from io import BytesIO
import pandas as pd
pd.set_option('display.expand_frame_repr',False)
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#get the zip file from the website and either extract it in a specific location or just get the content
# z=requests.get('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip')

# with zipfile.ZipFile(BytesIO(z.content)) as zf:
# 	zf.extractall('C:\\Users\\tutu_\\OneDrive\\Desktop\\Data_Science\\IBM_ML\\zf')

# zf=zipfile.ZipFile(BytesIO(z.content))

movies_df=pd.read_csv(r'C:\Users\tutu_\OneDrive\Desktop\Data_Science\IBM_ML\zf\ml-latest\movies.csv')
print(movies_df.head(5))
print(movies_df.columns)
print(movies_df['title'].unique())
rating_df=pd.read_csv(r'C:\Users\tutu_\OneDrive\Desktop\Data_Science\IBM_ML\zf\ml-latest\ratings.csv')
print(rating_df.columns)

#Use regular expression to find the year in the titles but avoid including the year that is in some movies' title.
movies_df['year']=movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
#take out the parantheses first
movies_df['year']=movies_df.year.str.extract('(\d\d\d\d)', expand=False)
#take out 'year' info from 'title' feature in movies_df and make it an independent feature.
movies_df['title']=movies_df.title.str.replace('(\(\d\d\d\d\))','')
#get rid of any ending whitespace character that might have been left from applying the previous steps
movies_df['title']=movies_df['title'].apply(lambda x: x.strip())
print(movies_df.head(5))

#split Genres column into a list geners. Geners are separated by | in data
print(movies_df['genres'].unique)
movies_df['genres']=movies_df.genres.str.split('|')
print(movies_df['genres'].head())
#content-based recommendation system techniques don't work well with lists so convert genres list into vector. For this purpose convert genres list into ventors corresponding to each gener type and value assigned to each case per movie is either 0 or 1
#copy movie_df into a new one with genres vector
moviesWithGenres_df=movies_df.copy()
#for every row in the original data fram iterate through the list of genres and place 1 into corresponding columns
for index,row in movies_df.iterrows():
	for genre in row['genres']:
		moviesWithGenres_df.at[index,genre]=1
#fill NaN values with 0	next
moviesWithGenres_df=moviesWithGenres_df.fillna(0)	
print(moviesWithGenres_df['Comedy'].head(5))
#remove timestamp column from ratings_df
print(rating_df.head(5))
rating_df=rating_df.drop('timestamp',1)	
print(rating_df.head(5))

#implement Content-based or item-item recommendation systems.
#start by creating a user input dataframe
userInput=[{'title':'Breakfast Club, The', 'rating':5},{'title':'Toy Story', 'rating':3.5},{'title':'Jumanji', 'rating':2},   {'title':"Pulp Fiction", 'rating':5},            {'title':'Akira', 'rating':4.5}]
inputMovies=pd.DataFrame(userInput)
print(inputMovies)

#Add movieIds from movies_df to userInput dataframe.
#filter out movie titles from movie_db
inputId=movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#merge inputId with inputMovie
inputMovies=pd.merge(inputId,inputMovies)
#drop information we won't use from the dataframe
inputMovies=inputMovies.drop('genres',1).drop('year',1)
print(inputMovies)
#add back 'genres' to the datafram but from the moviesWithGenres data frame that has binary values
userMovies=moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
#reset the index and drop some of the extra columns from the dataframe
userMovies=userMovies.reset_index(drop=True)
userGenreTable=userMovies.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)
print(userGenreTable)
#start learning user's preferences. To do this, assign a weight to each genre. This is done by multiplying user's rating into genre table and summing up the resulting table by column (dot product between a matrix and a vector).
#switch rows with columns in the genres table and use dot product to get weights
userProfile=userGenreTable.transpose().dot(inputMovies['rating'])
print(userProfile)

#extract genre table from the original dataframe
genreTable=moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#drop unnessary information
genreTable=genreTable.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)
print(genreTable.head())
#multiply the genres by the weights and then take the weighted average
recommendationTable_df=((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
print(recommendationTable_df.head())
#sort recommendations in descending order
recommendationTable_df=recommendationTable_df.sort_values(ascending=False)
#the final recommendation table
recTable=movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
print(recTable)













					  