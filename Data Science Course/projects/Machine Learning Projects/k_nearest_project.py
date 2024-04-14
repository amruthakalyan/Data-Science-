#Movie Rating and Recomondaton Project (k-Nearest Neighbour)

'''
1.Importing Libraries
2.Importing the dataset
3.Data Analysis - DE,DM,DC,DV,EDA
4.Feature Engineering-Encoders,Feature Scaling
5.Splitting the data into two sets using the CV
6.Model Selection-KNN
7.Training the model
8.Test the model
9.Performance -Confusion Metric
'''

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import StandardScaler

#import dataset

Credits = pd.read_csv('tmdb_5000_credits.csv')
Movies = pd.read_csv('tmdb_5000_movies.csv')

#3.Data Analysis - DE,DM,DC,DV,EDA
print(Movies.head())  
print(Credits.head())
print(Movies.describe())
print(Movies.columns)
print(Movies['genres'][0])
print(re.findall('name',Movies.genres[0]))
# for i ,k in zip(Movies.genres,range(len(Movies.genres))):
#     Movies.genres[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]

print(Movies.isnull().sum())

#Drop the column homepage
Movies.drop('homepage',axis=1,inplace=True)
#Drop the column tagline
Movies.drop('tagline',axis=1,inplace=True)
print(Movies.columns)
Movies.dropna(inplace=True)

print(Movies.isnull().sum())

# Movies = pd.get_dummies(Movies.release_date)
# print(Movies.columns)
Movies[['Release_year', 'Release_month', 'Release_day']] = Movies['release_date'].str.split('-', expand=True)
print(Movies.columns)
print(Movies['Release_year'])


#Credits Dataset
print(Credits.head())
print(Credits.isnull().sum())


#merge two datasets based on id and movie_id
Movies =  pd.merge(Movies, Credits, left_on='id', right_on='movie_id', how='inner')
print(Movies.columns)