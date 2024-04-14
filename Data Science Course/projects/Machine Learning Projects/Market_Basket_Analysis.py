'''
This is a Market-Basket-Analysis Project
(Unsupervised Learning --> Assocition-->Apriori Algorithm)
''' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori


#Link: https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv

df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv")

#Data Analysis
print(df.head())

#info() -->information
print(df.info())

#Shape() -->rows and cols
print(df.shape)

#size() -->Total no.of elements in dataset
print(df.size)

#check null values -->isna()
print(df.isna().sum())

#most selling items
print(df.itemDescription.value_counts().head(10))

#bargraph
df.itemDescription.value_counts().head(10).plot.bar()

#Least Selling items
print(df.itemDescription.value_counts().tail(10).sort_values())

#bargraph
df.itemDescription.value_counts().tail(10).sort_values().plot.bar()

#Top 10 customers
print(df.Member_number.value_counts().head(10))

#Bargraph
df.Member_number.value_counts().head(10).plot.bar()


#Create a new columns year ,month,day
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day
print(df.head())
#which year the majority of the transactions happened
print(df.Year.value_counts())
print(df.Month.value_counts())
print(df.Day.value_counts())

#Apriori Implementation
data = df.copy()

data = pd.get_dummies(data['itemDescription'])
data.drop(['itemDescription'],xis=1,inplace=True)


