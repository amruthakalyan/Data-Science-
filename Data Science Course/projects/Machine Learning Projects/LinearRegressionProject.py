# Project : Predicting the House Price


#importing all the Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import re 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#problem statement
#You have been given a dataset that describes  the houses in the california. now based on the given features ,you have to predict the house prices

#Creating a Dataframe
bostan= "HousingData.csv"
df = pd.read_csv(bostan).dropna()
print(df.head())
# print(df.info())



#EDA(EXPLORATORY DATA ANALYSIS)
#Target variable is MEDV and replace it with PRICE 
df.rename(columns = {'MEDV':'PRICE'}, inplace = True)
print(df['PRICE'].array)
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
#To get all the unique rows in a dataframe
print(df.nunique())
#To check the null values in the dataframe
print(df.isnull().sum())
print(df.describe())
#finding corelation
print(df.corr)

print(plt.figure(figsize=(10,10)))
print(sns.heatmap(data= df.corr(),annot=True,cmap='Greens'))
# print(plt.show())
#print Minimum price in PRICE column
print(df.PRICE.min())
#print Maxmum price in PRICE column
print(df.PRICE.max())
#Find the Standard Deviation
print(df.PRICE.std())



#Machine Learning - Linear Regression

print(df.head())
# X = df.drop("PRICE",axis = 1) #Features in X
# y = df.PRICE #Target in y
# print(X)
# print(y)

#1)=> convert in the form of 2D-array
#since it is in 2D i.e  X.ndim() = 2

X =  np.array( df.drop("PRICE",axis = 1) )
y = np.array(df.PRICE)
print(X)

#2)=> Splitting the data

X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#3)=>Choosing the steps

model = LinearRegression()

#4)=>Fitting/Train the model

print(model.fit(X_train,y_train))

#Intercept value
print(model.intercept_)
#coefficient
print(model.coef_)


#5)=>Prediction 

print("y-test")
print(y_test)
print("y-predict")
y_pred = model.predict(X_test)
print(y_pred)

#6)=>Testing the model performance
print("Model Score:\n")
print(model.score(X_test,y_test))

#R2_squred
print("R2_squred error:\n")
print(r2_score(y_test,y_pred))

#Adjusted r2_srquared



#MSE
print("mean_squared_error:\n")
print(mean_squared_error(y_test,y_pred))

#MAE
print("mean_absolute_error:\n")
print(mean_absolute_error(y_test,y_pred))

print(plt.scatter(y_test,y_pred))
print(plt.show())
print(plt.xlabel("Actual Price"))
print(plt.ylabel("Predicted Price"))
print(plt.title("Actual Price v/s Predicted Price"))
print(plt.show())
