import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Problem Statement
df = pd.DataFrame(load_diabetes().data)
df.columns = load_diabetes().feature_names
print(df.head())
print("Target Values:")
print(load_diabetes().target)

#Exploratory Data Analysis
#To check the null values in the dataframe
print(df.isnull().sum())
print(df.describe())
#finding corelation
print(df.corr)
print(plt.figure(figsize=(10,10)))
print(sns.heatmap(data= df.corr(),annot=True,cmap='Greens'))
# print(plt.show())
print(df.describe)



#Machine Learning-Linear Regression
print("X")
X = np.array( df.drop("s6",axis = 1) )
print(X)
print("y")
y = np.array(df.s6)
print(y)


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

print("Model Score")
print(model.score(X_test,y_test))

#R2_squred
print("R_squred")
print(r2_score(y_test,y_pred))

#Adjusted r2_srquared



#MSE
print("MSE")
print(mean_squared_error(y_test,y_pred))

#MAE
print("MAE")
print(mean_absolute_error(y_test,y_pred))

