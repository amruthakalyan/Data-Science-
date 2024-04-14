'''
--Support Vector Machine(SVM) is used in Classification
--SVM is used to classify the data points into classes using HyperPlane
--HyperPlane - It is a line that separates two or more classes with equal distances from the best fit line
--Kernal - It is a function that converts the lower Dimentional dataPoints into Higher Dimensional dataPoints.
--kernals -1)Polynomial kernal
           2)Sigmoid kernal
           3)Gaussian kernal
           4)rbf kernal
---SVM -- 2 parts
        1)SVR
        2)SVC
1)SVR and SVC
  -- Collect the Training set
  --Choose the kernal(perform the HPT)
  --Model
  --Train the model
  --Performance
'''
#importing Libraries
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
# In this Data ,we are Having one independent variable(Feature---"Hours of Study") and a dependent variable(Target--"Marks").You have to train a SVR model to understand the relationship between the Hours of study and the marks of the student to predict the student marks based on their no.of hours dedicated to their studies.

#importing the DataSet
df = pd.read_csv("https://raw.githubusercontent.com/mk-gurucharan/Regression/master/SampleData.csv")


#EDA
print("Head")
print(df.head())
print("Tail")
print(df.tail())
#check statistical values
print(df.describe())
#check if any null values
print(df.info())
#Scatter Plot

print(plt.scatter(df['Hours of Study'], df['Marks']))
print(plt.xlabel("Hours of study"))
print(plt.xlabel("Marks"))
print(plt.title('Hours of Study V/s Marks'))
print(plt.show())

from sklearn.preprocessing import StandardScaler
#iloc()- function is used to get the rows in a dataframe
X = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values
stanscale = StandardScaler()
X = stanscale.fit_transform(X.reshape(-1,1))
y = stanscale.fit_transform(y.reshape(-1,1))
print(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)
print(X_train.shape)

from sklearn.svm import SVR
model = SVR(kernel='rbf')
print(model.fit(X_train,y_train))
y_pred = model.predict(X_test)
print(y_pred)
y_pred = stanscale.inverse_transform(y_pred)
print(y_pred)
y_test = stanscale.inverse_transform(y_test)
print(y_test)
print(plt.scatter(y_test,y_pred))
print(plt.xlabel("Actual marks"))
print(plt.ylabel("Predicted marks"))
print(plt.title("Actual marks V/s Predicted marks"))
# print(plt.show())
print(model.score(X_test,y_test))
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
