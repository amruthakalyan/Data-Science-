# importing the libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
iris = load_iris().data
df = pd.DataFrame(iris, columns=['sepal length','sepal width','petal length','petal width'])

print(df.head())

#Features
print("Features:")
print(load_iris().feature_names)
#Target
print("Target:")
print(load_iris().target_names)
print(df.head())
print(df.describe)

print(sns.pairplot(df,height=3))
print(plt.show())

#Train test split
X = load_iris().data
y = load_iris().target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42 )

#Select the model
model = LinearRegression()

#Training the model
model.fit(X_train,y_train)

#Testing the model
print("Y_Test")
print(y_test)
print("Y_Prediction")
y_pred = model.predict(X_test)
print(y_pred)

# performance

print("Model Score")
print(model.score(X_test,y_test))

#r2_error
print("r2_score:")
print(r2_score(y_test,y_pred))

#mean_absolute_error
print("mean_absolute_error:")
print(mean_absolute_error(y_test,y_pred))

#mean_squarred_error
print("mean_squarred_error:")
print(mean_squared_error(y_test,y_pred))