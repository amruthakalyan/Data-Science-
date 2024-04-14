# importing all the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
# Assiging the dataset to the Dataframe  
var_x = np.array([1.1,1.3,1.5,2.0,2.2,2.9,3.0,3.2,3.2,3.7,3.9,4.0,4.0,4.7,4.8,4.9,5.6])
var_y = np.array([39343,46205,37731,43535,39821,56642,60150,54445,64445,57189,63218,55794,56957,57000,58968,58980,60989])
print(len(var_x))
print(len(var_y))
X = var_x.reshape(-1,1)
y = var_y
# print(plt.scatter(var_x,var_y))
# print(plt.show())
df = pd.DataFrame({'Experience':var_x,'salary':var_y})
print(df.head())
print(df.info())
X = var_x
y = var_y
X = df.Experience.values
y = df.salary.values 
# splitting the dataset into Training set and the Testing set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
print(X_train ,"\n",X_test,"\n" ,y_train,"\n",y_test)
# choose the model 
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model on the training set
print(model.fit(X_train,y_train) )
