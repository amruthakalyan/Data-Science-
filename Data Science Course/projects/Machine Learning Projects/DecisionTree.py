# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.tree import DecisionTreeRegressor

#problem statement
'''In this dataset we have one independent variable 'Temparature' and one dependent variable 'Revenue' Build a DecisionTreeRegressor to study the relationship between two variables and predict the revenue of the iceCream shop based on the temparature on a particular day.'''
#import the dataset
# link of dataset: https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv

df = pd.read_csv("https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv")


#EDA
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())

plt.scatter(df.Temperature,df.Revenue)
plt.xlabel("Temparature")
plt.ylabel("Revenue")
plt.title("Temparature V/s Revenue")
plt.show()
# sns.heatmap(df.corr ,annot=True ,cmap='Greens')

#Machine Learning
#Splitting the Data - Training and Testing set
X = np.array(df.Temperature.values)
y = np.array(df.Revenue.values)
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

#choosing model
regressor = DecisionTreeRegressor()

#Training the model
regressor.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))

#Testing the model
y_pred = regressor.predict(X_test.reshape(-1,1))

#Comparing the y_test with y_pred
comp = pd.DataFrame({"Actual Values":y_test.reshape(-1)},
{"Predicted Values":y_pred.reshape(-1)})
print(comp)

plt.scatter(y_test.reshape(-1),color='red')
plt.scatter(y_pred.reshape(-1),color='green')
plt.xlabel("X_test")
plt.ylabel("y_test/y_pred")
plt.show()

#Performance
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print("r2_score:")
print(r2_score(y_test.reshape(-1),y_pred.reshape(-1)))
print("Mean squarred error")
print(mean_squared_error(y_test.reshape(-1),y_pred.reshape(-1)))
print("Mean absolute Error:")
print(mean_absolute_error(y_test.reshape(-1),y_pred.reshape(-1)))


