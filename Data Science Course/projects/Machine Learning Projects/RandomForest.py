'''
Random Forest is a Ensemble technique and it is also called as Bagging model.
-->In Random forest , the data is divided into Bootstrap Data
-->Bootstrap Data = it is a data taken from original data
it has the following properties:
1: Data is randomly taken from the original data.
2: Bootstrap data can have duplicate data ie., it can select repeated data from the original dataset
3:It is not neccessary that bootstrap data should have entire original data,it can have less amount of data from the original data as well.
'''
#1. importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

#import the dataset
# link of dataset: https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv

df = pd.read_csv("https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv")
print(df.head())

#EDA
print(df.info())

#Assigining the Feature and target
#Splitting the Data - Training and Testing set
X = np.array(df.Temperature.values)
y = np.array(df.Revenue.values)
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.05,random_state=42)

#Choosing the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=10,random_state=42)

#Training the model
model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))

#Testing the model
print("....Y_TEST......")
print(y_test)
print("......Y_PREDICT.....")
y_pred = model.predict(X_test.reshape(-1,1))
print(y_pred)
pred = pd.DataFrame({"Actual":y_test.reshape(-1),"Predicted":y_pred.reshape(-1)})
print(pred.head())
plt.scatter(X_test,y_test,color='red')
plt.scatter(X_test,y_pred,color='green')
plt.xlabel("Temparature")
plt.ylabel('Revenue')
plt.show()
print(sns.heatmap(pred.corr(),annot=True,cmap='Greens'))
plt.show()

#r2_score
print("r2_score:")
print(r2_score(y_test,y_pred))

