#Decision Tree Project
'''
1.Import libraries
2.Import the dataset
3.Perform Data Analysis and EDA
4.Splitting the data
5.[optional] Data Preprocessing-Feature Scaling..
6.Choosing a model-Decision Tree Regressor
7.Training the model
8.Testing the model
9.Checking the performance of the matrix.
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

#2.Importing the dataset
df = pd.read_csv('car_data.csv')
#Perform Data Analysis and EDA
print(df.head())
print(df.tail())
print(df.ndim)
print(df.shape)
print(df.columns)
print(df.describe())
print(df.info())
print(df.fuel.unique())
print(df.seller_type.unique() )
print(df.owner.unique())
print(df.transmission.unique())
#create a column  new_fuel with index 4
print(df.columns)
# x = df.fuel.replace({'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4})
# df.insert(df.columns.get_loc('fuel'),column='new_fuel',value=x) 
# print(df.columns)
# print(df.head())

# y = df.seller_type.replace({'Individual':0,'Dealar':1,'Trustmark Dealar':2})
# df.insert(df.columns.get_loc('seller_type'),column='new_seller_type', value=y)
# print(df.columns)
# print(df['new_seller_type'])
# new_df = df[['fuel','seller_type']]
# print(new_df.head())

#ENCODERS
'''
Two types:
1.One hot Encoders
2.Lable Encoders
#Lable Encoders
from sklearn.preprocessing import LabelEncoder
new_df['fuel'] = LabelEncoder().fit_transform(new_df['fuel'])
'''
#perform LabelEncoders
from sklearn.preprocessing import LabelEncoder
df['fuel'] = LabelEncoder().fit_transform(df['fuel'])
df['seller_type'] = LabelEncoder().fit_transform(df['seller_type'])
df['transmission'] = LabelEncoder().fit_transform(df['transmission'])
df['owner'] = LabelEncoder().fit_transform(df['owner'])
print(df.head())
#create a column no_of_years = currentyear - year
currentyear = 2024
values =currentyear- df.year
print(values)
df.insert(loc=2,column='no_of_years',value=values)
print(df.columns)
print(df['no_of_years'])
#Drop columns name, year
#Rename selling_price to current_selling_price
df.drop(['name','year'], axis=1, inplace=True)
print(df.columns)
df.rename(columns={'selling_price': 'current_selling_price'}, inplace=True)
print(df.columns)
# print(df.corr())
# print(sns.heatmap(df.corr(),annot=True,cmap='Greens'))
# print(sns.pairplot(df))


#Select the features and target
X =np.array(df.drop('current_selling_price',axis=1))
Y =np.array(df.current_selling_price)
print(".....X....")
print(X)
print(".....Y....")
print(Y)

#Splitting the data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)

#Choosing the model
regressor = DecisionTreeRegressor()

#Training the model
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
#Testing the model
target = pd.DataFrame({"actual":Y_test,"Predicted":Y_pred})
regressor.predict(X_test)
print(target.head())
plt.scatter(Y_test,Y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#Performance
print("Model score:")
print(r2_score(Y_test,Y_pred))

