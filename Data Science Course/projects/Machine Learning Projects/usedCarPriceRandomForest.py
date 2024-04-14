#Random Forest Project
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
from sklearn.ensemble import RandomForestRegressor
#importing the dataset
df = pd.read_csv('cardekho_data.csv')

#EDA
print(df.head)
print(df.columns)
print("Fuel_Type:",df.Fuel_Type.unique())
print("Seller_Type:",df.Seller_Type.unique())
print("Transmission:",df.Transmission.unique())
print("Owner:",df.Owner.unique())
print(df.describe())
df['Current_year'] =2024
df['No_of_years'] = df['Current_year']-df['Year']

#Drop the columns carName,year,Current_year
df.drop(['Car_Name','Year','Current_year'],axis=1,inplace=True)
print(df.columns)

#Convert data from Categorical form to Numerical form
from sklearn.preprocessing import LabelEncoder
df['Fuel_Type'] = LabelEncoder().fit_transform(df['Fuel_Type'])
df['Seller_Type'] = LabelEncoder().fit_transform(df['Seller_Type'])
df['Transmission'] = LabelEncoder().fit_transform(df['Transmission'])
df['Owner'] = LabelEncoder().fit_transform(df['Owner'])
print(df.head())
print(df['Seller_Type'])
print(df['Owner'])
print(df['Transmission'])
#we can also use get_dummies() method to convert from categorial data into numerical data
#get_dummies() works similar to One_Hot_Encoders ie., it creates a separate column for each insight
#pairplot
print(sns.pairplot(df))
# plt.show()
#heatmap
print(sns.heatmap(df.corr(),annot=True,cmap='RdYlGn'))
# plt.show()

#Splitting the data
X = df.drop('Selling_Price',axis=1)
y = df.Selling_Price
print(y)

#Feature Selection(Feature Importance)
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
feat_imp = model.fit(X,y)
print(feat_imp.feature_importances_)
imp = pd.Series(feat_imp.feature_importances_,index = X.columns)
imp.nlargest(5).plot(kind = 'barh') 

#Splitting the data
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=0)

#Model selection
regressor =  RandomForestRegressor()

#Hyper parameter Tunning
n_estimators = [int(i) for i in np.linspace(start=100,stop=1200,num=12)]
max_features =['auto','sqrt']
max_depth = [int(i) for i in np.linspace(start=5,stop=30,num=6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]
random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf
               }
print(random_grid)

from sklearn.model_selection import RandomizedSearchCV
rf_regressor = RandomizedSearchCV(estimator=regressor,param_distributions=random_grid,scoring='neg_mean_squared_error',cv=5,verbose=2,random_state=42,n_jobs=1)

#Training the model
rf_regressor.fit(X_train,y_train)
print(rf_regressor.best_params_)
y_pred = rf_regressor.predict(X_test)
plt.scatter(y_test,y_pred)
plt.show()
finaldf = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(finaldf)
sns.heatmap(finaldf.corr(),annot=True,cmap= 'Greens')

#Performance/Accuracy of the model
print("r2_score:")
print(r2_score(y_test,y_pred))
