# SVM(SVR) Prediction of Baltimore Salary

#Problem Statement:
#You r given a dataset which captures the salary from july 1st ,2013 through june 30th,2014. It includes only those employees who are employedd on june 30,2014.Predict the salary of Employees working in baltimore.

#1. importing the libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# 2. importing the Dataset
df = pd.read_csv('train.csv')
#first 5 rows
print(df.head())
#info about dataset
print(df.info())
#last 5 rows
print(df.tail())
#columns names
print(df.columns)
#no.of rows and columns
print(df.shape)
#statistical info
print(df.describe())
#Removing trailing spaces of the column names
df.columns = df.columns.str.strip().str.lower()
print(df.columns)
#checking for null values
print(df.isnull().sum())
#delete grosspay column
df = df.drop(columns=['grosspay'])
print(df.columns)
#value_counts  for agencyid
print(df.agencyid.value_counts())
print(df.agencyid.value_counts().map(df.agency))
print(df.jobtitle.value_counts())
print(df.annualsalary.value_counts())
#remove all the $ from annualsalary and change the dtype to integer
df['annualsalary'] = df['annualsalary'].str.replace('$', '').astype(float)
print(df['annualsalary'])
# Split 'hiredate' into 'hireday', 'hiremonth', and 'hireyear'
df[['hireday', 'hiremonth', 'hireyear']] = df['hiredate'].str.split('/', expand=True)
# Remove the original 'hiredate' column
df.drop(columns=['hiredate'], inplace=True)
#keep the same column_order
column_order = ['name','jobtitle','agencyid','agency', 'hireday', 'hiremonth', 'hireyear', 'annualsalary']
df = df[column_order]
print(df.columns)
#boxplot for the annual salary
print(df.annualsalary.plot.box())
plt.title('Boxplot for Annual Salary')
plt.show()
#plot top 10 jobs based on the hiring
print(df.groupby(['jobtitle'])['name'].count().sort_values(ascending=False).head(10).plot.bar())
print(plt.show())

#plot top 10 jobs with highest salary
print()

#plot top 10 agency's ID that has highest no.of employes
print(df.groupby(['agencyid'])['name'].count().sort_values(ascending=False).head(10).plot.bar())
print(plt.show())

#plot highest salary v/s year


#plot avg salary v/s year

#plot the graph to check on which month most of the people hired

#plot a pairplot
print(sns.pairplot(df,size=3))
print(plt.show())
# #plot a heatmap
# print(sns.heatmap(df.corr(),annot=True))
# print(plt.show())



#Machine Learning
# 1) Train Test Split
X = np.array(df.drop('annualsalary',axis=1))
y = np.array(df.annualsalary)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# 2) Data Preprocessing - Feature scaling
# 3) choose the model -SVR model
svr = SVR()
# 4) Train the model
svr.fit(X_train,y_train)
y_pred = svr.predict(X_test)
# 5) Test the model
svr.predict(X_test)
target = pd.DataFrame({"actual":y_test,"Predicted":y_pred})
print(".....Predicting.....")
print(target.head())
# 6)Performace matrices
# 7)conclusion - if an employee joins on the following date: 02/09/2018, 04/04/2015,21/12/2021 ,predict the salaries.
