#Decision tree classifier and Random Forest Classifier(Credit card fraud Detection)




#Steps
'''
1.Import Libraries
2.Import Dataset
3.Perform Data Analysis(DM,DC,DE,DV,EDA)
4.Data Preprocessing-Feature Engineering(Encoders,Scaling,Feature Importance/Selection,Hyper Parameter Tunning ,etc.)
5.Splitting of data into sets-CV
6.Model Selection
7.Train the model
8.Test the model
9.Performance metric:Condusion matrix,accuracy_score
'''



#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.model_selection import KFold,StratifiedKFold,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier


#Import Dataset
#Link : https://drive.google.com/file/d/1Q-gkb2MTSiRxn_MLoLHIA4AmRsGJ5Aiv/view?usp=sharing

df = pd.read_csv('https://drive.google.com/file/d/1Q-gkb2MTSiRxn_MLoLHIA4AmRsGJ5Aiv/view?usp=sharing')

#Data Analysis
print(df.shape)
print(df.isnull().sum())
print(df.describe())

print(df.Class.value_counts())

sns.heatmap(df.corr(),annot=True,cmap='Greens')

#Feature importance/Selection
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

model = ExtraTreesClassifier()

print(model.fit(X,y))
print(model.feature_importances_)

#Feature Selection based on high impact on target value
feat = pd.Series(model.feature_importances_,index=X.columns)
feat.nlargest(18).plot(kind='barh')

cols =['V17','V14','V12','V10','V11','V16','V18','V9','V4','V3','V7','V21','V1','V26','Time','V2','V19','V8']

X_new = X[cols]

print(X_new.columns)

#Splitting the data
skf = StratifiedKFold(n_splits=10)

for train_index,test_index in skf.split(X,y):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]


for train_index,test_index in skf.split(X,y):
    X_new_train,X_new_test = X_new.iloc[train_index],X_new.iloc[test_index]
    y_new_train,y_new_test = y.iloc[train_index],y.iloc[test_index]

print(X_train.shape)
print(X_new_train.shape)

#Model Selection
decision = DecisionTreeClassifier()
randomf = RandomForestClassifier()

#Hyper parameter tuning for rfc
n_estimators = [int(i) for i in np.linspace(100,2000,20)]
max_features = ['auto','sqrt','log2']
max_depth = [int(i) for i in np.linspace(5,100,20)] 
min_samples_split = [int(i) for i in range(2,101,2)]
min_samples_leaf = [int(i) for i in range(1,11)]

parameters = {
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf ':min_samples_leaf 
}

print(parameters)

rf_model = RandomizedSearchCV(estimator=randomf,param_distributions=parameters,scoring='neg_mean_squared_error',n_jobs=1,cv=5,verbose=2,random_state=42)

print(rf_model.fit(X_train,y_train))

y_pred = randomf.predict(X_test)

#Accuracy
print(accuracy_score(y_test,y_pred))


