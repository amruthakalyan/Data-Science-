#Cancer_Dataset

#Importing the Libraries
import numpy as np
import pandas as pd

#Import dataset
df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Types-Of-Cross-Validation/main/cancer_dataset.csv')
print(df.head())
print(df.columns)
df.drop('Unnamed: 32',axis=1,inplace=True)
X = df.iloc[:,2:]
y = df.iloc[:,1]
print('X:',X)
print('y:',y)
print(X.describe())
print(X.info())
print(df.isna().sum())


'''
train_test_split()-does not provide an accurate target i.e., their is a fluctuation in the accuracy of the model when the randon state is changed..
To avoid this we use CV (Cross Validation) method.
'''
#Train-Test-Split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Select the model
model = LogisticRegression()

#fit the training data
model.fit(X_train,y_train)
result = model.score(X_test,y_test)
print(result)



''''
Cross Validation
types:
1.LooCV - Low one out CV
2.K-fold CV
3.Stratified K-fold CV
4.Time Series CV 
'''

#K-Fold Cross Validation
from sklearn.model_selection import KFold,cross_val_score

k = KFold(10)
model = LogisticRegression()
result = cross_val_score(model, X,y, cv=k)
print(result)
#Overall Accuracy
print('K_FOLD:',np.mean(result))



#Stratified K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5)
result = cross_val_score(model , X,y, cv=skfold)
print(result)
print('SK_FOLD:' ,np.mean(result))



#LOOCV

'''
LOOCV takes lots of time to procees entire data
'''
from sklearn.model_selection import LeaveOneOut
lc = LeaveOneOut()
result = cross_val_score(model, X,y, cv=lc)
print('LOOCV:' ,np.mean(result))