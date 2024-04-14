#Logistic Regression -Titanic Servival Dataset project
'''
1.Import libraries
2.import the dataset
3.Perform Data analysis(DM,DE,DC,DV,EDA)
4.Feature Scaling [optional]
5.Encoding
6.Feature Selection
7.Choosing the model -Logistic Regression
8.Split the data-CV
9.Training the model
10.Test the model
11.Performance-Confusion metric.
'''
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import confusion_matrix

#imort dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

#Data Analysis
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.isnull().sum())
#heatmap for null values
plt.figure(figsize=(20,20))
sns.heatmap(df.isnull(),annot=True,cmap='Greens')
# plt.show()
#Age % of null values
print((df.Age.isnull().sum()/len(df.Age))*100)

#Cabin % of null values
print((df.Cabin.isnull().sum()/len(df.Cabin))*100)

print(df.Cabin.unique())

#Drop the column Cabin since we cannot predict the NaN values exactly
df.drop('Cabin',axis=1,inplace=True)
print(df.columns)

#Different Categories in Embarked feature
print(df.Embarked.unique())

#Show the row where the Embarked is null
print(df[df.Embarked.isnull()])

print(df[['Pclass','Embarked']])

#Drop the rows where Embarked is NaN
# df.dropna()

#Mean of Age column
print('Mean:',df.Age.mean())
#Mode of Age column
print('Mode:',df.Age.mode())
#Median of Age column
print('Median:',df.Age.median())

#Plot a boxplot to find out the outliers int the Age Column
print(sns.boxplot(df.Age))

#Fill all the null values in teh Age column with its median value
df.Age.fillna(value=df.Age.median(),inplace=True)

#Now we can Drop the rows where Embarked is NaN
df.dropna(inplace=True)
print(df.info())
print(df.isnull().sum())
print(df.shape)

#Drop off the PassengerId,Name,Ticket
df.drop(columns=['PassengerId','Name','Ticket'],inplace=True,axis=1)
print(df.columns)

#Plot a graph :Strength of Male V/s Strength of Female
df.Sex.value_counts().plot.bar(df.Sex)
plt.grid()

#Plot a graph :Strength of Survived V/s Strength of not Survived
df.Survived.value_counts().plot.bar(df.Survived)
plt.grid()

#Plot a graph to find out the survival and non-survival w.r.t Sex
print(sns.countplot(x='Survived',data=df,hue='Sex'))

#Plot a graph to find out the strength of the Pclass
print(df.Pclass.value_counts().plot.bar(df.Pclass))


#Plot a graph to find out the survival and non-survival w.r.t Pclass
print(sns.countplot(x='Survived',data=df,hue='Pclass'))

#Change the dtype of Age Column
df['Age'] = df['Age'].astype('int')

#Chnage the Fare upto 2 decimal places
df['Fare'] = round(df['Fare'],2)


#Encoders - To convert data into numerical form

from sklearn.preprocessing import LabelEncoder
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
print(df['Sex'])
print(df.info())

#One Hot encoding for the column Embarked
newdf = df.copy()
new = df.copy()
# newdf = pd.get_dummies(newdf['Embarked'])
df =pd.concat([df,pd.get_dummies(df['Embarked'])],axis=1)
print(df)
# df = df.astype(int)
df.S= df['S'].astype('Int32')
df.Q= df['Q'].astype('Int32')
df.C= df['C'].astype('Int32')
print(df.head())
df.drop(['Embarked','C'],axis=1,inplace=True)
print(df.columns)

#Feature Importance/Featuree selection
X = df.iloc[:,1:]
y = df.iloc[:,0]

print('X:',X)
print('y:',y)

from sklearn.ensemble import ExtraTreesClassifier
feat = ExtraTreesClassifier()
feat.fit(X,y)
print(feat.feature_importances_)
feat_imp = pd.Series(feat.feature_importances_,index=X.columns)
feat_imp.nlargest(5).plot(kind='barh')

#Splitting the data
# from sklearn.model_selection import cross_val_score
# model = LogisticRegression()
skf = StratifiedKFold(n_splits=5)

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Model Selection
Classifier = LogisticRegression()
#Train the model
Classifier.fit(X_train,y_train)

#Test the model
y_pred = Classifier.predict(X_test)

#EDA
final = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print(final.head())

sns.heatmap(final.corr(),annot=True,cmap='Greens')

#Performance metric - Confusion matrix

print('Confusion_matrix:',confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy_score:",accuracy_score(y_test,y_pred))



#Exportation of model & dataset
#(Serialization-Deserialisation)
# import pickle
# pick = pickle.dump(Classifier)

# unpickle = pickle.load(pick)

