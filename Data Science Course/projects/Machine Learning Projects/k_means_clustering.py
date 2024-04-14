# from sklearn.datasets import fetch_california_housing
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Load the California Housing dataset
# california_housing = fetch_california_housing()
# X = california_housing.data

# # Apply K-means clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X)

# # Plot the clusters
# plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=20)
# plt.title('K-means Clustering of California Housing Dataset')
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.colorbar(label='Cluster')
# plt.show()


#importing libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#Import the Dataset
df = pd.read_csv('https://raw.githubusercontent.com/ShapeAI/Data-Analysis-and-Machine-Learning/main/Clustering%20Customer_segmentaion/Mall_Customers.csv')

#Data Analysis
print(df.head())
print(df.tail())
print(df.shape())
print(df.isnull().sum())
print(df.info())
print(df.describe())
plt.figure(figsize=(20,15))
sns.Countplot(data=df,x='Age')

#show How many % of male & Female visits the mall with the help of a plot

print(df.Gender.value_counts().plot(kind='pie',autopct='%.2f%%'))




