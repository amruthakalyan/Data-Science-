from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# url = "https://raw.githubusercontent.com/ozlerhakan/mongodb-json-files/master/datasets/grades.json"
# req = requests.get(url)
# print(req)
# df = pd.read_json("https://raw.githubusercontent.com/ozlerhakan/mongodb-json-files/master/datasets/grades.json",lines=True)
# print(df)
# print(df.info)
# print(df["id"])


# 1.Converting the JSON data into a DataFrame
#2.Explore the data
#3.Clean & manupulate the data
#4.Analyse of data
#5.Concludion

# JSON data url:https://raw.githubusercontent.com/ozelerhakan/momgodb-json-files/master/datasets/books.json
# url = "https://raw.githubusercontent.com/ozelerhakan/momgodb-json-files/master/datasets/books.json"
# con = requests.get(url)
# print(con)
#1.converting into dataframe
df = pd.read_json("https://raw.githubusercontent.com/ozelerhakan/momgodb-json-files/master/datasets/books.json",lines=True)
print(df.head())
#print all the coloumns
print(df.coloumns)
#no.of coloums
print(df.shape[1])
#all the statistical info
print(df.describe())
#total no.of null values in each coloumns
print(df.isnull().sum())
print(df.info())

# 3.Data cleaning

#clean ISBN coloumn
#list all the rows in isbn coloumn where ISBN value is null


