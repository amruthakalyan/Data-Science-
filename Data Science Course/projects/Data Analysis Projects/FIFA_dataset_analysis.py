import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re

url ="https://en.wikipedia.org/wiki/List_of_Academy_Award-winning_films"
req = requests.get(url)
print(req)
soup = BeautifulSoup(req.content)
# print(soup.prettify())
arr =[]
# for i in soup.findAll('tr'):
#   arr.append(i) 
# print(arr[1])

for i in soup.findAll('td'):
  arr.append(i)
print(arr[0])  
print(arr[1])  
print(arr[2])  
print(arr[3])  
print(arr[4])  

