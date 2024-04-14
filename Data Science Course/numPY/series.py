import pandas as pd
import numpy as np
data = np.array([1,2,3,4,5])
s = pd.Series(data)
print(s)

#Creating series using dictionary
data = {"a":11, "b":12, "c":13, "d":14, "d":15}
s = pd.Series(data)
print(s)
s = pd.Series(data,index=['f','g','h','i','j'])
print(s)

#CRUD
#Creating a series using Scalar Value
data = 99
s = pd.Series(data , index =[101,102,103,104,105])
print(s)

#Update
data = np.array([1,2,3,4,5])
s = pd.Series(data)
print(s)
print(s[4])
print(s[:3])
print(s[2:])
data = {"a":11, "b":12, "c":13, "d":14, "e":15}
s = pd.Series(data)
print(s)
print(s['a':'c'])
print(s[['c','d']])

#Boolean Masking
1 - True
0 - False
print(s>13)
print(s[s>13])

