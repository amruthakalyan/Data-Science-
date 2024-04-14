import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
df = pd.read_json('technology_JSON')
data = df.copy()
print(data)
#manupulate the data in such a way that all the coloumns become rows and vice versa
data = data.T
print(data.head())
print(data.info())
print(data.columns)


