#1. importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.datasets import load_iris

#importing the dataset
data = load_iris()
df = pd.DataFrame(data.data)
