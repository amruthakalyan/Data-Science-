'''
Unsupervised Learning two types:
1. Clustering 
2. Association --> Apriori Algorithm and ECLAT algorithm
'''

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd



# Sample transaction dataset
dataset = [
    ['milk', 'bread', 'biscuit'],
    ['bread', 'biscuit', 'eggs'],
    ['milk', 'biscuit', 'cornflakes'],
    ['bread', 'biscuit'],
    ['bread', 'milk'],
    ['bread', 'milk', 'biscuit', 'eggs'],
    ['bread', 'milk', 'eggs'],
    ['milk', 'biscuit', 'eggs'],
    ['bread', 'milk']
]

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset, columns=['item1', 'item2', 'item3', 'item4'])

# Encode the items as binary values
df_encoded = pd.get_dummies(df)

# Apply the Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)